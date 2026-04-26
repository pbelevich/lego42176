#!/usr/bin/env python3
"""
Run a trained ACT policy to drive the LEGO 42176 car into the garage.

Usage:
    source venv_record/bin/activate
    python inference.py <checkpoint_path>

    # Example:
    python inference.py outputs/train/checkpoints/010000/pretrained_model

Controls:
    Enter          - start autonomous driving (model takes over)
    Space          - emergency stop (cancels autonomous mode)
    Arrows / WASD  - manual drive (same as main.py)
    Q / Esc        - quit

The script connects to the LEGO hub via BLE, opens the RTSP camera stream,
loads the trained policy, and shows a pygame window with the camera feed.
Press Enter to let the model drive. Press Space to stop at any time.
"""

import argparse
import asyncio
import os
import platform
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pygame
import torch
from bleak import BleakScanner, BleakClient

# ---------------------------------------------------------------------------
# LEGO BLE constants (from main.py)
# ---------------------------------------------------------------------------
HUB_NAME = "Technic Move"
SERVICE_UUID = "00001623-1212-EFDE-1623-785FEABCD123"
CHAR_UUID = "00001624-1212-EFDE-1623-785FEABCD123"

LIGHTS_ON = 0x00
LIGHTS_BRAKE = 0x01

MAX_SPEED = 80
MAX_STEERING = 80
ACCEL_PER_SEC = 180
DECEL_PER_SEC = 140
STEER_PER_SEC = 300
CENTER_PER_SEC = 240

STEERING_CALIBRATE_1 = bytes.fromhex("0d008136115100030000001000")
STEERING_CALIBRATE_2 = bytes.fromhex("0d008136115100030000000800")

# ---------------------------------------------------------------------------
# Camera / inference configuration
# ---------------------------------------------------------------------------
RTSP_URL = os.environ.get(
    "RTSP_URL",
    "rtsp://user:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=1",
)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15            # must match the dataset FPS the model was trained on
LOOP_HZ = 30        # control loop rate


# ---------------------------------------------------------------------------
# Helpers (from main.py)
# ---------------------------------------------------------------------------
def clamp(x, lo=-100, hi=100):
    return max(lo, min(hi, x))


def move_towards(current, target, max_delta):
    if current < target:
        return min(target, current + max_delta)
    if current > target:
        return max(target, current - max_delta)
    return current


def drive_cmd(speed=0, steering=0, lights=LIGHTS_ON):
    speed = int(clamp(round(speed), -100, 100))
    steering = int(clamp(round(steering), -100, 100))
    return bytearray([
        0x0D, 0x00, 0x81, 0x36, 0x11, 0x51, 0x00, 0x03, 0x00,
        speed & 0xFF,
        steering & 0xFF,
        lights & 0xFF,
        0x00,
    ])


async def find_hub():
    print("Searching for LEGO Technic Move Hub...")

    def match(device, adv):
        name = device.name or adv.local_name or ""
        return HUB_NAME in name

    device = await BleakScanner.find_device_by_filter(
        match, timeout=15.0, service_uuids=[SERVICE_UUID],
    )
    if device is None:
        raise RuntimeError("Hub not found. Turn it on and make sure the LEGO app is closed.")

    print(f"Found: {device.name} / {device.address}")
    return device


async def send_drive(client, speed, steering, lights=LIGHTS_ON):
    await client.write_gatt_char(
        CHAR_UUID,
        drive_cmd(speed=speed, steering=steering, lights=lights),
        response=True,
    )


# ---------------------------------------------------------------------------
# RTSP camera reader (background thread)
# ---------------------------------------------------------------------------
class CameraReader:
    def __init__(self, url: str, width: int, height: int):
        self.url = url
        self.width = width
        self.height = height
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _reader_loop(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("[CAMERA] ERROR: could not open RTSP stream")
            return

        print(f"[CAMERA] Stream opened, target {self.width}x{self.height}")
        while self._running:
            ret, frame = cap.read()
            if not ret:
                print("[CAMERA] Lost frame, reconnecting...")
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                continue

            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with self._lock:
                self._frame = frame

        cap.release()
        print("[CAMERA] Reader stopped")


# ---------------------------------------------------------------------------
# Load trained policy
# ---------------------------------------------------------------------------
def load_policy(checkpoint_path: str):
    """Load ACT policy + preprocessor/postprocessor from a checkpoint."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"Loading policy from {ckpt}...")
    policy = ACTPolicy.from_pretrained(str(ckpt))
    policy.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    policy.to(device)
    print(f"Policy loaded on {device} ({sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M params)")

    # Load preprocessor/postprocessor (handles normalization)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(ckpt),
    )

    return policy, preprocessor, postprocessor, device


# ---------------------------------------------------------------------------
# Pygame drawing
# ---------------------------------------------------------------------------
def draw(screen, font, speed, steering, mode, cam_frame, action_info=""):
    screen.fill((20, 20, 20))

    # Camera preview
    if cam_frame is not None:
        surf = pygame.surfarray.make_surface(cam_frame.swapaxes(0, 1))
        preview_w, preview_h = 320, 240
        surf = pygame.transform.scale(surf, (preview_w, preview_h))
        screen.blit(surf, (280, 10))

    # Mode indicator
    if mode == "auto":
        mode_color = (60, 255, 60)
        mode_text = "AUTONOMOUS"
    else:
        mode_color = (120, 120, 120)
        mode_text = "MANUAL"

    mode_surf = font.render(f"[{mode_text}]", True, mode_color)
    screen.blit(mode_surf, (20, 10))

    lines = [
        "",
        "Controls:",
        "  Enter       = start autonomous",
        "  Space       = emergency stop",
        "  Arrows/WASD = manual drive",
        "  Q / Esc     = quit",
        "",
        f"Speed:    {speed:6.1f}",
        f"Steering: {steering:6.1f}",
    ]
    if action_info:
        lines.append("")
        lines.append(action_info)

    y = 40
    for line in lines:
        img = font.render(line, True, (230, 230, 230))
        screen.blit(img, (20, y))
        y += 22

    pygame.display.flip()


# ---------------------------------------------------------------------------
# Main control + inference loop
# ---------------------------------------------------------------------------
async def control_loop(client, checkpoint_path):
    camera = CameraReader(RTSP_URL, FRAME_WIDTH, FRAME_HEIGHT)
    camera.start()

    # Wait for first frame
    print("Waiting for camera...")
    for _ in range(100):
        if camera.get_frame() is not None:
            break
        await asyncio.sleep(0.1)
    else:
        print("WARNING: Camera not producing frames, continuing anyway")

    policy, preprocessor, postprocessor, device = load_policy(checkpoint_path)

    pygame.init()
    screen = pygame.display.set_mode((620, 400))
    pygame.display.set_caption("LEGO 42176 - Autonomous Driver")
    font = pygame.font.SysFont("Menlo", 16)

    speed = 0.0
    steering = 0.0
    last_sent = None
    dt = 1.0 / LOOP_HZ

    mode = "manual"     # "manual" or "auto"
    action_info = ""
    last_inference_time = 0.0
    inference_interval = 1.0 / FPS

    # Action queue: the ACT policy predicts a chunk of future actions.
    # We execute them one at a time at FPS rate.
    action_queue = deque()

    await send_drive(client, 0, 0)

    running = True
    try:
        while running:
            loop_start = time.monotonic()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

                    elif event.key == pygame.K_SPACE:
                        # Emergency stop
                        speed = 0.0
                        steering = 0.0
                        mode = "manual"
                        action_queue.clear()
                        policy.reset()
                        action_info = "STOPPED"

                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        if mode == "manual":
                            mode = "auto"
                            speed = 0.0
                            steering = 0.0
                            action_queue.clear()
                            policy.reset()
                            last_inference_time = 0.0
                            action_info = "Starting autonomous..."
                            print("\n=== AUTONOMOUS MODE ===")
                        else:
                            mode = "manual"
                            action_queue.clear()
                            policy.reset()
                            speed = 0.0
                            steering = 0.0
                            action_info = ""
                            print("=== MANUAL MODE ===")

            cam_frame = camera.get_frame()

            if mode == "auto" and cam_frame is not None:
                # Run inference at FPS rate
                now = time.monotonic()
                if now - last_inference_time >= inference_interval:
                    last_inference_time = now

                    # Current state (normalized to [-1, 1] like training)
                    norm_speed = speed / 100.0
                    norm_steering = steering / 100.0

                    # Build observation batch
                    # Image: (H, W, 3) uint8 -> (1, 3, H, W) float32 [0, 1]
                    img_tensor = torch.from_numpy(cam_frame).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

                    state_tensor = torch.tensor(
                        [[norm_speed, norm_steering]], dtype=torch.float32
                    )

                    batch = {
                        "observation.image": img_tensor.to(device),
                        "observation.state": state_tensor.to(device),
                        "task": "Drive the LEGO car into the garage",
                    }

                    # Preprocess (normalization)
                    batch = preprocessor(batch)

                    # Get action from policy
                    with torch.no_grad():
                        action = policy.select_action(batch)

                    # Postprocess (denormalization)
                    action = postprocessor(action)

                    # action shape: (1, 2) -> [speed, steering] in [-1, 1]
                    act = action.squeeze(0).cpu().numpy()
                    target_speed = float(act[0]) * 100.0
                    target_steering = float(act[1]) * 100.0

                    # Clamp to safe range
                    target_speed = clamp(target_speed, -MAX_SPEED, MAX_SPEED)
                    target_steering = clamp(target_steering, -MAX_STEERING, MAX_STEERING)

                    action_info = (
                        f"Model: spd={target_speed:+.0f} str={target_steering:+.0f}"
                    )

                    # Smoothly move towards target (same ramping as manual)
                    speed = move_towards(speed, target_speed, ACCEL_PER_SEC * dt)
                    steering = move_towards(steering, target_steering, STEER_PER_SEC * dt)

            elif mode == "manual":
                # Manual keyboard control (same as main.py)
                keys = pygame.key.get_pressed()
                forward = keys[pygame.K_UP] or keys[pygame.K_w]
                reverse = keys[pygame.K_DOWN] or keys[pygame.K_s]
                left = keys[pygame.K_LEFT] or keys[pygame.K_a]
                right = keys[pygame.K_RIGHT] or keys[pygame.K_d]

                if forward and not reverse:
                    throttle_input = +1
                elif reverse and not forward:
                    throttle_input = -1
                else:
                    throttle_input = 0

                if right and not left:
                    steering_input = +1
                elif left and not right:
                    steering_input = -1
                else:
                    steering_input = 0

                target_speed = throttle_input * MAX_SPEED
                target_steering = steering_input * MAX_STEERING

                speed_rate = DECEL_PER_SEC if throttle_input == 0 else ACCEL_PER_SEC
                steering_rate = CENTER_PER_SEC if steering_input == 0 else STEER_PER_SEC

                speed = move_towards(speed, target_speed, speed_rate * dt)
                steering = move_towards(steering, target_steering, steering_rate * dt)

            # Send to car
            send_speed = int(round(speed))
            send_steering = int(round(steering))
            lights = LIGHTS_BRAKE if send_speed == 0 else LIGHTS_ON

            command = (send_speed, send_steering, lights)
            if command != last_sent:
                await send_drive(client, send_speed, send_steering, lights)
                last_sent = command

            draw(screen, font, speed, steering, mode, cam_frame, action_info)

            elapsed = time.monotonic() - loop_start
            await asyncio.sleep(max(0.0, dt - elapsed))

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        try:
            await send_drive(client, 0, 0)
        except Exception:
            pass
        camera.stop()
        pygame.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main(checkpoint_path: str):
    device = await find_hub()

    async with BleakClient(device) as client:
        print("Connected to hub")

        if platform.system() != "Darwin":
            try:
                await client.pair(protection_level=2)
            except Exception as e:
                print(f"Pairing skipped/failed: {e}")

        print("Calibrating steering...")
        await client.write_gatt_char(CHAR_UUID, STEERING_CALIBRATE_1, response=True)
        await asyncio.sleep(0.2)
        await client.write_gatt_char(CHAR_UUID, STEERING_CALIBRATE_2, response=True)
        await asyncio.sleep(0.5)

        try:
            await control_loop(client, checkpoint_path)
        finally:
            try:
                await send_drive(client, 0, 0)
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run trained ACT policy on LEGO 42176",
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to pretrained_model directory "
             "(e.g. outputs/train/checkpoints/010000/pretrained_model)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.checkpoint))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
