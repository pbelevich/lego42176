#!/usr/bin/env python3
"""
Record a LeRobot-compatible dataset while driving LEGO 42176 via keyboard.

Usage:
    source venv_record/bin/activate
    python record_dataset.py

Controls (same as main.py):
    Arrow keys / WASD  - drive
    Space              - emergency stop
    R                  - start / stop recording an episode
    N                  - discard current episode
    Q / Esc            - quit and finalize dataset

The script connects to the LEGO hub via BLE, opens the RTSP camera stream,
and shows a pygame window. Press R to start recording an episode, drive the
car into the garage, then press R again to save the episode. Repeat for as
many episodes as needed, then press Q to quit and finalize the dataset.

Dataset is saved to ./lego_garage_dataset/ in LeRobot v3.0 format.
"""

import asyncio
import os
import platform
import threading
import time

import cv2
import numpy as np
import pygame
from bleak import BleakScanner, BleakClient
from lerobot.datasets.lerobot_dataset import LeRobotDataset

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
# Recording configuration
# ---------------------------------------------------------------------------
RTSP_URL = os.environ.get(
    "RTSP_URL",
    "rtsp://user:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=1",
)
DATASET_DIR = "./lego_garage_dataset"
DATASET_REPO_ID = "lego42176/garage_parking"
TASK_DESCRIPTION = "Drive the LEGO car into the garage"

FPS = 15  # recording fps (lower than 30 to keep dataset size sane)
FRAME_WIDTH = 640   # native resolution of subtype=1 stream
FRAME_HEIGHT = 480

LOOP_HZ = 30  # control loop rate (same as main.py)


# ---------------------------------------------------------------------------
# Helpers (from main.py, unchanged)
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
# RTSP camera reader (runs in a background thread)
# ---------------------------------------------------------------------------
class CameraReader:
    """Continuously grabs frames from an RTSP stream in a background thread."""

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

            # Resize only if the stream doesn't match the target
            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            # Convert BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with self._lock:
                self._frame = frame

        cap.release()
        print("[CAMERA] Reader stopped")


# ---------------------------------------------------------------------------
# Create or resume the LeRobot dataset
# ---------------------------------------------------------------------------

DATASET_FEATURES = {
    "observation.image": {
        "dtype": "video",
        "shape": (FRAME_HEIGHT, FRAME_WIDTH, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (2,),
        "names": ["speed", "steering"],
    },
    "action": {
        "dtype": "float32",
        "shape": (2,),
        "names": ["speed", "steering"],
    },
}

# Shared codec/writer settings for create and resume
_WRITER_KWARGS = dict(
    vcodec="h264_videotoolbox",
    image_writer_processes=0,
    image_writer_threads=0,
)

# Module-level reference so atexit can always finalize it
_active_dataset: LeRobotDataset | None = None


def _atexit_finalize():
    """Safety net: finalize the dataset if it wasn't finalized during normal shutdown."""
    global _active_dataset
    if _active_dataset is not None and not _active_dataset._is_finalized:
        print("[atexit] Finalizing dataset (safety net)...")
        try:
            _active_dataset.finalize()
            print(f"[atexit] Dataset saved to {os.path.abspath(DATASET_DIR)}")
        except Exception as e:
            print(f"[atexit] ERROR during finalize: {e}")
        _active_dataset = None


import atexit
atexit.register(_atexit_finalize)


def open_dataset() -> LeRobotDataset:
    """Create a new dataset or resume an existing one in write mode."""
    global _active_dataset

    root = os.path.abspath(DATASET_DIR)
    if os.path.exists(os.path.join(root, "meta", "info.json")):
        print(f"Resuming existing dataset at {root}")
        ds = LeRobotDataset.resume(
            DATASET_REPO_ID, root=root, **_WRITER_KWARGS,
        )
    else:
        print(f"Creating new dataset at {root}")
        ds = LeRobotDataset.create(
            repo_id=DATASET_REPO_ID,
            fps=FPS,
            features=DATASET_FEATURES,
            root=root,
            robot_type="lego42176",
            use_videos=True,
            **_WRITER_KWARGS,
        )

    _active_dataset = ds
    return ds


# ---------------------------------------------------------------------------
# Pygame drawing
# ---------------------------------------------------------------------------
def draw(screen, font, speed, steering, recording, ep_idx, frame_count, cam_frame):
    screen.fill((20, 20, 20))

    # Draw camera preview (right side, scaled to 320x240 for the UI)
    if cam_frame is not None:
        # cam_frame is RGB (H, W, 3), pygame expects (W, H) surface
        surf = pygame.surfarray.make_surface(cam_frame.swapaxes(0, 1))
        preview_w, preview_h = 320, 240
        surf = pygame.transform.scale(surf, (preview_w, preview_h))
        screen.blit(surf, (280, 10))

    rec_color = (255, 60, 60) if recording else (120, 120, 120)
    rec_text = "RECORDING" if recording else "IDLE"
    rec_surf = font.render(f"[{rec_text}]", True, rec_color)
    screen.blit(rec_surf, (20, 10))

    lines = [
        "",
        f"Episode:  {ep_idx}",
        f"Frames:   {frame_count}",
        "",
        "Controls:",
        "  Arrows/WASD = drive",
        "  Space       = stop",
        "  R           = start/stop episode",
        "  N           = discard episode",
        "  Q / Esc     = quit",
        "",
        f"Speed:    {speed:6.1f}",
        f"Steering: {steering:6.1f}",
    ]

    y = 40
    for line in lines:
        img = font.render(line, True, (230, 230, 230))
        screen.blit(img, (20, y))
        y += 22

    pygame.display.flip()


# ---------------------------------------------------------------------------
# Main control + recording loop
# ---------------------------------------------------------------------------
async def control_loop(client):
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

    dataset = open_dataset()
    ep_idx = dataset.meta.total_episodes

    pygame.init()
    screen = pygame.display.set_mode((620, 500))
    pygame.display.set_caption("LEGO 42176 - Dataset Recorder")
    font = pygame.font.SysFont("Menlo", 16)

    speed = 0.0
    steering = 0.0
    last_sent = None
    dt = 1.0 / LOOP_HZ

    recording = False
    frame_count = 0
    last_record_time = 0.0
    record_interval = 1.0 / FPS

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
                        speed = 0.0
                        steering = 0.0

                    elif event.key == pygame.K_r:
                        if not recording:
                            # Start a new episode
                            recording = True
                            frame_count = 0
                            last_record_time = 0.0
                            print(f"\n=== Episode {ep_idx}: RECORDING STARTED ===")
                        else:
                            # Save the episode, finalize, and reopen for next
                            recording = False
                            if frame_count > 0:
                                print(f"=== Episode {ep_idx}: SAVING ({frame_count} frames) ===")
                                dataset.save_episode()
                                dataset.finalize()
                                ep_idx += 1
                                print(f"=== Episode saved & flushed. Total: {ep_idx} ===")
                                # Reopen in write mode for the next episode
                                dataset = open_dataset()
                                print(f"=== Ready for next episode ===\n")
                            else:
                                print("=== Episode discarded (0 frames) ===\n")

                    elif event.key == pygame.K_n:
                        if recording:
                            recording = False
                            # Discard: finalize current (no episode saved) and reopen
                            dataset.finalize()
                            dataset = open_dataset()
                            frame_count = 0
                            print("=== Episode DISCARDED ===\n")

            # ---- Keyboard input (same logic as main.py) ----
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

            send_speed = int(round(speed))
            send_steering = int(round(steering))
            lights = LIGHTS_BRAKE if send_speed == 0 else LIGHTS_ON

            command = (send_speed, send_steering, lights)
            if command != last_sent:
                await send_drive(client, send_speed, send_steering, lights)
                last_sent = command

            # ---- Record a frame at FPS rate ----
            cam_frame = camera.get_frame()

            if recording and cam_frame is not None:
                now = time.monotonic()
                if now - last_record_time >= record_interval:
                    last_record_time = now

                    # Normalize speed/steering to [-1, 1] for the model
                    norm_speed = speed / 100.0
                    norm_steering = steering / 100.0

                    frame_data = {
                        "observation.image": cam_frame,  # (H, W, 3) uint8 RGB
                        "observation.state": np.array(
                            [norm_speed, norm_steering], dtype=np.float32
                        ),
                        "action": np.array(
                            [norm_speed, norm_steering], dtype=np.float32
                        ),
                        "task": TASK_DESCRIPTION,
                    }
                    dataset.add_frame(frame_data)
                    frame_count += 1

            # ---- Draw ----
            draw(screen, font, speed, steering, recording, ep_idx, frame_count, cam_frame)

            elapsed = time.monotonic() - loop_start
            await asyncio.sleep(max(0.0, dt - elapsed))

    except Exception as e:
        print(f"\nERROR in control loop: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Stop car
        try:
            await send_drive(client, 0, 0)
        except Exception:
            pass

        # Save any in-progress recording
        try:
            if recording and frame_count > 0:
                print(f"Saving in-progress episode ({frame_count} frames)...")
                dataset.save_episode()
                ep_idx += 1
        except Exception as e:
            print(f"WARNING: Failed to save in-progress episode: {e}")

        # Finalize (writes parquet footer)
        try:
            print(f"Finalizing dataset ({ep_idx} total episodes)...")
            dataset.finalize()
            global _active_dataset
            _active_dataset = None
            print(f"Dataset saved to {os.path.abspath(DATASET_DIR)}")
        except Exception as e:
            print(f"ERROR during finalize: {e}")
            import traceback
            traceback.print_exc()

        camera.stop()
        pygame.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main():
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
            await control_loop(client)
        finally:
            try:
                await send_drive(client, 0, 0)
            except Exception:
                pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
