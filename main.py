import asyncio
import platform
import time

import pygame
from bleak import BleakScanner, BleakClient

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

LOOP_HZ = 30


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


STEERING_CALIBRATE_1 = bytes.fromhex("0d008136115100030000001000")
STEERING_CALIBRATE_2 = bytes.fromhex("0d008136115100030000000800")


async def find_hub():
    print("Searching for LEGO Technic Move Hub...")

    def match(device, adv):
        name = device.name or adv.local_name or ""
        return HUB_NAME in name

    device = await BleakScanner.find_device_by_filter(
        match,
        timeout=15.0,
        service_uuids=[SERVICE_UUID],
    )

    if device is None:
        raise RuntimeError(
            "Hub not found. Turn it on and make sure the LEGO app is closed."
        )

    print(f"Found: {device.name} / {device.address}")
    return device


async def send_drive(client, speed, steering, lights=LIGHTS_ON):
    await client.write_gatt_char(
        CHAR_UUID,
        drive_cmd(speed=speed, steering=steering, lights=lights),
        response=True,
    )


def draw(screen, font, speed, steering):
    screen.fill((20, 20, 20))

    lines = [
        "LEGO 42176 Control",
        "",
        "Hold multiple keys at once:",
        "  Up + Right   = drive forward and turn right",
        "  Up + Left    = drive forward and turn left",
        "  Down + Right = reverse and turn right",
        "  Down + Left  = reverse and turn left",
        "",
        "Controls:",
        "  Arrow keys or WASD",
        "  Space = stop + center",
        "  Q / Esc = quit",
        "",
        f"Speed:    {speed:6.1f}",
        f"Steering: {steering:6.1f}",
        "",
        "Important: keep this pygame window focused.",
    ]

    y = 20
    for line in lines:
        img = font.render(line, True, (230, 230, 230))
        screen.blit(img, (20, y))
        y += 24

    pygame.display.flip()


async def pygame_control_loop(client):
    pygame.init()
    screen = pygame.display.set_mode((620, 430))
    pygame.display.set_caption("LEGO 42176 Keyboard Control")
    font = pygame.font.SysFont("Menlo", 18)

    speed = 0.0
    steering = 0.0

    last_sent = None
    dt = 1.0 / LOOP_HZ

    await send_drive(client, 0, 0)

    running = True
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

        keys = pygame.key.get_pressed()

        forward = keys[pygame.K_UP] or keys[pygame.K_w]
        reverse = keys[pygame.K_DOWN] or keys[pygame.K_s]
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]

        # Allows Up+Right, Up+Left, Down+Right, Down+Left.
        # If opposite keys are both pressed, they cancel.
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

        if throttle_input == 0:
            speed_rate = DECEL_PER_SEC
        else:
            speed_rate = ACCEL_PER_SEC

        if steering_input == 0:
            steering_rate = CENTER_PER_SEC
        else:
            steering_rate = STEER_PER_SEC

        speed = move_towards(speed, target_speed, speed_rate * dt)
        steering = move_towards(steering, target_steering, steering_rate * dt)

        send_speed = int(round(speed))
        send_steering = int(round(steering))

        lights = LIGHTS_BRAKE if send_speed == 0 else LIGHTS_ON

        command = (send_speed, send_steering, lights)
        if command != last_sent:
            await send_drive(client, send_speed, send_steering, lights)
            last_sent = command

        draw(screen, font, speed, steering)

        elapsed = time.monotonic() - loop_start
        await asyncio.sleep(max(0.0, dt - elapsed))

    await send_drive(client, 0, 0)
    pygame.quit()


async def main():
    device = await find_hub()

    async with BleakClient(device) as client:
        print("Connected")

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
            await pygame_control_loop(client)
        finally:
            await send_drive(client, 0, 0)


if __name__ == "__main__":
    asyncio.run(main())
