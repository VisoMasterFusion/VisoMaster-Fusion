"""
Generate synthetic test images that do not require real model weights.
Run once from the project root:
    python tests/fixtures/generate_fixtures.py
Outputs are committed to the repo so CI never re-generates them.
"""

import os
import cv2
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_face_512() -> None:
    """512×512 synthetic face: flesh-tone background, dark oval head, white eye circles."""
    img = np.full((512, 512, 3), (180, 140, 110), dtype=np.uint8)
    # Head oval
    cv2.ellipse(img, (256, 256), (160, 200), 0, 0, 360, (220, 180, 150), -1)
    # Eyes
    cv2.circle(img, (190, 210), 28, (255, 255, 255), -1)
    cv2.circle(img, (322, 210), 28, (255, 255, 255), -1)
    cv2.circle(img, (190, 210), 14, (50, 40, 30), -1)
    cv2.circle(img, (322, 210), 14, (50, 40, 30), -1)
    # Mouth
    cv2.ellipse(img, (256, 330), (60, 25), 0, 0, 180, (160, 80, 80), 3)
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "face_512.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    )
    print("Written: face_512.png")


def make_equirect_360_720() -> None:
    """360×720 equirectangular gradient image (H=360, W=720)."""
    img = np.zeros((360, 720, 3), dtype=np.uint8)
    # R channel: horizontal gradient (theta proxy)
    img[:, :, 0] = np.tile(np.linspace(0, 255, 720, dtype=np.uint8), (360, 1))
    # B channel: vertical gradient (phi proxy)
    img[:, :, 2] = np.tile(np.linspace(0, 255, 360, dtype=np.uint8)[:, None], (1, 720))
    # Add a bright blob at center-left (simulates left-eye face bbox)
    cv2.circle(img, (180, 180), 30, (200, 200, 50), -1)
    # Add a bright blob at center-right (simulates right-eye face bbox)
    cv2.circle(img, (540, 180), 30, (50, 200, 200), -1)
    path = os.path.join(OUTPUT_DIR, "equirect_360_720.png")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("Written: equirect_360_720.png")


def make_equirect_half_180_360() -> None:
    """180×360 single-eye equirectangular (full frame = one hemisphere)."""
    img = np.zeros((180, 360, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.linspace(0, 255, 360, dtype=np.uint8), (180, 1))
    img[:, :, 2] = np.tile(np.linspace(0, 255, 180, dtype=np.uint8)[:, None], (1, 360))
    cv2.circle(img, (180, 90), 20, (200, 200, 50), -1)
    path = os.path.join(OUTPUT_DIR, "equirect_180_360.png")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("Written: equirect_180_360.png")


if __name__ == "__main__":
    make_face_512()
    make_equirect_360_720()
    make_equirect_half_180_360()
    print("All fixtures generated.")
