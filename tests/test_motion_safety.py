"""Synthetic-only regressions for background-locked v0.9.0 motion."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from puppet_motion import HeadRig
from body_motion import TorsoMotion


def synthetic_head():
    h, w = 240, 320
    yy, xx = np.mgrid[:h, :w]
    image = np.zeros((h, w, 3), np.uint8)
    image[:, :, 0] = ((xx * 3 + yy) % 100 + 70).astype(np.uint8)
    image[:, :, 1] = ((xx + yy * 2) % 90 + 80).astype(np.uint8)
    image[:, :, 2] = ((xx * 2 + yy * 3) % 80 + 90).astype(np.uint8)
    cv2.ellipse(image, (160, 108), (48, 66), 0, 0, 360, (40, 70, 180), -1)
    cv2.ellipse(image, (160, 108), (40, 58), 0, 0, 360, (70, 150, 220), -1)
    cv2.circle(image, (145, 100), 4, (10, 10, 10), -1)
    cv2.circle(image, (175, 100), 4, (10, 10, 10), -1)
    cv2.ellipse(image, (160, 132), (14, 5), 0, 0, 180, (20, 20, 100), 2)
    return image


class MotionSafetyTests(unittest.TestCase):
    def test_head_motion_keeps_pixels_outside_head_union_unchanged(self):
        image = synthetic_head()
        rig = HeadRig(image.shape, [0.35, 0.28, 0.30, 0.42], image=image, style="cartoon")
        self.assertTrue(rig.ready, "synthetic cartoon head should yield a reliable mask")
        moved = rig.apply(image, 1.5, 0.004, 0.004)
        x1, y1, x2, y2 = rig.roi
        self.assertTrue(np.array_equal(moved[:y1], image[:y1]))
        self.assertTrue(np.array_equal(moved[y2:], image[y2:]))
        self.assertTrue(np.array_equal(moved[:, :x1], image[:, :x1]))
        self.assertTrue(np.array_equal(moved[:, x2:], image[:, x2:]))
        changed = np.any(moved[y1:y2, x1:x2] != image[y1:y2, x1:x2], axis=2)
        self.assertFalse(np.any(changed & ~rig.last_union),
                         "pixels outside the old/moved head union must remain identical")
        self.assertGreater(np.count_nonzero(changed), 0)

    def test_uniform_image_disables_head_motion(self):
        image = np.full((200, 240, 3), 128, np.uint8)
        rig = HeadRig(image.shape, [0.3, 0.2, 0.4, 0.5], image=image)
        self.assertFalse(rig.ready)
        self.assertTrue(np.array_equal(rig.apply(image, 2.0, 0.02, 0.02), image))

    def test_torso_motion_leaves_far_background_unchanged(self):
        image = synthetic_head()
        alpha = np.zeros(image.shape[:2], np.float32)
        alpha[90:220, 120:205] = 1.0
        drive = np.ones(100, np.float32)
        rig = TorsoMotion(image, [0.35, 0.28, 0.30, 0.42], drive, alpha=alpha, scale=1.0)
        self.assertTrue(rig.ready)
        moved = rig.apply(0.4)
        self.assertTrue(np.array_equal(moved[:50], image[:50]))
        self.assertTrue(np.array_equal(moved[:, :50], image[:, :50]))
        self.assertTrue(np.array_equal(moved[:, 270:], image[:, 270:]))


if __name__ == "__main__":
    unittest.main()
