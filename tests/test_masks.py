from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from seminspect.data.carinthia import decode_carinthia_mask


class CarinthiaMaskTest(unittest.TestCase):
    def test_carinthia_alpha_channel_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opaque_alpha_only.png"
            rgba = np.zeros((4, 4, 4), dtype=np.uint8)
            rgba[..., 3] = 255
            Image.fromarray(rgba, mode="RGBA").save(path)
            foreground, mode = decode_carinthia_mask(path)
            self.assertEqual(mode, "RGBA")
            self.assertFalse(foreground.any())

    def test_rgb_nonzero_becomes_foreground(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.png"
            rgb = np.zeros((4, 4, 3), dtype=np.uint8)
            rgb[1, 1] = [10, 10, 10]
            Image.fromarray(rgb, mode="RGB").save(path)
            foreground, mode = decode_carinthia_mask(path)
            self.assertEqual(mode, "RGB")
            self.assertEqual(int(foreground.sum()), 1)
