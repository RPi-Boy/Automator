import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "arm_perception"))

from arm_perception.overlay import compose_view, decode_esp32_status, draw_detections  # noqa: E402


class TestOverlay(unittest.TestCase):
    def test_esp32_status_bits(self):
        s = decode_esp32_status(1 | 4 | (300 << 8))
        self.assertTrue(s["estop_latched"] and s["cmd_fresh"])
        self.assertFalse(s["motors_enabled"] or s["hw_estop_line"])
        self.assertEqual(s["cmd_count"], 300)
        self.assertTrue(decode_esp32_status(2 | 8)["hw_estop_line"])

    def test_detections_drawn_and_input_untouched(self):
        img = np.zeros((480, 640, 3), np.uint8)
        out = draw_detections(img, [{"bbox": (100, 100, 50, 50), "cls": 1, "unc": 0.2},
                                    {"bbox": (300, 200, 60, 60), "cls": 0, "unc": 0.9}], timing_ms=7.5)
        self.assertEqual(int(img.sum()), 0)
        self.assertGreater(int(out.sum()), 0)
        self.assertGreater(out[100, 100:150, 1].max(), out[100, 100:150, 0].max())     # known box is green
        self.assertGreater(out[200, 300:360, 0].max(), out[200, 300:360, 1].max())     # flagged box is red

    def test_compose_handles_missing_images(self):
        view = compose_view(None, np.zeros((480, 640, 3), np.uint8), [("hello", (255, 255, 255))])
        self.assertEqual(view.shape, (480 + 96, 1280, 3))


if __name__ == "__main__":
    unittest.main()
