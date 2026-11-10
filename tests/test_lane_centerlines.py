import unittest
from itertools import product

import numpy as np

from src.main import get_lane_centerlines, get_road_boundaries, isolate_lane_markings


class LaneCenterlinesTests(unittest.TestCase):
    def test_each_combination_of_missing_markings(self):
        for present in product((False, True), repeat=3):
            with self.subTest(present=present):
                edges = np.zeros((240, 400), dtype=np.uint8)
                gray = np.zeros_like(edges)
                for x, exists in zip((60, 200, 340), present):
                    if exists:
                        edges[145:, x - 3] = 255
                        edges[145:, x + 3] = 255
                        gray[145:, x - 3:x + 4] = 255

                lines = get_lane_centerlines(edges, gray)

                self.assertIsInstance(lines, tuple)
                self.assertEqual(len(lines), 3)
                for x, exists, line in zip((60, 200, 340), present, lines):
                    if exists:
                        self.assertIsNotNone(line)
                        self.assertEqual(line.dtype, np.int32)
                        np.testing.assert_array_equal(line[:, 0, 0], x)
                    else:
                        self.assertIsNone(line)

    def test_drawing_with_no_markings(self):
        image = np.zeros((240, 400, 3), dtype=np.uint8)
        np.testing.assert_array_equal(get_road_boundaries(image), image)

    def test_middle_marking_on_a_curve_can_be_right_of_image_center(self):
        left = np.array([[[80, 360]], [[20, 240]]], dtype=np.int32)
        center = np.array([[[430, 350]], [[200, 230]]], dtype=np.int32)
        right = np.array([[[525, 305]], [[320, 220]]], dtype=np.int32)
        result = isolate_lane_markings([right, left, center], 547)
        for actual, expected in zip(result, (left, center, right)):
            self.assertIs(actual, expected)

    def test_extra_interior_line_keeps_longest_center_marking(self):
        left = np.array([[[10, 380]], [[285, 240]]], dtype=np.int32)
        center = np.array([[[360, 400]], [[365, 285]]], dtype=np.int32)
        fragment = np.array([[[385, 370]], [[375, 285]]], dtype=np.int32)
        right = np.array([[[750, 370]], [[470, 240]]], dtype=np.int32)
        result = isolate_lane_markings([fragment, right, center, left], 764)
        for actual, expected in zip(result, (left, center, right)):
            self.assertIs(actual, expected)


if __name__ == '__main__':
    unittest.main()
