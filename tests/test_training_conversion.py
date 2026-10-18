# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prepare_training_pilot import convert_rows


class ConversionTests(unittest.TestCase):
    def test_geometry_and_class_mapping(self):
        labels, excluded, clipped = convert_rows([['10','20','20','40','1','10','0','0']], 100, 200)
        self.assertEqual(labels, [(9, .2, .2, .2, .2)])
        self.assertEqual((excluded, clipped), ([], 0))

    def test_ignored_others_and_unscored_preserved(self):
        rows = [[0,0,10,10,score,category,0,0] for score, category in [(0,0),(1,11),(0,4)]]
        labels, excluded, _ = convert_rows([[str(x) for x in row] for row in rows], 100, 100)
        self.assertEqual(labels, [])
        self.assertEqual(len(excluded), 3)
        self.assertEqual(excluded[2]['annotation']['category'], 4)

    def test_boundary_clip_is_explicit(self):
        labels, _, clipped = convert_rows([['90','90','20','20','1','1','0','0']], 100, 100)
        self.assertEqual(labels, [(0, .95, .95, .1, .1)])
        self.assertEqual(clipped, 1)

    def test_outside_target_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no image intersection'):
            convert_rows([['110','0','5','5','1','1','0','0']], 100, 100)

    def test_invalid_class_rejected(self):
        with self.assertRaises(ValueError):
            convert_rows([['0','0','10','10','1','12','0','0']], 100, 100)


if __name__ == '__main__':
    unittest.main()
