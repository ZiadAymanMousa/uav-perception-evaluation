# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from study_inputs import native_detections


class NativePredictionBridgeTests(unittest.TestCase):
    def test_fractional_coordinates_and_last_class_do_not_shift_pixels(self):
        output=native_detections([[3.25,7.5,10.75,17.5,.9,9]])
        np.testing.assert_allclose(output,[[3.25,7.5,7.5,10,.9,10,-1,-1]],rtol=0,atol=1e-7)

    def test_empty_predictions_remain_an_eight_column_native_array(self):
        self.assertEqual(native_detections([]).shape,(0,8))

    def test_incompatible_labels_and_degenerate_boxes_fail(self):
        for rows in ([[0,0,1,1,.5,28]],[[0,0,0,1,.5,0]]):
            with self.assertRaises(ValueError):
                native_detections(rows)

    def test_global_score_order_is_required_before_native_cap(self):
        with self.assertRaises(ValueError):
            native_detections([[0,0,1,1,.1,0],[0,0,1,1,.9,1]])


if __name__=='__main__':
    unittest.main()
