# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
import os
from pathlib import Path
import sys
import unittest
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(PROJECT/'artifacts/ultralytics_config')
sys.path.insert(0, str(PROJECT/'src'))
from analyse_cap_matching import constructed_controls, match_pair


class CapMatchingTests(unittest.TestCase):
    def test_opposed_iou_control_does_not_reassign_retained_detection(self):
        result = constructed_controls()['opposed_confidence_iou']
        self.assertLess(result['prediction_pair_iou'], .7)
        scores = result['statistics']
        self.assertAlmostEqual(scores['all2']['ap50_percent'], 99.5)
        self.assertAlmostEqual(scores['top1_fixed_matches']['ap50_percent'], 99.5)
        self.assertAlmostEqual(scores['top1_rematched']['ap50_percent'], 99.5)

    def test_false_positive_tail_changes_pinned_ap_without_higher_recall(self):
        scores = constructed_controls()['incomplete_recall_fp_tail']['statistics']
        self.assertAlmostEqual(scores['all2']['ap50_percent'], 62.25)
        self.assertAlmostEqual(scores['top1_rematched']['ap50_percent'], 75)
        self.assertEqual(scores['all2']['matched_tp_by_iou'], scores['top1_rematched']['matched_tp_by_iou'])
        self.assertTrue(all(x == 50 for x in scores['all2']['class_macro_attained_recall_by_iou_percent']))
        self.assertAlmostEqual(scores['top1_rematched']['ap_grid_area_beyond_attained_recall_percent'],25)

    def test_no_truncation_leaves_assignments_identical(self):
        full, retained = match_pair([[0, 10, 10, 20, 20]], [[10, 10, 20, 20, .8, 0]], 500)
        np.testing.assert_array_equal(full, retained)

    def test_empty_predictions_or_ground_truth_have_no_matches(self):
        for gt, pred in [([], [[10, 10, 20, 20, .8, 0]]), ([[0, 10, 10, 20, 20]], [])]:
            full, retained = match_pair(gt, pred, 500)
            self.assertFalse(full.any())
            self.assertFalse(retained.any())


if __name__ == '__main__':
    unittest.main()
