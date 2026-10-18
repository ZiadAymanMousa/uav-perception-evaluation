# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
(PROJECT/'artifacts/ultralytics_config/Ultralytics').mkdir(parents=True,exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(PROJECT/'artifacts/ultralytics_config')
os.environ['YOLO_OFFLINE']='true'
os.environ['YOLO_AUTOINSTALL']='false'
sys.path.insert(0,str(PROJECT/'src'))
from score_converted_predictions import label_rows_as_xyxy,score_records


class ConvertedScoringTests(unittest.TestCase):
    names={i:str(i) for i in range(10)}

    def test_perfect_control_follows_pinned_interpolation(self):
        result=score_records([{'gt':[[0,0,0,10,10]],'predictions':[[0,0,10,10,.8,0]]}],self.names)
        self.assertAlmostEqual(result['metrics']['metrics/mAP50-95(B)'],.995)

    def test_high_score_false_positive_halves_precision_envelope(self):
        result=score_records([{'gt':[[0,0,0,10,10]],'predictions':[[30,30,40,40,.9,0],[0,0,10,10,.8,0]]}],self.names)
        self.assertAlmostEqual(result['metrics']['metrics/mAP50(B)'],.4975)

    def test_wrong_class_is_not_a_true_positive(self):
        result=score_records([{'gt':[[0,0,0,10,10]],'predictions':[[0,0,10,10,.8,1]]}],self.names)
        self.assertEqual(result['metrics']['metrics/mAP50(B)'],0)

    def test_no_targets_does_not_invent_accuracy(self):
        result=score_records([{'gt':[],'predictions':[]}],self.names)
        self.assertIsNone(result['metrics'])

    def test_single_and_multi_label_candidates_are_different_protocols(self):
        import torch
        from ultralytics.utils.nms import non_max_suppression
        raw=torch.zeros((1,14,7))
        raw[0,:4,0]=torch.tensor([5.,5.,10.,10.])
        raw[0,4:6,0]=torch.tensor([.9,.8])
        scores=[]
        for multi_label in (False,True):
            predictions=non_max_suppression(raw.clone(),conf_thres=.001,iou_thres=.7,max_det=1000,multi_label=multi_label)[0].numpy()
            result=score_records([{'gt':[[1,0,0,10,10]],'predictions':predictions}],self.names)
            scores.append(result['metrics']['metrics/mAP50(B)'])
        self.assertEqual(scores[0],0)
        self.assertAlmostEqual(scores[1],.995)

    def test_normalised_labels_map_to_original_image(self):
        with tempfile.TemporaryDirectory(dir=PROJECT/'artifacts') as folder:
            path=Path(folder)/'labels.txt'
            path.write_text('3 0.5 0.5 0.2 0.4\n')
            row=label_rows_as_xyxy(path,100,200)[0]
            np.testing.assert_allclose(row,[3,40,60,60,140],rtol=0,atol=1e-5)


if __name__=='__main__':
    unittest.main()
