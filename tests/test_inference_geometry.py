# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from inference_geometry import Window,fixed_windows,to_image_coordinates,merge_predictions,prediction_rows


class InferenceGeometryTests(unittest.TestCase):
    def test_tiles_cover_edges_without_duplicates(self):
        for width,height in [(1,1),(639,640),(640,640),(641,641),(1360,765),(1920,1080)]:
            windows=fixed_windows(width,height)
            self.assertEqual(len(windows),len(set(windows)))
            covered=np.zeros((height,width),dtype=bool)
            for win in windows:
                self.assertTrue(0<=win.x1<win.x2<=width and 0<=win.y1<win.y2<=height)
                covered[win.y1:win.y2,win.x1:win.x2]=True
            self.assertTrue(covered.all())
        self.assertEqual(len(fixed_windows(1360,765)),6)

    def test_invalid_overlap_rejected(self):
        for value in [-.1,1,float('nan')]:
            with self.assertRaises(ValueError):
                fixed_windows(100,100,overlap=value)

    def test_reprojection_clips_to_crop_and_logs_zero_area(self):
        boxes=[[-2,1,15,15,.9,0],[20,20,30,30,.8,1]]
        rows,stats=to_image_coordinates(boxes,Window(100,50,120,70))
        np.testing.assert_array_equal(rows[:,:4],[[100,51,115,65]])
        self.assertEqual(stats,{'clipped':2,'zero_area_removed':1})

    def test_nms_preserves_different_classes_and_global_cap(self):
        rows,counts=merge_predictions([[[0,0,10,10,.9,0],[0,0,10,10,.8,0]],[[0,0,10,10,.85,1],[30,30,40,40,.7,2]]],max_det=2,return_counts=True)
        self.assertEqual(rows[:,5].tolist(),[0,1])
        self.assertEqual(counts,{'before_merge':4,'after_nms_before_cap':3,'after_cap':2,'removed_nms':1,'removed_cap':1})

    def test_invalid_prediction_values_rejected(self):
        for row in [[0,0,1,1,float('nan'),0],[0,0,1,1,.5,10],[0,0,1,1,.5,.5],[1,1,0,0,.5,0]]:
            with self.assertRaises(ValueError):
                prediction_rows([row])


if __name__=='__main__':
    unittest.main()
