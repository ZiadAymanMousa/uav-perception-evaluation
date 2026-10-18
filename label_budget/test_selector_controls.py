# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Meaningful decision/input-boundary and direct-scoring controls for the pilot."""
import os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
(HERE/'runtime/ultralytics/Ultralytics').mkdir(parents=True,exist_ok=True)
(HERE/'runtime/matplotlib').mkdir(parents=True,exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['YOLO_CONFIG_DIR']=str(HERE/'runtime/ultralytics')
os.environ['YOLO_AUTOINSTALL']='false'
os.environ['YOLO_OFFLINE']='true'
os.environ['MPLCONFIGDIR']=str(HERE/'runtime/matplotlib')
os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,str(HERE/'source_snapshot'))
from dataclasses import replace
import json,time,unittest
import numpy as np
from selector_core import METHODS,NAMES,Stats,Revealed,CacheOracle,score_revealed,select_scores


def simple_stats(classes=tuple(range(10)),perfect=True):
    count=len(classes)
    return Stats(np.full((count,10),perfect,dtype=bool),np.linspace(.99,.5,count,dtype=np.float32),
        np.asarray(classes,dtype=np.float32),np.asarray(classes,dtype=np.float32))

def pool(first=None,hidden=None):
    first=simple_stats() if first is None else first
    hidden=simple_stats() if hidden is None else hidden
    return CacheOracle(('queried','hidden'),{m:[first,hidden] for m in METHODS})

class BoundaryTests(unittest.TestCase):
    def test_exact_tie_uses_declared_order(self):
        scores={m:3. for m in METHODS}
        self.assertEqual(select_scores(scores),METHODS[0])
        scores[METHODS[2]]=np.nextafter(3.,4.)
        self.assertEqual(select_scores(scores),METHODS[2])
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):select_scores({m:float('nan') for m in METHODS})
    def test_missing_class_abstains(self):
        result=score_revealed(pool(simple_stats(tuple(range(9)))).reveal([0]))
        self.assertEqual(result['status'],'abstained_missing_class')
        self.assertEqual(result['missing_classes'],[9]);self.assertIsNone(result['selected'])
    def test_empty_targets_abstain(self):
        result=score_revealed(pool(simple_stats(())).reveal([0]))
        self.assertEqual(result['missing_classes'],list(range(10)))
    def test_query_duplicates_and_bounds_rejected(self):
        oracle=pool()
        for indices in ([],[0,0],[-1],[2],[True]):
            with self.subTest(indices=indices),self.assertRaises(ValueError):oracle.reveal(indices)
    def test_query_order_canonical(self):
        oracle=pool(hidden=simple_stats(perfect=False))
        a,b=oracle.reveal([1,0]),oracle.reveal([0,1])
        self.assertEqual(a.image_ids,b.image_ids)
        self.assertEqual(score_revealed(a),score_revealed(b))
    def test_false_class_receipt_rejected(self):
        reveal=pool().reveal([0])
        with self.assertRaises(ValueError):score_revealed(replace(reveal,target_classes=(0,)))
    def test_false_count_receipt_rejected(self):
        reveal=pool().reveal([0])
        with self.assertRaises(ValueError):score_revealed(replace(reveal,target_count=999))
    def test_candidate_target_disagreement_rejected(self):
        reveal=pool().reveal([0]);changed=dict(reveal.stats)
        changed[METHODS[1]]=(simple_stats(tuple(range(9))),)
        with self.assertRaises(ValueError):score_revealed(replace(reveal,stats=changed))
    def test_candidate_identity_and_missing_image_rejected(self):
        reveal=pool().reveal([0]);changed=dict(reveal.stats);changed.pop(METHODS[-1])
        with self.assertRaises(ValueError):score_revealed(replace(reveal,stats=changed))
        changed=dict(reveal.stats);changed[METHODS[0]]=()
        with self.assertRaises(ValueError):score_revealed(replace(reveal,stats=changed))
    def test_unrevealed_labels_cannot_change_selection(self):
        # The hidden image's target labels and correctness flags both change.
        a=pool(hidden=simple_stats()).reveal([0])
        b=pool(hidden=simple_stats((9,),perfect=False)).reveal([0])
        self.assertEqual(score_revealed(a),score_revealed(b))
    def test_unrevealed_storage_is_never_touched(self):
        good=simple_stats()
        class PoisonRows:
            def __len__(self):return 2
            def __getitem__(self,index):
                if index!=0:raise AssertionError('Unrevealed labels accessed')
                return good
        oracle=CacheOracle(('queried','hidden'),{m:PoisonRows() for m in METHODS})
        self.assertEqual(score_revealed(oracle.reveal([0]))['status'],'selected')
        with self.assertRaises(AssertionError):oracle.reveal([1])
    def test_pooled_ap_is_not_mean_image_ap(self):
        first=simple_stats();second=Stats(np.zeros((0,10),bool),np.zeros(0,np.float32),
            np.zeros(0,np.float32),np.arange(10,dtype=np.float32))
        oracle=pool(first,second)
        full=score_revealed(oracle.reveal([0,1]))['scores_ap50_95_percent'][METHODS[0]]
        mean=np.mean([score_revealed(oracle.reveal([i]))['scores_ap50_95_percent'][METHODS[0]] for i in (0,1)])
        self.assertGreater(abs(full-mean),1.)
    def test_constructed_cached_vs_direct_ap(self):
        import torch
        from types import SimpleNamespace
        from ultralytics.engine.validator import BaseValidator
        from ultralytics.utils.metrics import box_iou
        from score_converted_predictions import score_records
        cases=[];stats=[]
        for index in range(2):
            gt=np.asarray([[c,c*20,0,c*20+10,10] for c in range(10)],dtype=np.float32)
            pred=np.asarray([[c*20,0,c*20+10,10,.9-.2*index,c] for c in range(10)],dtype=np.float32)
            # One high-confidence duplicate false positive precedes valid matches.
            if index==1:
                pred=np.concatenate([np.asarray([[300,300,310,310,.95,0]],np.float32),pred])
            tp=BaseValidator.match_predictions(SimpleNamespace(iouv=torch.linspace(.5,.95,10)),
                torch.from_numpy(pred[:,5]),torch.from_numpy(gt[:,0]),
                box_iou(torch.from_numpy(gt[:,1:]),torch.from_numpy(pred[:,:4]))).numpy()
            stats.append(Stats(tp,pred[:,4],pred[:,5],gt[:,0]));cases.append({'gt':gt,'predictions':pred})
        oracle=CacheOracle(('a','b'),{m:stats for m in METHODS})
        result=score_revealed(oracle.reveal([1,0]))
        direct=score_records(cases,dict(enumerate(NAMES)))
        self.assertAlmostEqual(result['scores_ap50_95_percent'][METHODS[0]],
            100*direct['metrics']['metrics/mAP50-95(B)'],places=10)

if __name__=='__main__':
    started=time.perf_counter()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(BoundaryTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    import torch
    output={'status':'pass' if result.wasSuccessful() else 'failed','tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'seconds':time.perf_counter()-started,
        'cuda_initialised':torch.cuda.is_initialized(),
        'limit':'Architectural/invariance tests, not a malicious-code security sandbox or statistical guarantee'}
    (HERE/'unit_controls.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(output))
    raise SystemExit(0 if result.wasSuccessful() else 1)
