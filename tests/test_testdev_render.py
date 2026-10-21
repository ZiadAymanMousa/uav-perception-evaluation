# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Controls against silently publishing incomplete or contradictory aggregates."""
import copy
import hashlib
import json
import sys
import tempfile
from unittest import mock
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('testdev_renderer',ROOT/'src/render_testdev_results.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class AggregateControls(unittest.TestCase):
    def setUp(self):
        d=ROOT/'results/testdev'
        self.ap=m.read_csv(d/'ap_comparison.csv',m.AP_FIELDS)
        self.contrasts=m.read_csv(d/'registered_contrasts.csv',m.CONTRAST_FIELDS)
        self.costs=m.read_csv(d/'diagnostic_costs.csv',m.COST_FIELDS)
    def check(self):
        return m.validate(self.ap,self.contrasts,self.costs)
    def test_complete_published_table(self):
        self.assertEqual(len(self.check()),12)
    def test_missing_aggregate_setting_cell_rejected(self):
        self.ap.pop()
        with self.assertRaisesRegex(ValueError,'Missing AP'): self.check()
    def test_duplicate_ap_cell_rejected(self):
        self.ap[-1]=copy.deepcopy(self.ap[0])
        with self.assertRaisesRegex(ValueError,'duplicate AP'): self.check()
    def test_nan_rejected(self):
        self.ap[0]['testdev_ap50_95_percent']='NaN'
        with self.assertRaisesRegex(ValueError,'Nonfinite'): self.check()
    def test_contradictory_difference_rejected(self):
        self.ap[0]['testdev_minus_development_APpp']='0'
        with self.assertRaisesRegex(ValueError,'difference disagrees'): self.check()
    def test_sign_reversed_contrast_rejected(self):
        self.contrasts[0]['APpp']=-float(self.contrasts[0]['APpp'])
        with self.assertRaisesRegex(ValueError,'Contrast disagrees'): self.check()
    def test_missing_or_duplicate_contrast_rejected(self):
        self.contrasts.pop()
        with self.assertRaisesRegex(ValueError,'Missing registered'): self.check()
        self.setUp(); self.contrasts[-1]=copy.deepcopy(self.contrasts[0])
        with self.assertRaisesRegex(ValueError,'Duplicate contrast'): self.check()
    def test_unsupported_seed_rejected(self):
        self.ap[0]['seed']='3'
        with self.assertRaisesRegex(ValueError,'Unexpected'): self.check()
    def test_incomplete_pass_accounting_rejected(self):
        self.costs[-1]['detector_passes']='11447'
        with self.assertRaisesRegex(ValueError,'pass count'): self.check()
    def test_wrong_pixel_total_rejected(self):
        self.costs[0]['processed_input_pixels']='1'
        with self.assertRaisesRegex(ValueError,'Pixel accounting'): self.check()
    def test_misleading_timing_order_rejected(self):
        self.costs[0]['diagnostic_median_ms']='99999'
        with self.assertRaisesRegex(ValueError,'diagnostic timing'): self.check()



class SummaryControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        pairs=[]
        for seed in range(3):
            pair={'seed':seed}
            for split in ('development','testdev'):
                summary={}
                for method in m.METHODS:
                    size={'full_640':640,'full_960':960,'full_1280':1280,m.METHODS[-1]:640}[method]
                    passes=11448 if method==m.METHODS[-1] else 1610
                    summary[method]={'scoring':{'target_count':38759 if split=='development' else 75102,
                        'metrics':{'metrics/mAP50-95(B)':.2+seed*.001}},'detector_passes':passes,
                        'processed_input_pixels':passes*size**2,'diagnostic_median_ms':1.,'diagnostic_p95_ms':2.}
                path=self.root/f'{seed}_{split}.json'
                raw=json.dumps(summary).encode('utf-8'); path.write_bytes(raw)
                pair[split+'_summary']=path.name
                pair[split+'_sha256']=hashlib.sha256(raw).hexdigest()
            pairs.append(pair)
        self.index=self.root/'index.json'; self.value={'summaries':pairs}; self.save()
    def save(self): self.index.write_text(json.dumps(self.value),encoding='utf-8')
    def test_index_and_parsed_summary_hashes(self):
        ap,c,costs,hashes=m.from_index(self.index)
        self.assertEqual((len(ap),len(c),len(costs)),(12,6,12))
        self.assertEqual(hashes['summary_index'],m.sha(self.index))
    def test_wrong_hash_rejected(self):
        self.value['summaries'][0]['testdev_sha256']='0'*64; self.save()
        with self.assertRaisesRegex(ValueError,'bytes differ'): m.from_index(self.index)
    def test_wrong_development_task_rejected_despite_valid_hash(self):
        pair=self.value['summaries'][0]; path=self.root/pair['development_summary']
        value=json.loads(path.read_text()); value['full_640']['scoring']['target_count']=1
        raw=json.dumps(value).encode(); path.write_bytes(raw)
        pair['development_sha256']=hashlib.sha256(raw).hexdigest(); self.save()
        with self.assertRaisesRegex(ValueError,'development target count'): m.from_index(self.index)
    def test_duplicate_seed_rejected(self):
        self.value['summaries'][-1]=copy.deepcopy(self.value['summaries'][0]); self.save()
        with self.assertRaisesRegex(ValueError,'duplicate seed'): m.from_index(self.index)
    def test_missing_summary_pair_rejected(self):
        self.value['summaries'].pop(); self.save()
        with self.assertRaisesRegex(ValueError,'three summary pairs'): m.from_index(self.index)
    def test_existing_output_sentinel_is_preserved(self):
        output=self.root/'existing'; output.mkdir(); sentinel=output/'keep.txt'; sentinel.write_text('keep')
        with mock.patch.object(sys,'argv',['renderer','--summary-index',str(self.index),'--output-dir',str(output)]):
            with self.assertRaisesRegex(ValueError,'Preserve existing'): m.main()
        self.assertEqual(sentinel.read_text(),'keep')
        self.assertEqual({p.name for p in output.iterdir()},{'keep.txt'})

if __name__=='__main__': unittest.main()
