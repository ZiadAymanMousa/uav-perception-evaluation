# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Meaningful CPU controls for prospective completion and exact resume gates."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'src'))
from audit_dataset import digest
from execution_inputs import (EXECUTION_SCHEMA, assert_same_launch_data, assert_resume_inputs,
                              checked_resume_checkpoint, completed_checkpoint,
                              review_training_history, safe_id)
from portable_preparer import write_json
from study_inputs import native_detections
from rescore_saved import validate_scoring_runtime, FRAMEWORK_FILES


def context():
    return {'binding_sha256':'binding-A','verified':{'config_sha256':'config-A',
        'dataset_manifest_sha256':'manifest-A','source_audit_sha256':'audit-A',
        'dataset_yaml_sha256':'yaml-A','semantic_identity_sha256':'task-A',
        'implementation_sha256':{'verifier':'v1'}}}


class ExecutionGuardTests(unittest.TestCase):
    def setUp(self):
        parent=HERE/'test_artifacts';parent.mkdir(exist_ok=True)
        self.run=Path(tempfile.mkdtemp(prefix=self._testMethodName+'_',dir=parent))
        (self.run/'source_snapshot').mkdir();(self.run/'fit/weights').mkdir(parents=True)
        write_json(self.run/'frozen_config.json',{'training':{'epochs':2}})
        (self.run/'source_snapshot/train_full_baseline.py').write_text('reviewed runner',encoding='utf-8')
        (self.run/'source_snapshot/execution_inputs.py').write_text('reviewed helper',encoding='utf-8')
        self.weights=self.run/'fit/weights/best.pt';self.weights.write_bytes(b'synthetic binary; not a model')
        (self.run/'fit/results.csv').write_text('epoch,metrics/mAP50-95(B),train/box_loss\n1,0.1,2.0\n2,0.2,1.0\n')
        events=[{'epochs_completed':i,'metrics':{'metrics/mAP50-95(B)':v},'loss_items':[1.]}
                for i,v in [(1,.1),(2,.2),(2,.2)]]
        (self.run/'epoch_records.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        write_json(self.run/'progress.json',{'epochs_completed':2,'epochs_planned':2})
        self.context=context()
        self.record={'status':'completed','execution_schema':EXECUTION_SCHEMA,
            'dataset_context':self.context,'config_sha256':digest(self.run/'frozen_config.json'),
            'runner_sha256':digest(self.run/'source_snapshot/train_full_baseline.py'),
            'manifest_sha256':'manifest-A','output_hashes':{'fit/weights/best.pt':digest(self.weights)},
            'source_hashes':{'train_full_baseline.py':digest(self.run/'source_snapshot/train_full_baseline.py'),
                             'execution_inputs.py':digest(self.run/'source_snapshot/execution_inputs.py')}}
        self.save_record()

    def save_record(self):write_json(self.run/'run_record.json',self.record)
    def check(self):return completed_checkpoint(self.run,self.weights,context=self.context)

    def test_completed_portable_checkpoint_and_history_pass(self):
        self.assertEqual(self.check()['status'],'completed')
        self.assertEqual(review_training_history(self.run),{'planned_epochs':2,'csv_epochs':2,'callback_events':3})

    def test_incomplete_training_rejected(self):
        self.record['status']='failed';self.save_record()
        with self.assertRaisesRegex(ValueError,'incomplete'):self.check()

    def test_progress_planned_count_cannot_fake_shortened_run(self):
        write_json(self.run/'progress.json',{'epochs_completed':1,'epochs_planned':1})
        with self.assertRaisesRegex(ValueError,'epochs differ'):self.check()

    def test_noncontiguous_training_csv_rejected(self):
        (self.run/'fit/results.csv').write_text('epoch,score\n1,0.1\n1,0.2\n')
        with self.assertRaisesRegex(ValueError,'noncontiguous'):self.check()

    def test_nonfinite_training_csv_rejected(self):
        (self.run/'fit/results.csv').write_text('epoch,score\n1,nan\n2,0.2\n')
        with self.assertRaisesRegex(ValueError,'Nonfinite training CSV'):self.check()

    def test_missing_posttraining_validation_event_rejected(self):
        path=self.run/'epoch_records.jsonl';path.write_text('\n'.join(path.read_text().splitlines()[:2])+'\n')
        with self.assertRaisesRegex(ValueError,'posttraining'):self.check()

    def test_missing_epoch_callback_rejected(self):
        path=self.run/'epoch_records.jsonl';lines=path.read_text().splitlines();path.write_text('\n'.join(lines[1:])+'\n')
        with self.assertRaisesRegex(ValueError,'callback history'):self.check()

    def test_wrong_checkpoint_rejected(self):
        self.weights.write_bytes(b'different checkpoint')
        with self.assertRaisesRegex(ValueError,'selected'):self.check()

    def test_changed_frozen_config_rejected(self):
        write_json(self.run/'frozen_config.json',{'training':{'epochs':1}})
        with self.assertRaisesRegex(ValueError,'config changed'):self.check()

    def test_changed_runner_snapshot_rejected(self):
        (self.run/'source_snapshot/train_full_baseline.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'runner snapshot'):self.check()

    def test_changed_helper_snapshot_rejected(self):
        (self.run/'source_snapshot/execution_inputs.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'helper snapshot'):self.check()

    def test_legacy_checkpoint_cannot_become_development_study(self):
        self.record.pop('execution_schema');self.save_record()
        with self.assertRaisesRegex(ValueError,'prospective portable'):self.check()
        self.assertEqual(completed_checkpoint(self.run,self.weights,require_portable=False)['status'],'completed')

    def test_same_semantics_different_manifest_cannot_resume(self):
        actual=copy.deepcopy(self.context);actual['verified']['dataset_manifest_sha256']='manifest-B'
        with self.assertRaisesRegex(ValueError,'manifest'):assert_same_launch_data(self.context,actual)

    def test_binding_record_change_cannot_resume(self):
        actual=copy.deepcopy(self.context);actual['binding_sha256']='binding-B'
        with self.assertRaisesRegex(ValueError,'binding'):assert_same_launch_data(self.context,actual)

    def test_same_semantics_different_yaml_cannot_resume(self):
        actual=copy.deepcopy(self.context);actual['verified']['dataset_yaml_sha256']='yaml-B'
        with self.assertRaisesRegex(ValueError,'yaml'):assert_same_launch_data(self.context,actual)

    def resume_record(self):
        return {'status':'failed','execution_schema':EXECUTION_SCHEMA,'dataset_context':self.context,
                'config_sha256':'cfg','runner_sha256':'runner','checkpoint_source_sha256':'initial',
                'source_hashes':{'helper':'h'}}

    def resume(self,record,**overrides):
        args={'context':self.context,'config_sha256':'cfg','runner_sha256':'runner',
              'initial_weights_sha256':'initial','source_hashes':{'helper':'h'}}
        args.update(overrides);return assert_resume_inputs(record,**args)

    def test_failed_portable_run_with_exact_inputs_may_pass_resume_guard(self):
        self.resume(self.resume_record())

    def test_completed_run_cannot_resume(self):
        rec=self.resume_record();rec['status']='completed'
        with self.assertRaisesRegex(ValueError,'Completed'):self.resume(rec)

    def test_changed_training_settings_cannot_resume(self):
        with self.assertRaisesRegex(ValueError,'config_sha256'):self.resume(self.resume_record(),config_sha256='other')

    def test_changed_starting_weights_cannot_resume(self):
        with self.assertRaisesRegex(ValueError,'checkpoint_source'):self.resume(self.resume_record(),initial_weights_sha256='other')

    def test_changed_source_helper_cannot_resume(self):
        with self.assertRaisesRegex(ValueError,'source_hashes'):self.resume(self.resume_record(),source_hashes={'helper':'changed'})

    def resume_checkpoint(self):
        checkpoint=self.run/'fit/weights/last.pt'
        checkpoint.write_bytes(b'independently specified saved training state')
        write_json(self.run/'progress.json',{'epochs_completed':1,'epochs_planned':2,
                                           'last_checkpoint_sha256':digest(checkpoint)})
        return checkpoint

    def test_recorded_exact_last_checkpoint_passes_resume_state_guard(self):
        checkpoint=self.resume_checkpoint()
        self.assertEqual(checked_resume_checkpoint(self.run),(checkpoint,digest(checkpoint)))

    def test_replaced_last_checkpoint_cannot_resume(self):
        checkpoint=self.resume_checkpoint()
        checkpoint.write_bytes(b'another otherwise available training state')
        with self.assertRaisesRegex(ValueError,'checkpoint bytes changed'):
            checked_resume_checkpoint(self.run)

    def test_unrecorded_last_checkpoint_requires_review(self):
        self.resume_checkpoint()
        write_json(self.run/'progress.json',{'epochs_completed':1,'epochs_planned':2})
        with self.assertRaisesRegex(ValueError,'retained last-checkpoint checksum'):
            checked_resume_checkpoint(self.run)

    def test_missing_last_checkpoint_cannot_resume(self):
        with self.assertRaisesRegex(ValueError,'No checkpoint to resume'):
            checked_resume_checkpoint(self.run)

    def test_unsafe_run_identifiers_rejected(self):
        for value in ['..','../x','a/b','a\\b','C:drive']:
            with self.assertRaises(ValueError):safe_id(value)

    def test_native_bridge_keeps_fractional_geometry_class_and_score_order(self):
        np.testing.assert_allclose(native_detections([[3.25,7.5,10.75,17.5,.9,9]]),
                                   [[3.25,7.5,7.5,10.,.9,10.,-1.,-1.]],rtol=0,atol=1e-7)
        with self.assertRaises(ValueError):native_detections([[0,0,1,1,.1,0],[0,0,1,1,.9,1]])


class ScoringRuntimeGuardTests(unittest.TestCase):
    def setUp(self):
        self.record={'ultralytics':'8.3.253','torch':'2.10.0+cu128',
                     'framework_source_hashes':{name:'original-'+name for name in FRAMEWORK_FILES}}
        self.observed=copy.deepcopy(self.record)

    def test_same_versions_and_exact_framework_sources_pass(self):
        result=validate_scoring_runtime(self.record,observed=self.observed)
        self.assertTrue(result['installed_versions_and_framework_sources_match_prediction_run'])

    def test_different_installed_ultralytics_rejected(self):
        self.observed['ultralytics']='different-version'
        with self.assertRaisesRegex(ValueError,'version changed: ultralytics'):
            validate_scoring_runtime(self.record,observed=self.observed)

    def test_different_installed_torch_rejected(self):
        self.observed['torch']='different-version'
        with self.assertRaisesRegex(ValueError,'version changed: torch'):
            validate_scoring_runtime(self.record,observed=self.observed)

    def test_same_version_modified_matcher_source_rejected(self):
        self.observed['framework_source_hashes']['matching']='modified-same-version'
        with self.assertRaisesRegex(ValueError,'framework source changed'):
            validate_scoring_runtime(self.record,observed=self.observed)

    def test_same_version_modified_ap_source_rejected(self):
        self.observed['framework_source_hashes']['metrics']='modified-same-version'
        with self.assertRaisesRegex(ValueError,'framework source changed'):
            validate_scoring_runtime(self.record,observed=self.observed)

    def test_missing_recorded_source_identity_rejected(self):
        self.record['framework_source_hashes'].pop('matching')
        with self.assertRaisesRegex(ValueError,'Missing/unknown'):
            validate_scoring_runtime(self.record,observed=self.observed)


if __name__=='__main__':unittest.main(verbosity=2)
