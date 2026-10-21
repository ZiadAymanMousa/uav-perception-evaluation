# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Synthetic CPU controls for strict content, source and result guards."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import (load_task, read_json, write_json, digest, snapshot_sources,
                    source_hashes, object_sha, runtime_context, offline_environment, prediction_arguments, validate_versions, parameter_device)
import data
from checkpoints import checkpoint_context
from records import score_log, validate_row, windows_for, methods_for_index, verify_run_files

class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='testdev_guards_')
        self.root = Path(self.temp.name)
        offline_environment(self.root/'runtime')
        self.addCleanup(self.temp.cleanup)
        self.raw = self.root/'raw'
        (self.raw/'images').mkdir(parents=True)
        (self.raw/'annotations').mkdir()
        from PIL import Image
        for index in range(2):
            Image.new('RGB',(16,16),(index*50,20,30)).save(self.raw/'images'/f'i{index}.jpg')
            (self.raw/'annotations'/f'i{index}.txt').write_text('1,2,5,6,1,1,0,0\n0,0,2,2,0,0,0,0\n')
        self.task = read_json(ROOT/'configs/testdev_task.json')
        entries = [data.convert_entry(self.raw/'images'/f'i{i}.jpg',self.raw/'annotations'/f'i{i}.txt')[0] for i in range(2)]
        self.task['expected_identity_sha256'] = data.semantic_identity(entries)
        self.task['expected_counts'] = dict(images=2,raw_rows=4,targets=2,excluded=2,duplicates=0,clipped=0,malformed_exceptions=0)
        self.task['complete_accounting'] = {'records':8,'detector_passes':{'full_640':2,'full_960':2,'full_1280':2,'fixed_slices_640_plus_global':4},'warmup_calls':15}
        self.config = self.root/'task.json'
        write_json(self.config,self.task)
        self.prepared = self.root/'prepared'
        data.prepare(self.raw,self.prepared,self.config,self.task)
        self.weights = self.root/'selected.pt'
        self.weights.write_bytes(b'fixture-only-not-a-model')
        self.task['checkpoints']['seed0']['weights_sha256'] = digest(self.weights)

    def verify(self):
        return data.verify(self.raw,self.prepared,self.config,self.task)

    def entries(self):
        return read_json(self.prepared/'manifest.json')['entries']

    def rows(self):
        rows = []
        for index,entry in enumerate(self.entries()):
            for method in methods_for_index(self.task,index):
                windows = windows_for(entry,method)
                timing = {'image':entry['image'],'method':method['name'],'decode_ms':1.,'views_ms':1.,'merge_ms':1.,'end_to_end_ms':3.,
                    'detector_passes':len(windows),'processed_input_pixels':len(windows)*method['input_size']**2,'prediction_count':0,
                    'geometry':{'clipped':0,'zero_area_removed':0},
                    'per_view_counts':[{'window_xyxy':[w.x1,w.y1,w.x2,w.y2],'output_before_geometry':0,'output_after_geometry':0} for w in windows],
                    'merge_counts':{'before_merge':0,'after_nms_before_cap':0,'after_cap':0,'removed_nms':0,'removed_cap':0}}
                rows.append({'image':entry['image'],'method':method['name'],'image_sha256':entry['image_sha256'],'timing':timing,'predictions':[]})
        return rows

    def log(self,rows):
        path = self.root/'predictions.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        return path

    def score(self,rows,smoke=False):
        return score_log(self.log(rows),self.prepared,self.task,smoke=smoke,
                         scorer=lambda rows,names: {'target_count':sum(len(r['gt']) for r in rows)})

    def run_record(self):
        run = self.root/'run'
        run.mkdir()
        shutil.copyfile(self.config,run/'task_config.json')
        write_json(run/'selection.json',[e['image'] for e in self.entries()])
        shutil.copyfile(self.log(self.rows()),run/'predictions_and_timings.jsonl')
        write_json(run/'summary.json',{})
        context = self.verify()
        checkpoint = checkpoint_context(self.task,self.weights,'frozen-replay',seed='seed0')
        record = {'schema':'uvd1-portable-testdev-run-v1','status':'completed','scope':'complete-testdev','data_context':context,
                  'checkpoint_context':checkpoint,'config_sha256':digest(self.config),'source_hashes':snapshot_sources(run/'source_snapshot'),
                  'prediction_log_sha256':digest(run/'predictions_and_timings.jsonl'),'summary_sha256':digest(run/'summary.json'),
                  'selection_sha256':digest(run/'selection.json'),'selection':[e['image'] for e in self.entries()],
                  'prepared_root':str(self.prepared),'images_compared':2,'warmup_calls':15,'device_requested':'cpu',
                  'accounting':{'images':2,'records':8},'actual_parameter_device':'cpu',
                  'predict_arguments':prediction_arguments(self.task,'cpu'),
                  'environment':{'versions':copy.deepcopy(self.task['runtime_versions']),
                                 'framework_source_hashes':copy.deepcopy(self.task['framework_source_hashes']),
                                 'torch_num_threads':2}}
        write_json(run/'run_record.json',record)
        write_json(run/'progress.json',{'status':'completed','images_completed':2,'images_planned':2,'records_written':8})
        return run,record,context,checkpoint

class DataGuards(Fixture):
    def test_valid_preparation_and_original_ignored_rows(self):
        self.assertEqual(self.verify()['counts']['targets'],2)
        side = read_json(self.prepared/'excluded_annotations/testdev/i0.json')
        self.assertEqual(side['not_scored_targets'][0]['original_fields'],['0','0','2','2','0','0','0','0'])
        self.assertEqual(side['not_scored_targets'][0]['line_number'],2)
        self.assertEqual((self.raw/'annotations/i0.txt').read_bytes(),(self.prepared/'source_annotations/testdev/i0.txt').read_bytes())

    def test_relocated_raw_and_prepared_semantics(self):
        relocated_raw, relocated = self.root/'raw2',self.root/'prepared2'
        shutil.copytree(self.raw,relocated_raw)
        shutil.copytree(self.prepared,relocated)
        self.assertEqual(data.verify(relocated_raw,relocated,self.config,self.task),self.verify())

    def test_changed_raw_image_rejected(self):
        with (self.raw/'images/i0.jpg').open('ab') as f: f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'content identity'): self.verify()

    def test_changed_ignored_annotation_rejected(self):
        p=self.raw/'annotations/i0.txt';p.write_text(p.read_text().replace('0,0,2,2,0,0','0,0,2,2,0,11'))
        with self.assertRaisesRegex(ValueError,'content identity'): self.verify()

    def test_malformed_ignored_row_rejected(self):
        p=self.raw/'annotations/i0.txt';p.write_text(p.read_text().replace('0,0,2,2,0,0','0,0,0,2,0,0'))
        with self.assertRaisesRegex(ValueError,'nonpositive'): self.verify()

    def test_missing_raw_annotation_rejected(self):
        (self.raw/'annotations/i0.txt').unlink()
        with self.assertRaisesRegex(ValueError,'membership'): self.verify()

    def test_extra_raw_annotation_rejected(self):
        (self.raw/'annotations/extra.txt').write_text('')
        with self.assertRaisesRegex(ValueError,'membership'): self.verify()

    def test_duplicate_label_not_silently_accepted(self):
        p=self.raw/'annotations/i0.txt';p.write_text(p.read_text()+'1,2,5,6,1,1,0,0\n')
        with self.assertRaisesRegex(ValueError,'totals differ'): self.verify()

    def test_changed_label_rejected(self):
        (self.prepared/'labels/testdev/i0.txt').write_text('0 0.3 0.3 0.1 0.1\n')
        with self.assertRaisesRegex(ValueError,'label bytes'): self.verify()

    def test_changed_sidecar_rejected(self):
        p=self.prepared/'excluded_annotations/testdev/i0.json';s=read_json(p);s['not_scored_targets']=[];write_json(p,s)
        with self.assertRaisesRegex(ValueError,'sidecar'): self.verify()

    def test_changed_retained_annotation_rejected(self):
        (self.prepared/'source_annotations/testdev/i0.txt').write_text('')
        with self.assertRaisesRegex(ValueError,'original annotation'): self.verify()

    def test_changed_source_snapshot_rejected(self):
        p=self.prepared/'source_snapshot/src/data.py';p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'snapshot'): self.verify()

    def test_current_source_change_rejected(self):
        with patch('common.source_hashes',return_value={'changed':'source'}):
            with self.assertRaisesRegex(ValueError,'Current source'): self.verify()

    def test_changed_config_rejected(self):
        self.config.write_text(self.config.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'config'): self.verify()

    def test_changed_binding_rejected(self):
        p=self.prepared/'binding.json';s=read_json(p);s['content_identity_sha256']='0'*64;write_json(p,s)
        with self.assertRaisesRegex(ValueError,'semantic binding'): self.verify()

    def test_forged_complete_manifest_rejected(self):
        p=self.prepared/'manifest.json';s=read_json(p);s['entries']=s['entries'][:1];write_json(p,s)
        b=read_json(self.prepared/'binding.json');b['manifest_sha256']=digest(p);write_json(self.prepared/'binding.json',b)
        with self.assertRaisesRegex(ValueError,'reconstruction'): self.verify()

    def test_failed_manifest_rejected(self):
        p=self.prepared/'manifest.json';s=read_json(p);s['status']='failed';write_json(p,s)
        with self.assertRaisesRegex(ValueError,'not complete'): self.verify()

    def test_no_overwrite_preparation(self):
        with self.assertRaisesRegex(ValueError,'exists'): data.prepare(self.raw,self.prepared,self.config,self.task)

    def test_unreviewed_config_not_accepted_by_cli_loader(self):
        with self.assertRaisesRegex(ValueError,'release-time content pin'): load_task(self.config)

class ResultGuards(Fixture):
    def test_valid_complete_matrix(self):
        summary, accounting=self.score(self.rows())
        self.assertEqual(accounting['records'],8)
        self.assertEqual(summary['full_640']['scoring']['target_count'],2)

    def test_valid_one_image_smoke_distinct_scope(self):
        _,accounting=self.score(self.rows()[:4],smoke=True)
        self.assertEqual(accounting['scope'],'one-image-cpu-smoke')
        self.assertEqual(accounting['images'],1)

    def test_smoke_cannot_be_complete(self):
        with self.assertRaisesRegex(ValueError,'Missing'):self.score(self.rows()[:4])

    def test_missing_image_method_row(self):
        rows=self.rows();del rows[2]
        with self.assertRaisesRegex(ValueError,'Missing, duplicate'):self.score(rows)

    def test_duplicate_method_row(self):
        rows=self.rows();rows[1]=copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError,'Missing, duplicate'):self.score(rows)

    def test_extra_row_rejected(self):
        rows=self.rows();rows.append(rows[0])
        with self.assertRaisesRegex(ValueError,'Extra or duplicate'):self.score(rows)

    def test_changed_image_identity_rejected(self):
        rows=self.rows();rows[0]['image_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'image content'):self.score(rows)

    def test_invalid_prediction_geometry_rejected(self):
        rows=self.rows();rows[0]['predictions']=[[0,0,17,2,.5,0]]
        with self.assertRaisesRegex(ValueError,'outside image'):self.score(rows)

    def test_nonfinite_prediction_rejected(self):
        rows=self.rows();rows[0]['predictions']=[[0,0,2,2,float('nan'),0]]
        with self.assertRaisesRegex(ValueError,'finite'):self.score(rows)

    def test_nonintegral_prediction_class_rejected(self):
        rows=self.rows();rows[0]['predictions']=[[0,0,2,2,.5,.3]]
        with self.assertRaisesRegex(ValueError,'zero-based'):self.score(rows)

    def test_unsorted_predictions_rejected(self):
        rows=self.rows();rows[0]['predictions']=[[0,0,2,2,.4,0],[1,1,3,3,.5,0]]
        with self.assertRaisesRegex(ValueError,'confidence sorted'):self.score(rows)

    def test_cap_excess_rejected(self):
        rows=self.rows();rows[0]['predictions']=[[0,0,2,2,.5,0]]*1001
        with self.assertRaisesRegex(ValueError,'cap exceeded'):self.score(rows)

    def test_malformed_empty_prediction_rows_rejected(self):
        rows=self.rows();rows[0]['predictions']=[[]]
        with self.assertRaisesRegex(ValueError,'Nx6'):self.score(rows)

    def test_changed_pass_count_rejected(self):
        rows=self.rows();rows[0]['timing']['detector_passes']=99
        with self.assertRaisesRegex(ValueError,'pass/input-pixel'):self.score(rows)

    def test_changed_window_rejected(self):
        rows=self.rows();rows[0]['timing']['per_view_counts'][0]['window_xyxy']=[1,1,16,16]
        with self.assertRaisesRegex(ValueError,'View geometry'):self.score(rows)

    def test_inconsistent_merge_rejected(self):
        rows=self.rows();rows[0]['timing']['merge_counts']['before_merge']=1
        with self.assertRaisesRegex(ValueError,'Merge count'):self.score(rows)

    def test_changed_timing_boundary_rejected(self):
        rows=self.rows();rows[0]['timing']['end_to_end_ms']=999
        with self.assertRaisesRegex(ValueError,'Timing boundary'):self.score(rows)

    def test_frozen_checkpoint_changed_rejected(self):
        self.weights.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Checkpoint differs'):checkpoint_context(self.task,self.weights,'frozen-replay',seed='seed0')

    def test_historical_seed_forbidden_on_prospective_route(self):
        with self.assertRaisesRegex(ValueError,'separate actual'):checkpoint_context(self.task,self.weights,'prospective-completed-baseline',seed='seed0')

    def test_prospective_uses_training_context_not_testdev(self):
        actual_training={'training':'explicit separate task'}
        with patch('execution_inputs.data_context',return_value=actual_training) as ctx, patch('execution_inputs.completed_checkpoint',side_effect=ValueError('Baseline training is incomplete')) as guard:
            with self.assertRaisesRegex(ValueError,'training is incomplete'):
                checkpoint_context(self.task,self.weights,'prospective-completed-baseline',baseline_run=self.root/'baseline',
                                   training_data=self.root/'training',training_config=self.root/'training_task.json',training_binding=self.root/'training_binding.json')
            ctx.assert_called_once_with(self.root/'training',self.root/'training_task.json',self.root/'training_binding.json')
            guard.assert_called_once_with(self.root/'baseline',self.weights,context=actual_training)

    def test_valid_run_metadata(self):
        run,record,context,checkpoint=self.run_record()
        self.assertFalse(verify_run_files(run,self.task,context,checkpoint,prepared=self.prepared)[1])

    def test_changed_run_predictions_rejected(self):
        run,record,context,checkpoint=self.run_record()
        with (run/'predictions_and_timings.jsonl').open('a') as f:f.write('\n')
        with self.assertRaisesRegex(ValueError,'Predictions changed'):verify_run_files(run,self.task,context,checkpoint,prepared=self.prepared)

    def test_false_complete_progress_rejected(self):
        run,record,context,checkpoint=self.run_record()
        write_json(run/'progress.json',{'status':'completed','images_completed':1,'images_planned':2,'records_written':4})
        with self.assertRaisesRegex(ValueError,'False complete'):verify_run_files(run,self.task,context,checkpoint,prepared=self.prepared)

    def test_running_status_cannot_be_complete(self):
        run,record,context,checkpoint=self.run_record();record['status']='running';write_json(run/'run_record.json',record)
        with self.assertRaisesRegex(ValueError,'not completed'):verify_run_files(run,self.task,context,checkpoint,prepared=self.prepared)

    def test_changed_run_config_snapshot_rejected(self):
        run,record,context,checkpoint=self.run_record();(run/'task_config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'config snapshot'):verify_run_files(run,self.task,context,checkpoint,prepared=self.prepared)

    def test_changed_checkpoint_context_rejected(self):
        run,record,context,checkpoint=self.run_record();checkpoint['weights_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Checkpoint/training'):verify_run_files(run,self.task,context,checkpoint,prepared=self.prepared)

    def test_unsupported_framework_rejected(self):
        task=copy.deepcopy(self.task);task['runtime_versions']['ultralytics']='0.0.0'
        with self.assertRaisesRegex(ValueError,'Unsupported Ultralytics'):runtime_context(task)

    def test_same_pinned_scorer_cpu_positive_control(self):
        from score_converted_predictions import score_records
        import numpy as np
        result=score_records([{'gt':np.array([[0,1,2,6,8]],dtype=np.float32),
                               'predictions':np.array([[1,2,6,8,.99,0]],dtype=np.float32)}],dict(enumerate(self.task['class_names'])))
        self.assertEqual(result['target_count'],1)
        self.assertGreater(result['metrics']['metrics/mAP50-95(B)'],.99)


class ReviewCorrections(Fixture):
    def test_run_verification_uses_relocated_verified_preparation(self):
        run,record,context,checkpoint=self.run_record()
        relocated=self.root/'relocated_prepared'
        self.prepared.rename(relocated)
        self.assertFalse(self.prepared.exists())
        verified=data.verify(self.raw,relocated,self.config,self.task)
        self.assertEqual(context,verified)
        self.assertFalse(verify_run_files(run,self.task,verified,checkpoint,prepared=relocated)[1])

    def test_changed_recorded_prediction_policy_rejected(self):
        run,record,context,checkpoint=self.run_record()
        record['predict_arguments']['conf']=.25
        write_json(run/'run_record.json',record)
        with self.assertRaisesRegex(ValueError,'prediction arguments'):
            verify_run_files(run,self.task,context,checkpoint,prepared=self.prepared)

    def test_changed_recorded_precision_rejected(self):
        run,record,context,checkpoint=self.run_record()
        record['predict_arguments']['half']=True
        write_json(run/'run_record.json',record)
        with self.assertRaisesRegex(ValueError,'prediction arguments'):
            verify_run_files(run,self.task,context,checkpoint,prepared=self.prepared)

    def test_changed_recorded_runtime_version_rejected(self):
        run,record,context,checkpoint=self.run_record()
        record['environment']['versions']['torch']='0.0.0'
        write_json(run/'run_record.json',record)
        with self.assertRaisesRegex(ValueError,'Unsupported torch'):
            verify_run_files(run,self.task,context,checkpoint,prepared=self.prepared)

    def test_changed_recorded_framework_hash_rejected(self):
        run,record,context,checkpoint=self.run_record()
        record['environment']['framework_source_hashes']['metrics']='0'*64
        write_json(run/'run_record.json',record)
        with self.assertRaisesRegex(ValueError,'Recorded framework hashes'):
            verify_run_files(run,self.task,context,checkpoint,prepared=self.prepared)

    def test_cpu_wheel_version_allowance_is_explicit(self):
        versions=copy.deepcopy(self.task['runtime_versions'])
        versions['torch']=versions['torch'].split('+')[0]+'+cpu'
        versions['torchvision']=versions['torchvision'].split('+')[0]+'+cpu'
        validate_versions(versions,self.task,exact_inference=False)
        with self.assertRaisesRegex(ValueError,'Unsupported torch'):
            validate_versions(versions,self.task,exact_inference=True)

    def test_recorded_parameter_device_mismatch_rejected(self):
        run,record,context,checkpoint=self.run_record()
        record['actual_parameter_device']='cuda:0'
        write_json(run/'run_record.json',record)
        with self.assertRaisesRegex(ValueError,'Actual parameter device'):
            verify_run_files(run,self.task,context,checkpoint,prepared=self.prepared)

    def test_cuda_device_is_local_index_after_pinned_selection(self):
        self.assertEqual(parameter_device('cpu'),'cpu')
        self.assertEqual(parameter_device('0'),'cuda:0')
        self.assertEqual(parameter_device('1'),'cuda:0')
        self.assertEqual(parameter_device('cuda:3'),'cuda:0')
        with self.assertRaisesRegex(ValueError,'one explicit CUDA'):
            parameter_device('0,1')

    def test_per_view_cap_even_when_merge_totals_consistent(self):
        entry=self.entries()[0]
        method=self.task['inference']['methods'][0]
        row=self.rows()[0]
        row['predictions']=[[0,0,2,2,.5,0]]*1000
        timing=row['timing']
        timing['prediction_count']=1000
        timing['per_view_counts'][0].update(output_before_geometry=1001,output_after_geometry=1001)
        timing['merge_counts'].update(before_merge=1001,after_nms_before_cap=1001,after_cap=1000,removed_nms=0,removed_cap=1)
        with self.assertRaisesRegex(ValueError,'per-view prediction cap'):
            validate_row(row,entry,method,self.task)

    def test_config_hash_and_parse_share_one_byte_buffer(self):
        target=self.root/'reviewed_config.json'
        shutil.copyfile(ROOT/'configs/testdev_task.json',target)
        original=Path.read_bytes
        def read_then_change(path):
            raw=original(path)
            if path==target:
                path.write_bytes(b'{"schema":"changed-after-read"}')
            return raw
        with patch.object(Path,'read_bytes',autospec=True,side_effect=read_then_change) as reader:
            task=load_task(target)
        reader.assert_called_once_with(target)
        self.assertEqual(task['schema'],'uvd1-portable-testdev-task-v1')
        with self.assertRaisesRegex(ValueError,'release-time content pin'):
            load_task(target)

    def assert_cli_rejects_before_runtime(self,extra,pattern):
        import testdev
        args=['testdev.py','rescore-log','--raw-root',str(self.raw),'--prepared',str(self.prepared),
              '--config',str(self.config),'--weights',str(self.weights),'--mode','frozen-replay','--seed','seed0']+extra
        before={str(p.relative_to(self.raw)):p.read_bytes() for p in self.raw.rglob('*') if p.is_file()}
        with patch.object(sys,'argv',args),patch('testdev.offline_environment') as offline,patch('testdev.runtime_context') as runtime,patch('testdev.load_task') as task:
            with self.assertRaisesRegex(ValueError,pattern):testdev.main()
            offline.assert_not_called();runtime.assert_not_called();task.assert_not_called()
        after={str(p.relative_to(self.raw)):p.read_bytes() for p in self.raw.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_rescore_output_in_raw_rejected_before_runtime(self):
        self.assert_cli_rejects_before_runtime(['--output',str(self.raw/'blocked/score.json')],'outside input')
        self.assertFalse((self.raw/'blocked').exists())

    def test_rescore_output_in_prepared_rejected_before_runtime(self):
        self.assert_cli_rejects_before_runtime(['--output',str(self.prepared/'blocked/score.json')],'outside input')
        self.assertFalse((self.prepared/'blocked').exists())

    def test_existing_rescore_output_rejected_before_runtime(self):
        output=self.root/'old_score.json';output.write_bytes(b'original retained result')
        self.assert_cli_rejects_before_runtime(['--output',str(output)],'Output exists')
        self.assertEqual(output.read_bytes(),b'original retained result')

    def test_cache_in_raw_rejected_before_runtime(self):
        predictions=self.log(self.rows())
        self.assert_cli_rejects_before_runtime(['--output',str(self.root/'score.json'),
            '--predictions',str(predictions),'--cache-root',str(self.raw/'blocked_cache')],'Runtime cache must be outside')
        self.assertFalse((self.raw/'blocked_cache').exists())

    def test_cache_settings_subdirectory_cannot_overlap_raw(self):
        from argparse import Namespace
        from testdev import validate_paths
        args=Namespace(raw_root=self.root/'runtime_settings',prepared=self.prepared,training_data=None,
            baseline_run=None,operation='preflight',output=None,predictions=None,run=None,smoke=False,cache_root=self.root)
        with self.assertRaisesRegex(ValueError,'Runtime settings must be outside'):
            validate_paths(args)

if __name__=='__main__':
    unittest.main(verbosity=2)
