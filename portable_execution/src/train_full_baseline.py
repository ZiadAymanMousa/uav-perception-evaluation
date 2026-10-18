# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Run a frozen full-data baseline locally; persist live epoch/checkpoint evidence."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time

from audit_dataset import digest
from execution_inputs import (EXECUTION_SCHEMA, data_context, assert_resume_inputs,
                              checked_resume_checkpoint, safe_id)

PROJECT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT / 'artifacts/ultralytics_config'
(CONFIG_DIR / 'Ultralytics').mkdir(parents=True, exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(CONFIG_DIR)
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['TORCH_HOME'] = str(PROJECT / 'artifacts/torch_cache')
os.environ['MPLCONFIGDIR'] = str(PROJECT / 'artifacts/matplotlib_config')


def write_json(path, value):
    partial = path.with_suffix('.tmp')
    partial.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT/'configs/baseline_portable_seed0.json')
    parser.add_argument('--resume',action='store_true',help='Use only after the old process is verified terminal')
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--task-config',type=Path,required=True)
    parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--initial-weights',type=Path,required=True)
    parser.add_argument('--initial-weights-metadata',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true',help='CPU data/source preflight; no model loading/training')
    cli = parser.parse_args()
    config_path = cli.config.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    run_id = safe_id(config['run_id'])
    run = PROJECT / 'artifacts' / run_id
    weights = cli.initial_weights.resolve()
    metadata = cli.initial_weights_metadata.resolve()
    if not weights.is_file() or not metadata.is_file():
        raise RuntimeError('Explicit local starting weights and metadata required')
    context = data_context(cli.data_root, cli.task_config, cli.binding)
    data = Path(context['data_root']) / 'dataset.yaml'
    manifest_path = data.with_name('manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['protocol'] != config['protocol']:
        raise RuntimeError('Prepared dataset does not match protocol')
    expected_weights = json.loads(metadata.read_text(encoding='utf-8'))['sha256']
    if digest(weights) != expected_weights:
        raise RuntimeError('Starting weights changed')
    source_hashes = {p.name:digest(p) for p in (PROJECT/'src').glob('*.py')}
    if cli.preflight_only:
        if cli.resume:
            raise ValueError('Preflight does not execute/resume a training run')
        print(json.dumps({'status':'preflight-passed','execution_schema':EXECUTION_SCHEMA,
              'model_execution_performed':False,'training_performed':False,
              'dataset_context':context,'config_sha256':digest(config_path),
              'initial_weights_sha256':expected_weights,
              'initial_weights_metadata_sha256':digest(metadata),
              'source_hashes':source_hashes},indent=2),flush=True)
        return
    import psutil
    import torch
    from ultralytics import YOLO, settings
    settings.update({k:False for k in ('sync','hub','wandb','comet','clearml','mlflow','neptune','tensorboard','dvc')})
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no implicit CPU training')
    if shutil.disk_usage(PROJECT).free < 2*1024**3:
        raise RuntimeError('Less than 2 GiB disk reserve')
    record_path = run / 'run_record.json'
    current_process = psutil.Process()
    if cli.resume:
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if record.get('status') == 'completed':
            raise RuntimeError('Run already completed; do not resume or retrain it')
        old_pid = record.get('pid')
        if old_pid and psutil.pid_exists(old_pid):
            old = psutil.Process(old_pid)
            if abs(old.create_time()-record.get('process_create_time',0)) < 1:
                raise RuntimeError('Prior process still exists; do not restart a live job')
        assert_resume_inputs(record,context=context,config_sha256=digest(config_path),
                             runner_sha256=digest(Path(__file__)),initial_weights_sha256=expected_weights,
                             source_hashes=source_hashes)
        if record['manifest_sha256'] != digest(manifest_path):
            raise RuntimeError('Resume manifest bytes changed')
        if record['initial_weights_metadata_sha256'] != digest(metadata):
            raise RuntimeError('Resume starting-weight metadata changed')
        checkpoint, checkpoint_hash = checked_resume_checkpoint(run)
        record.setdefault('resumes',[]).append({'utc':datetime.now(timezone.utc).isoformat(),
                                                'checkpoint_sha256':checkpoint_hash})
        model = YOLO(str(checkpoint))
    else:
        if run.exists():
            raise RuntimeError('Run exists; preserve it rather than overwrite/restart')
        run.mkdir()
        shutil.copyfile(config_path,run/'frozen_config.json')
        source = run/'source_snapshot'
        source.mkdir()
        for path in sorted((PROJECT/'src').glob('*.py')):
            shutil.copyfile(path,source/path.name)
        packages = {d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
        write_json(run/'observed_packages.json',packages)
        record = {'run_id':run_id,'execution_schema':EXECUTION_SCHEMA,
                  'dataset_context':context,'source_hashes':source_hashes,
                  'initial_weights_metadata_sha256':digest(metadata),
                  'status':'starting','started_utc':datetime.now(timezone.utc).isoformat(),
                  'config_sha256':digest(config_path),'manifest_sha256':digest(manifest_path),
                  'checkpoint_source_sha256':expected_weights,'runner_sha256':digest(Path(__file__)),
                  'environment':{'python':platform.python_version(),'torch':torch.__version__,
                                 'ultralytics':importlib.metadata.version('ultralytics'),
                                 'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},
                  'limitations':config['limits']}
        model = YOLO(str(weights))
    record.update(status='running',pid=os.getpid(),process_create_time=current_process.create_time())
    write_json(record_path,record)
    started = time.perf_counter()

    def epoch_progress(trainer):
        values = {k:float(v) for k,v in trainer.metrics.items()}
        losses = [float(x) for x in trainer.loss_items.detach().cpu()]
        if not all(math.isfinite(x) for x in [*values.values(),*losses]):
            raise RuntimeError('Nonfinite loss/metric; stop and preserve failed run')
        progress = {'run_id':run_id,'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat(),
                    'epochs_completed':trainer.epoch+1,'epochs_planned':trainer.epochs,
                    'segment_elapsed_seconds':time.perf_counter()-started,'epoch_seconds':trainer.epoch_time,
                    'actual_amp':bool(trainer.amp),'metrics':values,'loss_items':losses,
                    'last_checkpoint':str(trainer.last),
                    'last_checkpoint_sha256':digest(trainer.last) if trainer.last.exists() else None}
        write_json(run/'progress.json',progress)
        with (run/'epoch_records.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(progress)+'\n')
        print('RUN_PROGRESS '+json.dumps({k:progress[k] for k in ('epochs_completed','epoch_seconds','actual_amp','metrics')}),flush=True)
        if shutil.disk_usage(PROJECT).free < 1024**3:
            raise RuntimeError('Disk reserve fell below 1 GiB; stop after saved epoch')

    model.add_callback('on_fit_epoch_end',epoch_progress)
    # Framework AMP capability check uses this checkpoint basename; keep it local/offline.
    os.chdir(weights.parent)
    try:
        if cli.resume:
            metrics = model.train(resume=True)
        else:
            args = dict(config['training'],data=str(data),project=str(run),name='fit',exist_ok=False)
            record['training_arguments'] = args
            write_json(record_path,record)
            metrics = model.train(**args)
        torch.cuda.synchronize()
        record.update(status='completed',segment_seconds=time.perf_counter()-started,
                      metrics={k:float(v) for k,v in metrics.results_dict.items()},
                      final_class_names=model.names,
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                      output_hashes={str(x.relative_to(run)):digest(x) for x in (run/'fit').rglob('*')
                                     if x.is_file() and x.suffix in ('.pt','.csv','.yaml')})
    except BaseException as exc:
        record.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                      error=repr(exc),segment_seconds=time.perf_counter()-started)
        raise
    finally:
        record['segment_finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(record_path,record)
    print('RUN_COMPLETE '+json.dumps({'run_id':run_id,'metrics':record['metrics']}),flush=True)


if __name__=='__main__':
    main()
