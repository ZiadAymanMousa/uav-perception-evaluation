# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Read-only integrity/curve summary after the frozen baseline has completed.

Writes a separate review directory and inspects the verified local weights on CPU.
Does not retrain or modify the checkpoint.
Descriptive trends cannot certify convergence or detector adequacy.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from run_records import canonical_output_hashes

PROJECT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(PROJECT/'artifacts/matplotlib_config')
os.environ['YOLO_CONFIG_DIR'] = str(PROJECT/'artifacts/ultralytics_config')
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['YOLO_AUTOINSTALL'] = 'false'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):
            value.update(part)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='b001_uvd1_yolo11n640_seed0')
    args = parser.parse_args()
    if Path(args.run).name != args.run or ':' in args.run or '\\' in args.run:
        raise ValueError('Unsafe run id')
    run = PROJECT/'artifacts'/args.run
    record = json.loads((run/'run_record.json').read_text(encoding='utf-8'))
    progress = json.loads((run/'progress.json').read_text(encoding='utf-8'))
    if record['status'] != 'completed' or progress['epochs_completed'] != progress['epochs_planned']:
        raise RuntimeError('Training is not completed; no final review generated')
    config = json.loads((run/'frozen_config.json').read_text(encoding='utf-8'))
    checks = {'frozen_config_matches_launch':digest(run/'frozen_config.json') == record['config_sha256'],
              'manifest_matches_launch':digest(PROJECT/'data/yolo_full_uvd1/manifest.json') == record['manifest_sha256'],
              'runner_snapshot_matches_launch':digest(run/'source_snapshot/train_full_baseline.py') == record['runner_sha256']}
    output_hashes = canonical_output_hashes(record['output_hashes'])
    for path,expected in output_hashes.items():
        checks['output:'+path] = digest(run/path) == expected
    with (run/'fit/results.csv').open(encoding='utf-8',newline='') as stream:
        rows = [{key.strip():float(value) for key,value in row.items()} for row in csv.DictReader(stream)]
    planned = config['training']['epochs']
    checks['csv_epochs_complete_unique_contiguous'] = [r['epoch'] for r in rows] == list(range(1,planned+1))
    checks['csv_all_values_finite'] = all(math.isfinite(v) for row in rows for v in row.values())
    epoch_lines = [json.loads(line) for line in (run/'epoch_records.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    # This pinned trainer calls on_fit_epoch_end again after best-checkpoint
    # final_eval. Preserve that final-validation event, rather than count it as
    # another training epoch or demand an incorrect exactly-150 callback count.
    checks['epoch_records_complete_contiguous'] = [r['epochs_completed'] for r in epoch_lines[:planned]] == list(range(1,planned+1))
    extra_events = epoch_lines[planned:]
    checks['one_expected_posttraining_validation_event'] = len(extra_events) == 1 and extra_events[0]['epochs_completed'] == planned
    checks['epoch_records_finite'] = all(math.isfinite(v) for row in epoch_lines for v in [*row['metrics'].values(),*row['loss_items']])
    if not all(checks.values()):
        raise RuntimeError(f'Integrity review failed: {[key for key,value in checks.items() if not value]}')
    import torch
    import ultralytics
    from ultralytics import YOLO
    from prepare_training_pilot import NAMES
    torch.set_num_threads(2)
    model = YOLO(str(run/'fit/weights/best.pt'))
    checks['framework_version_matches_launch'] = ultralytics.__version__ == record['environment']['ultralytics']
    checks['checkpoint_names_match_task'] = dict(model.names) == dict(enumerate(NAMES))
    checks['checkpoint_parameters_on_cpu'] = all(p.device.type == 'cpu' for p in model.model.parameters())
    checks['checkpoint_parameters_finite'] = all(bool(torch.isfinite(p).all()) for p in model.model.parameters())
    parameter_count = sum(p.numel() for p in model.model.parameters())
    metric = 'metrics/mAP50-95(B)'
    # The pinned Metric.fitness uses AP50-95 only. Equal fitness saves the
    # latest checkpoint, so resolve recorded callback ties by epoch. The pinned
    # training validator has already rounded these metrics to five decimals.
    selected_event = max(epoch_lines[:planned],key=lambda r:(r['metrics'][metric],r['epochs_completed']))
    selected_train_metrics = model.ckpt.get('train_metrics',{})
    checks['checkpoint_training_metric_matches_max_fitness'] = selected_train_metrics.get(metric) == selected_event['metrics'][metric]
    if not all(checks.values()):
        raise RuntimeError(f'Checkpoint review failed: {[key for key,value in checks.items() if not value]}')
    best = max(rows,key=lambda r:r[metric])
    tail = rows[-15:]
    prior = rows[-30:-15]
    summary = {'utc':datetime.now(timezone.utc).isoformat(),'run':args.run,
               'integrity_checks':checks,'epochs':len(rows),'callback_events':len(epoch_lines),
               'posttraining_validation_events':len(extra_events),'seed':config['training']['seed'],
               'checkpoint_parameter_count':parameter_count,'checkpoint_names':dict(model.names),
               'selected_epoch_from_recorded_callback_fitness':selected_event['epochs_completed'],
               'fitness_precision_note':'Pinned training validator rounds callback metrics to five decimals; this is not unrounded detector AP',
               'checkpoint_training_metrics':selected_train_metrics,
               'framework_source_hashes':{name:digest(Path(ultralytics.__file__).parent/path) for name,path in {
                   'trainer':'engine/trainer.py','fitness':'utils/metrics.py'}.items()},
               'best_csv_epoch':int(best['epoch']),'best_csv_metrics':{key:value for key,value in best.items() if key.startswith('metrics/')},
               'last_csv_metrics':{key:value for key,value in rows[-1].items() if key.startswith('metrics/')},
               'completed_selected_checkpoint_validation':record['metrics'],
               'last15_mean_ap50_95_percent':100*statistics.mean(r[metric] for r in tail),
               'preceding15_mean_ap50_95_percent':100*statistics.mean(r[metric] for r in prior),
               'last15_minus_preceding15_pp':100*(statistics.mean(r[metric] for r in tail)-statistics.mean(r[metric] for r in prior)),
               'last15_range_ap50_95_percent':[100*min(r[metric] for r in tail),100*max(r[metric] for r in tail)],
               'best_weights_sha256':output_hashes['fit/weights/best.pt'],
               'inputs':{'run_record_sha256':digest(run/'run_record.json'),'csv_sha256':digest(run/'fit/results.csv'),
                         'progress_sha256':digest(run/'progress.json'),'epoch_records_sha256':digest(run/'epoch_records.jsonl'),
                         'reviewer_sha256':digest(Path(__file__)),
                         'record_path_helper_sha256':digest(PROJECT/'src/run_records.py')},
               'limits':['Single seed; descriptive trends do not prove convergence',
                         'Development checkpoint selection; no independent final evaluation',
                         'Converted training-validator AP; not native VisDrone scoring',
                         'Training validator is multi-label; inference study is single-label',
                         'Review does not certify optimal hyperparameters or adequate field performance']}
    dest = PROJECT/'artifacts'/(args.run+'_completion_review_20261016')
    if dest.exists():
        raise RuntimeError('Preserve existing review instead of overwriting')
    dest.mkdir()
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    (dest/'review_source.py').write_bytes(Path(__file__).read_bytes())
    (dest/'run_records.py').write_bytes((PROJECT/'src/run_records.py').read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    epochs = [r['epoch'] for r in rows]
    for key,label in [('metrics/mAP50(B)','AP50'),(metric,'AP50-95')]:
        axes[0,0].plot(epochs,[100*r[key] for r in rows],label=label)
    axes[0,0].set(ylabel='Converted validation AP (%)',title='Development scores; multi-label validator')
    for axis,loss in [(axes[0,1],'box_loss'),(axes[1,0],'cls_loss')]:
        for group in ['train','val']:
            axis.plot(epochs,[r[group+'/'+loss] for r in rows],label=group)
        axis.set(ylabel=loss,title='Training and validation loss')
    axes[1,1].plot(epochs,[r['lr/pg1'] for r in rows],label='parameter group 1')
    axes[1,1].set(ylabel='Learning rate',title='Recorded cosine schedule')
    for axis in axes.flat:
        axis.set_xlabel('Epoch')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle(args.run+' | one seed, exposed development data',fontsize=12)
    fig.savefig(dest/'learning_curves.png',dpi=170)
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
