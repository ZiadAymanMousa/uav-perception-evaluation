# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Explicit data bindings and honest historical-checkpoint smoke certificates."""
import argparse
import csv
import math
from datetime import datetime, timezone
import json
from pathlib import Path

from audit_dataset import digest
from portable_preparer import (verify_prepared_dataset, load_reviewed_config,
                               validate_audit, require, read_json, write_json)
from run_records import canonical_output_hashes
from semantic_dataset import identity

EXECUTION_SCHEMA = 'uvd1-portable-execution-v1'
CERTIFICATE_SCHEMA = 'uvd1-historical-checkpoint-smoke-equivalence-v1'


def safe_id(value):
    require(isinstance(value, str) and Path(value).name == value and
            value not in ('.', '..') and not any(c in value for c in ('/', '\\', ':')), 'Unsafe run id')
    return value


def data_context(data_root, task_config, binding_path):
    root, config, binding = (Path(p).resolve() for p in (data_root, task_config, binding_path))
    verified = verify_prepared_dataset(root, config, expected_binding=binding)
    return {'data_root': str(root), 'task_config_path': str(config), 'binding_path': str(binding),
            'binding_sha256': digest(binding), 'verified': verified}


def assert_same_launch_data(recorded, actual):
    """Exact byte continuity for a launched run; semantics alone never resume it."""
    for key in ('binding_sha256',):
        require(recorded[key] == actual[key], 'Launch data binding changed: ' + key)
    for key in ('config_sha256', 'dataset_manifest_sha256', 'source_audit_sha256',
                'dataset_yaml_sha256', 'semantic_identity_sha256', 'implementation_sha256'):
        require(recorded['verified'][key] == actual['verified'][key], 'Launch data bytes changed: ' + key)


def assert_resume_inputs(record, *, context, config_sha256, runner_sha256, initial_weights_sha256,
                         source_hashes):
    require(record['status'] != 'completed', 'Completed run cannot resume')
    require(record.get('execution_schema') == EXECUTION_SCHEMA, 'Legacy/unknown run cannot resume here')
    assert_same_launch_data(record['dataset_context'], context)
    for key, actual in [('config_sha256', config_sha256), ('runner_sha256', runner_sha256),
                        ('checkpoint_source_sha256', initial_weights_sha256), ('source_hashes', source_hashes)]:
        require(record[key] == actual, 'Resume inputs changed: ' + key)


def checked_resume_checkpoint(run_dir):
    """Bind the restart state to the checksum saved by the last epoch callback."""
    run = Path(run_dir)
    checkpoint = run / 'fit/weights/last.pt'
    require(checkpoint.is_file(), 'No checkpoint to resume')
    progress = read_json(run / 'progress.json')
    expected = progress.get('last_checkpoint_sha256')
    require(isinstance(expected, str) and len(expected) == 64,
            'No retained last-checkpoint checksum; review interrupted state before resume')
    actual = digest(checkpoint)
    require(actual == expected, 'Resume checkpoint bytes changed from retained progress')
    return checkpoint, actual


def review_training_history(run_dir):
    """Cheap CPU checks shared by inference gate and full checkpoint reviewer."""
    run = Path(run_dir)
    config, progress = read_json(run/'frozen_config.json'), read_json(run/'progress.json')
    planned = config['training']['epochs']
    require(isinstance(planned,int) and planned > 0, 'Invalid planned epoch count')
    require(progress['epochs_planned'] == planned and progress['epochs_completed'] == planned,
            'Training progress/config epochs differ')
    with (run/'fit/results.csv').open(encoding='utf-8',newline='') as stream:
        rows = [{k.strip():float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    require([r['epoch'] for r in rows] == list(range(1,planned+1)), 'Training CSV epochs incomplete/noncontiguous')
    require(all(math.isfinite(v) for row in rows for v in row.values()), 'Nonfinite training CSV')
    events = [json.loads(line) for line in (run/'epoch_records.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    require([e['epochs_completed'] for e in events[:planned]] == list(range(1,planned+1)),
            'Epoch callback history incomplete/noncontiguous')
    require(len(events) == planned+1 and events[-1]['epochs_completed'] == planned,
            'Expected one preserved posttraining validation event')
    require(all(math.isfinite(v) for e in events for v in [*e['metrics'].values(),*e['loss_items']]),
            'Nonfinite epoch callback history')
    return {'planned_epochs':planned,'csv_epochs':len(rows),'callback_events':len(events)}


def completed_checkpoint(run_dir, weights, *, context=None, require_portable=True):
    run, weights = Path(run_dir).resolve(), Path(weights).resolve()
    record, progress = read_json(run / 'run_record.json'), read_json(run / 'progress.json')
    require(record['status'] == 'completed' and
            progress['epochs_completed'] == progress['epochs_planned'], 'Baseline training is incomplete')
    require(digest(weights) == canonical_output_hashes(record['output_hashes'])['fit/weights/best.pt'],
            'Checkpoint is not the selected completed-baseline output')
    require(digest(run / 'frozen_config.json') == record['config_sha256'], 'Frozen baseline config changed')
    require(digest(run / 'source_snapshot/train_full_baseline.py') == record['runner_sha256'],
            'Baseline runner snapshot changed')
    if require_portable:
        require(record.get('execution_schema') == EXECUTION_SCHEMA, 'Development study requires a prospective portable baseline')
        require(context is not None, 'Explicit dataset context required')
        assert_same_launch_data(record['dataset_context'], context)
        require(record['manifest_sha256'] == context['verified']['dataset_manifest_sha256'], 'Baseline manifest bytes differ')
        review_training_history(run)
        for name, expected in record['source_hashes'].items():
            require(digest(run / 'source_snapshot' / name) == expected, 'Baseline helper snapshot changed')
    return record


def historical_certificate(*, baseline_run, weights, legacy_data_root, legacy_audit, context):
    """Read legacy evidence; do not modify or relabel its training provenance."""
    run = Path(baseline_run).resolve()
    weights = Path(weights).resolve()
    root = Path(legacy_data_root).resolve()
    audit_path = Path(legacy_audit).resolve()
    record = completed_checkpoint(run, weights, require_portable=False)
    require(record.get('execution_schema') != EXECUTION_SCHEMA, 'Use prospective baseline route for new records')
    manifest = read_json(root / 'manifest.json')
    require(digest(root / 'manifest.json') == record['manifest_sha256'], 'Historical training manifest changed')
    require(digest(audit_path) == manifest['audit_file_sha256'], 'Historical source audit changed')
    task = load_reviewed_config(context['task_config_path'])
    validate_audit(read_json(audit_path), task)
    old_signature = identity(root, image_mode='rehash')
    expected = context['verified']['semantic_identity_sha256']
    require(old_signature['identity_sha256'] == expected == task['expected_semantic_identity_sha256'],
            'Historical and current semantic tasks differ')
    require(digest(root / 'manifest.json') != context['verified']['dataset_manifest_sha256'],
            'Certificate intended for an actually different fresh preparation')
    return {'schema': CERTIFICATE_SCHEMA, 'purpose': 'pipeline-smoke only',
            'historical_training_on_fresh_preparation': False,
            'historical_baseline_run': str(run), 'weights_path': str(weights),
            'historical_data_root': str(root), 'historical_audit_path': str(audit_path),
            'weights_sha256': digest(weights), 'historical_record_sha256': digest(run / 'run_record.json'),
            'historical_progress_sha256': digest(run / 'progress.json'),
            'historical_manifest_sha256': digest(root / 'manifest.json'),
            'historical_audit_sha256': digest(audit_path),
            'historical_semantic_identity_sha256': old_signature['identity_sha256'],
            'evaluation_semantic_identity_sha256': expected,
            'evaluation_manifest_sha256': context['verified']['dataset_manifest_sha256'],
            'evaluation_binding_sha256': context['binding_sha256'],
            'legacy_images_labels_sidecars_fully_rehashed': True,
            'checker_sha256': digest(Path(__file__)),
            'limits': ['Historical checkpoint remains trained on the original preparation',
                       'Certificate permits CPU pipeline-smoke only; no new model or training result']}


def check_historical_certificate(path, weights, context):
    expected = read_json(path)
    require(expected.get('schema') == CERTIFICATE_SCHEMA and
            expected.get('purpose') == 'pipeline-smoke only' and
            expected.get('historical_training_on_fresh_preparation') is False,
            'Invalid historical smoke certificate')
    actual = historical_certificate(baseline_run=expected['historical_baseline_run'], weights=weights,
        legacy_data_root=expected['historical_data_root'], legacy_audit=expected['historical_audit_path'], context=context)
    require(expected == actual, 'Historical certificate is stale/forged or inputs changed')
    return {'certificate_sha256': digest(Path(path)), 'certificate': actual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--task-config', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--baseline-run-dir', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--legacy-data-root', type=Path, required=True)
    parser.add_argument('--legacy-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing certificate')
    context = data_context(args.data_root, args.task_config, args.binding)
    certificate = historical_certificate(baseline_run=args.baseline_run_dir, weights=args.weights,
        legacy_data_root=args.legacy_data_root, legacy_audit=args.legacy_audit, context=context)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, certificate)
    print(json.dumps({'status': 'historical-equivalence-certificate-created',
                      'purpose': 'pipeline-smoke only', 'certificate_sha256': digest(args.output)}, indent=2))


if __name__ == '__main__':
    main()
