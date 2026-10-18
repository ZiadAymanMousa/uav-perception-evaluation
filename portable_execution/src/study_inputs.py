# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Strict prospective saved-output loaders with explicit data/config roots."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from audit_dataset import digest
from execution_inputs import (EXECUTION_SCHEMA, data_context, assert_same_launch_data, safe_id)
from inference_geometry import prediction_rows
from portable_preparer import read_json, require
from score_converted_predictions import label_rows_as_xyxy


def native_detections(value):
    """xyxy/conf/class0..9 -> xywh/conf/class1..10/-1/-1; original coordinate bridge."""
    rows = prediction_rows(value).astype(np.float64)
    require(not np.any(rows[:, 2:4] <= rows[:, :2]), 'Native bridge requires positive-area predictions')
    require(len(rows) < 2 or not np.any(np.diff(rows[:, 4]) > 0), 'Predictions must be score-sorted')
    output = np.full((len(rows), 8), -1., dtype=np.float64)
    output[:, :2] = rows[:, :2]
    output[:, 2:4] = rows[:, 2:4] - rows[:, :2]
    output[:, 4] = rows[:, 4]
    output[:, 5] = rows[:, 5] + 1
    return output


def saved_rows(run_dir, *, data_root, task_config, binding):
    run = Path(run_dir).resolve()
    record = read_json(run / 'run_record.json')
    require(record['status'] == 'completed', 'Prediction run is incomplete')
    require(record.get('execution_schema') == EXECUTION_SCHEMA, 'Unsupported saved prediction schema')
    context = data_context(data_root, task_config, binding)
    assert_same_launch_data(record['dataset_context'], context)
    config_name = safe_id(record['config_snapshot_filename'])
    config_path = run / 'source_snapshot' / config_name
    require(digest(config_path) == record['config_sha256'], 'Inference config snapshot changed')
    for name, expected in record['source_hashes'].items():
        require(digest(run / 'source_snapshot' / safe_id(name)) == expected, 'Inference helper snapshot changed')
    config = read_json(config_path)
    methods = [m['name'] for m in config['methods']]
    require(len(methods) == len(set(methods)), 'Duplicate method identifiers')
    selected = record['selection']
    require(selected and len(selected) == len(set(selected)), 'Invalid image selection')
    manifest = read_json(Path(context['data_root']) / 'manifest.json')
    entries = {e['image']: e for e in manifest['splits']['val']['entries']}
    require(all(safe_id(name) in entries for name in selected), 'Unknown selected image')
    log = run / 'predictions_and_timings.jsonl'
    require(digest(log) == record['prediction_log_sha256'], 'Saved prediction log changed')
    rows = {}
    for line in log.read_text(encoding='utf-8').splitlines():
        item = json.loads(line)
        key = (item['image'], item['method'])
        require(key not in rows and key[0] in selected and key[1] in methods,
                'Unexpected/duplicate prediction row')
        require(item['image_sha256'] == entries[key[0]]['image_sha256'], 'Prediction image differs')
        prediction_rows(item['predictions'])
        rows[key] = item
    require(len(rows) == len(selected) * len(methods), 'Missing image/method output')
    return record, context, methods, selected, entries, rows


def load_converted_cases(run_dir, *, data_root, task_config, binding):
    record, context, methods, selected, entries, rows = saved_rows(
        run_dir, data_root=data_root, task_config=task_config, binding=binding)
    root = Path(context['data_root'])
    output = {method: [] for method in methods}
    for name in selected:
        with Image.open(root / 'images/val' / name) as value:
            width, height = value.size
        gt = label_rows_as_xyxy(root / 'labels/val' / (Path(name).stem + '.txt'), width, height)
        for method in methods:
            output[method].append({'gt': gt, 'predictions': rows[(name, method)]['predictions']})
    return output, {'purpose': record['purpose'], 'run_record_sha256': digest(Path(run_dir) / 'run_record.json'),
                    'prediction_log_sha256': record['prediction_log_sha256'],
                    'dataset_context': context, 'checkpoint_provenance': record['checkpoint_provenance']}


def load_cases(run_dir, *, data_root, task_config, binding, raw_val_root):
    """Raw/native coordinate cases; no native evaluator is bundled or invoked."""
    record, context, methods, selected, entries, rows = saved_rows(
        run_dir, data_root=data_root, task_config=task_config, binding=binding)
    root, source = Path(context['data_root']), Path(raw_val_root).resolve()
    audit = read_json(root / 'source_audit.json')
    originals = {Path(e['image']).name: e for e in audit['splits']['val']['manifest']}
    output = {method: [] for method in methods}
    for name in selected:
        original = originals[name]
        image = source / 'images' / name
        annotation = source / 'annotations' / (Path(name).stem + '.txt')
        require(digest(image) == original['image_sha256'] == entries[name]['image_sha256'] and
                digest(annotation) == original['annotation_sha256'], 'Original validation inputs changed')
        gt = np.loadtxt(annotation, delimiter=',', dtype=np.float64, ndmin=2)
        for method in methods:
            output[method].append({'name': Path(name).stem, 'width': original['width'],
                                  'height': original['height'], 'gt': gt.tolist(),
                                  'detections': native_detections(rows[(name, method)]['predictions']).tolist()})
    return output, {'purpose': record['purpose'], 'run_record_sha256': digest(Path(run_dir) / 'run_record.json'),
                    'prediction_log_sha256': record['prediction_log_sha256'], 'dataset_context': context}
