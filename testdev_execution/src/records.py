# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Validate a complete image/method matrix before scoring retained predictions."""
import json
import math
from pathlib import Path
import numpy as np
from common import (require, digest, read_json, object_sha, verify_sources,
                    validate_versions, prediction_arguments, parameter_device)
from inference_geometry import Window, fixed_windows, prediction_rows
from score_converted_predictions import label_rows_as_xyxy, score_records

def methods_for_index(task, index):
    methods = task['inference']['methods']
    return methods[index % len(methods):] + methods[:index % len(methods)]

def windows_for(entry, method):
    windows = [Window(0, 0, entry['width'], entry['height'])]
    if method['kind'] == 'fixed':
        windows += fixed_windows(entry['width'], entry['height'], method['tile_size'], method['overlap'])
    return windows

def nonnegative_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0

def validate_row(row, entry, method, task):
    require(row['image'] == entry['image'] and row['method'] == method['name'], 'Missing, duplicate or out-of-order image/method row')
    require(row['image_sha256'] == entry['image_sha256'], 'Prediction image content identity differs')
    require(isinstance(row['predictions'],list) and all(isinstance(r,list) and len(r)==6 for r in row['predictions']), 'Prediction rows must be an empty list or Nx6 list')
    pred = prediction_rows(row['predictions'])
    require(len(pred) <= task['inference']['max_detections'], 'Prediction cap exceeded')
    require(not len(pred) or np.all(pred[:, 2:4] > pred[:, :2]), 'Zero-area prediction')
    require(not len(pred) or (np.all(pred[:, :4] >= 0) and np.all(pred[:, [0,2]] <= entry['width'])
                             and np.all(pred[:, [1,3]] <= entry['height'])), 'Prediction outside image')
    require(len(pred) < 2 or np.all(np.diff(pred[:,4]) <= 0), 'Predictions not globally confidence sorted')
    timing, windows = row['timing'], windows_for(entry, method)
    require(timing['image'] == entry['image'] and timing['method'] == method['name'], 'Timing image/method differs')
    for name in ('decode_ms', 'views_ms', 'merge_ms', 'end_to_end_ms'):
        require(isinstance(timing[name], (int,float)) and math.isfinite(timing[name]) and timing[name] >= 0, 'Invalid timing')
    require(abs(timing['end_to_end_ms'] - sum(timing[k] for k in ('decode_ms','views_ms','merge_ms'))) <= 1e-5, 'Timing boundary accounting mismatch')
    require(timing['detector_passes'] == len(windows) and timing['processed_input_pixels'] == len(windows)*method['input_size']**2,
            'Detector pass/input-pixel accounting differs')
    require(timing['prediction_count'] == len(pred), 'Prediction count differs')
    views = timing['per_view_counts']
    require(len(views) == len(windows), 'View accounting missing/extra')
    before = after = 0
    for value, window in zip(views, windows):
        require(value['window_xyxy'] == [window.x1,window.y1,window.x2,window.y2], 'View geometry differs')
        a, b = value['output_before_geometry'], value['output_after_geometry']
        require(nonnegative_int(a) and nonnegative_int(b) and b <= a <= task['inference']['max_detections'], 'Invalid view counts or per-view prediction cap exceeded')
        before += a
        after += b
    geometry, merge = timing['geometry'], timing['merge_counts']
    require(nonnegative_int(geometry['clipped']) and geometry['clipped'] <= before and
            geometry['zero_area_removed'] == before-after, 'Geometry removal accounting differs')
    require(all(nonnegative_int(merge[k]) for k in ('before_merge','after_nms_before_cap','after_cap','removed_nms','removed_cap')), 'Invalid merge counts')
    require(merge['before_merge'] == after and merge['after_nms_before_cap'] <= after and
            merge['after_cap'] == len(pred) == min(merge['after_nms_before_cap'], task['inference']['max_detections']) and
            merge['removed_nms'] == after-merge['after_nms_before_cap'] and
            merge['removed_cap'] == merge['after_nms_before_cap']-len(pred), 'Merge count accounting differs')
    require(method['kind'] != 'full' or merge['removed_nms'] == 0, 'Unexpected second NMS on full image')
    return pred

def score_log(path, prepared, task, *, smoke=False, scorer=score_records):
    manifest = read_json(Path(prepared)/'manifest.json')
    entries = manifest['entries'][:1] if smoke else manifest['entries']
    groups = {m['name']: [] for m in task['inference']['methods']}
    timings = {m['name']: [] for m in task['inference']['methods']}
    passes = {m['name']: 0 for m in task['inference']['methods']}
    pixels = dict(passes)
    row_count = 0
    with Path(path).open(encoding='utf-8') as stream:
        for index, entry in enumerate(entries):
            gt = label_rows_as_xyxy(Path(prepared)/'labels/testdev'/(Path(entry['image']).stem+'.txt'), entry['width'], entry['height'])
            for method in methods_for_index(task, index):
                line = stream.readline()
                require(bool(line.strip()), 'Missing or blank prediction row')
                row = json.loads(line)
                predictions = validate_row(row, entry, method, task)
                name = method['name']
                groups[name].append({'gt': gt, 'predictions': predictions})
                timings[name].append(row['timing']['end_to_end_ms'])
                passes[name] += row['timing']['detector_passes']
                pixels[name] += row['timing']['processed_input_pixels']
                row_count += 1
        require(stream.read() == '', 'Extra or duplicate prediction rows')
    expected_rows = 4 if smoke else task['complete_accounting']['records']
    require(row_count == expected_rows, 'Incomplete prediction matrix')
    if not smoke:
        require(passes == task['complete_accounting']['detector_passes'], 'Full sweep detector-pass totals differ')
    result = {}
    for method in task['inference']['methods']:
        name = method['name']
        result[name] = {'scoring': scorer(groups[name], dict(enumerate(task['class_names']))),
                        'diagnostic_median_ms': float(np.median(timings[name])),
                        'diagnostic_p95_ms': float(np.percentile(timings[name],95)),
                        'detector_passes': passes[name], 'processed_input_pixels': pixels[name]}
    return result, {'records': row_count, 'images': len(entries), 'detector_passes': passes,
                    'prediction_log_sha256': digest(Path(path)), 'scope': 'one-image-cpu-smoke' if smoke else 'complete-testdev'}

def verify_run_files(run, task, data_context, checkpoint_context, *, prepared):
    run = Path(run)
    record = read_json(run/'run_record.json')
    require(record['schema'] == 'uvd1-portable-testdev-run-v1', 'Unknown run schema')
    require(record['status'] == 'completed', 'Run is not completed')
    smoke = record['scope'] == 'one-image-cpu-smoke'
    require(record['scope'] in ('one-image-cpu-smoke','complete-testdev'), 'Unknown run scope')
    require(record['data_context'] == data_context, 'Prepared/raw launch context changed')
    require(record['checkpoint_context'] == checkpoint_context, 'Checkpoint/training context changed')
    require(record['config_sha256'] == data_context['config_sha256'], 'Run config changed')
    require(digest(run/'task_config.json') == record['config_sha256'], 'Run config snapshot changed')
    verify_sources(run/'source_snapshot', record['source_hashes'])
    require(digest(run/'predictions_and_timings.jsonl') == record['prediction_log_sha256'], 'Predictions changed')
    require(digest(run/'summary.json') == record['summary_sha256'], 'Saved summary changed')
    manifest = read_json(run/'selection.json')
    require(digest(run/'selection.json') == record['selection_sha256'], 'Selection snapshot changed')
    # Use the caller-verified preparation; retained launch paths are descriptive only.
    expected = read_json(Path(prepared)/'manifest.json')['entries']
    selected = [e['image'] for e in (expected[:1] if smoke else expected)]
    require(manifest == selected == record['selection'], 'Run selection differs')
    count = 1 if smoke else task['expected_counts']['images']
    progress = read_json(run/'progress.json')
    require(progress['status'] == 'completed' and progress['images_completed'] == progress['images_planned'] == count and
            progress['records_written'] == 4*count and record['images_compared'] == count,
            'False complete state or incomplete progress')
    require(record['accounting']['images'] == count and record['accounting']['records'] == 4*count, 'Run accounting differs')
    require(record['warmup_calls'] == task['complete_accounting']['warmup_calls'], 'Warm-up accounting differs')
    require(not smoke or record['device_requested'] == 'cpu', 'Smoke is not CPU')
    require(record['actual_parameter_device'] == parameter_device(record['device_requested']), 'Actual parameter device differs from requested device')
    require(record['predict_arguments'] == prediction_arguments(task,record['device_requested']), 'Recorded prediction arguments differ from fixed task policy')
    exact = record['checkpoint_context']['mode'] == 'frozen-replay' and not smoke
    validate_versions(record['environment']['versions'],task,exact_inference=exact)
    require(record['environment']['framework_source_hashes'] == task['framework_source_hashes'], 'Recorded framework hashes differ from task')
    require(isinstance(record['environment']['torch_num_threads'],int) and not isinstance(record['environment']['torch_num_threads'],bool) and record['environment']['torch_num_threads'] > 0, 'Invalid recorded inference thread count')
    return record, smoke
