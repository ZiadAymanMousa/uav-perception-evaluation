# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Explain observed saved-output cap sensitivity without inference or training."""
import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_dataset import digest
from inference_geometry import prediction_rows
from prepare_training_pilot import NAMES
from score_converted_predictions import label_rows_as_xyxy

PROJECT = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(PROJECT/'artifacts/ultralytics_config')
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['MPLCONFIGDIR'] = str(PROJECT/'artifacts/matplotlib_config')


def safe_run_id(value):
    if not value or Path(value).name != value or ':' in value or '\\' in value or value in ('.', '..'):
        raise ValueError('Unsafe run identifier')
    return value


def match_pair(gt, predictions, cap):
    """Run the unmodified pinned CPU matcher on all and retained detections."""
    import torch
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import box_iou
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, 5)
    predictions = prediction_rows(predictions)
    if not isinstance(cap, int) or cap < 1:
        raise ValueError('Expected positive integer cap')
    if not np.isfinite(gt).all() or np.any(gt[:, 3:5] <= gt[:, 1:3]):
        raise ValueError('Invalid target geometry')
    if np.any(gt[:, 0] != np.floor(gt[:, 0])) or np.any((gt[:, 0] < 0) | (gt[:, 0] > 9)):
        raise ValueError('Invalid target class')
    if len(predictions) > 1 and np.any(np.diff(predictions[:, 4]) > 0):
        raise ValueError('Expected score-sorted predictions')
    full = np.zeros((len(predictions), 10), dtype=bool)
    retained = np.zeros((min(cap, len(predictions)), 10), dtype=bool)
    if len(gt) and len(predictions):
        matcher = SimpleNamespace(iouv=torch.linspace(.5, .95, 10))
        overlap = box_iou(torch.from_numpy(gt[:, 1:5]), torch.from_numpy(predictions[:, :4]))
        pred_classes = torch.from_numpy(predictions[:, 5])
        true_classes = torch.from_numpy(gt[:, 0])
        full = BaseValidator.match_predictions(matcher, pred_classes, true_classes, overlap).numpy()
        retained = BaseValidator.match_predictions(matcher, pred_classes[:cap], true_classes, overlap[:, :cap]).numpy()
    return full, retained


def new_metrics():
    from ultralytics.utils.metrics import DetMetrics
    return DetMetrics(names=dict(enumerate(NAMES)))


def update(metrics, gt, pred, tp):
    metrics.update_stats({'tp': tp, 'conf': pred[:, 4], 'pred_cls': pred[:, 5],
                          'target_cls': gt[:, 0], 'target_img': np.unique(gt[:, 0])})


def finish(metrics):
    from ultralytics.utils.metrics import compute_ap
    stats = metrics.process(plot=False)
    order = np.argsort(-stats['conf'])
    tp, classes = stats['tp'][order], stats['pred_cls'][order]
    targets, numbers = np.unique(stats['target_cls'], return_counts=True)
    grid = np.linspace(0, 1, 101)
    observed, beyond, attained = [], [], []
    for c, number in zip(targets, numbers):
        correct = tp[classes == c]
        if not len(correct):
            observed.append(np.zeros(10)); beyond.append(np.zeros(10)); attained.append(np.zeros(10))
            continue
        tpc = correct.cumsum(axis=0)
        fpc = (1-correct).cumsum(axis=0)
        recall = tpc/(number+1e-16)
        precision = tpc/(tpc+fpc)
        observed_class, beyond_class = [], []
        for j in range(10):
            ap, mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            values = np.interp(grid, mrec, mpre)
            rmax = float(recall[-1, j])
            # Partition the exact 101-point trapezoid polygon at attained recall.
            # This is an accounting diagnostic, not a replacement AP convention.
            left = np.concatenate((grid[grid < rmax], [rmax]))
            right = np.concatenate(([rmax], grid[grid > rmax]))
            area_left = float(np.trapezoid(np.interp(left, grid, values), left))
            area_right = float(np.trapezoid(np.interp(right, grid, values), right))
            if not np.isclose(area_left+area_right, ap, atol=1e-12, rtol=0):
                raise RuntimeError('AP polygon partition failed')
            observed_class.append(area_left); beyond_class.append(area_right)
        observed.append(observed_class); beyond.append(beyond_class); attained.append(recall[-1])
    observed, beyond, attained = np.asarray(observed), np.asarray(beyond), np.asarray(attained)
    if not np.allclose(observed+beyond, metrics.box.all_ap, atol=1e-12, rtol=0):
        raise RuntimeError('AP accounting differs from pinned metrics')
    return {'ap50_percent': 100*float(metrics.box.map50),
            'ap50_95_percent': 100*float(metrics.box.map),
            'ap_grid_area_at_or_below_attained_recall_percent': 100*float(observed.mean()),
            'ap_grid_area_beyond_attained_recall_percent': 100*float(beyond.mean()),
            'class_macro_attained_recall_by_iou_percent': (100*attained.mean(axis=0)).tolist(),
            'matched_tp_by_iou': tp.sum(axis=0).tolist(),
            'per_class_ap50_95_percent': {NAMES[int(c)]: 100*float(v) for c, v in
                                          zip(metrics.ap_class_index, metrics.box.ap)}}


def constructed_controls():
    import torch
    from ultralytics.utils.metrics import box_iou
    gt = np.asarray([[0, 10, 10, 20, 20]], dtype=np.float32)
    pred = np.asarray([[7.6, 10, 17.6, 20, .9, 0],
                       [12.2, 10, 22.2, 20, .1, 0]], dtype=np.float32)
    full, retained = match_pair(gt, pred, 1)
    if not full[0, 0] or full[1, 0] or not np.array_equal(full[:1], retained):
        raise RuntimeError('Revised pinned-matcher control differs from inspected ordering')
    between = float(box_iou(torch.from_numpy(pred[:1, :4]), torch.from_numpy(pred[1:, :4]))[0, 0])
    if between >= .7:
        raise RuntimeError('Constructed pair would cross the study NMS IoU threshold')
    results = {}
    for name, p, t in [('all2', pred, full), ('top1_fixed_matches', pred[:1], full[:1]),
                       ('top1_rematched', pred[:1], retained)]:
        metric = new_metrics()
        update(metric, gt, p, t)
        results[name] = finish(metric)
    opposed = {'gt': gt.tolist(), 'predictions': pred.tolist(),
            'prediction_pair_iou': between, 'full_tp': full.tolist(),
            'retained_rematched_tp': retained.tolist(), 'statistics': results,
            'scope': 'Original reassignment hypothesis failed; retained prefix matches unchanged'}
    gt_tail = np.asarray([[0, 10, 10, 20, 20], [0, 40, 40, 50, 50]], dtype=np.float32)
    pred_tail = np.asarray([[10, 10, 20, 20, .9, 0], [70, 70, 80, 80, .1, 0]], dtype=np.float32)
    full_tail, retained_tail = match_pair(gt_tail, pred_tail, 1)
    tail_results = {}
    for name, p, t in [('all2', pred_tail, full_tail),
                       ('top1_fixed_matches', pred_tail[:1], full_tail[:1]),
                       ('top1_rematched', pred_tail[:1], retained_tail)]:
        metric = new_metrics()
        update(metric, gt_tail, p, t)
        tail_results[name] = finish(metric)
    if not np.array_equal(full_tail[:1], retained_tail):
        raise RuntimeError('Tail control unexpectedly changed retained matching')
    if not np.isclose(tail_results['all2']['ap50_percent'], 62.25) or not np.isclose(tail_results['top1_rematched']['ap50_percent'], 75):
        raise RuntimeError('Pinned AP tail control differs from measured exploration')
    return {'opposed_confidence_iou': opposed,
            'incomplete_recall_fp_tail': {'gt': gt_tail.tolist(), 'predictions': pred_tail.tolist(),
                 'statistics': tail_results, 'scope': 'Constructed AP integration control; recall stays 0.5'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-run', required=True)
    parser.add_argument('--cap-analysis-run', required=True)
    parser.add_argument('--output-run', required=True)
    args = parser.parse_args()
    import torch
    import ultralytics
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import compute_ap
    torch.set_num_threads(2)
    if ultralytics.__version__ != '8.3.253':
        raise RuntimeError('Pinned matcher version changed')
    run = PROJECT/'artifacts'/safe_run_id(args.prediction_run)
    earlier = PROJECT/'artifacts'/safe_run_id(args.cap_analysis_run)
    dest = PROJECT/'artifacts'/safe_run_id(args.output_run)
    if dest.exists():
        raise FileExistsError(dest)
    record_path = run/'run_record.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    log = run/'predictions_and_timings.jsonl'
    log_hash = digest(log)
    previous_path = earlier/'summary.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8'))
    if record['status'] != 'completed' or record['purpose'] != 'development-study':
        raise RuntimeError('Require completed development predictions')
    if log_hash != record['prediction_log_sha256'] or log_hash != previous['input_hashes']['prediction_log']:
        raise RuntimeError('Prediction log changed')
    manifest_path = PROJECT/'data/yolo_full_uvd1/manifest.json'
    if digest(manifest_path) != record['dataset_manifest_sha256']:
        raise RuntimeError('Dataset manifest changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    audit_path = PROJECT/'artifacts/train_val_audit_20261016.json'
    if digest(audit_path) != manifest['audit_file_sha256']:
        raise RuntimeError('Audit changed')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    dimensions = {Path(x['image']).name: (x['width'], x['height']) for x in audit['splits']['val']['manifest']}
    entries = {x['image']: x for x in manifest['splits']['val']['entries']}
    summary_path = run/'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    methods = list(summary)
    if len(entries) != 548 or set(entries) != set(record['selection']) or len(record['selection']) != 548:
        raise RuntimeError('Require full unique development selection')
    if set(methods) != set(previous['methods']):
        raise RuntimeError('Methods changed')
    gts = {}
    for name, entry in entries.items():
        label = PROJECT/'data/yolo_full_uvd1/labels/val'/f'{Path(name).stem}.txt'
        if digest(label) != entry['label_sha256']:
            raise RuntimeError('Labels changed')
        gts[name] = label_rows_as_xyxy(label, *dimensions[name])
    if sum(map(len, gts.values())) != 38759:
        raise RuntimeError('Ground truth count changed')
    control = constructed_controls()
    dest.mkdir()
    source_path = Path(__file__)
    (dest/'analysis_source.py').write_bytes(source_path.read_bytes())
    plan = PROJECT/'Cap_Matching_Diagnostic_Plan_2026-10-16.md'
    (dest/plan.name).write_bytes(plan.read_bytes())
    metrics = {name: {key: new_metrics() for key in ['all_saved', 'retained_fixed_matches', 'retained_rematched']} for name in methods}
    counts = {name: {'retained_fp_to_tp_by_iou': np.zeros(10, dtype=np.int64),
                     'retained_tp_to_fp_by_iou': np.zeros(10, dtype=np.int64),
                     'removed_tp_by_iou': np.zeros(10, dtype=np.int64),
                     'images_changed': 0, 'retained_detections_changed_at_any_iou': 0,
                     'images': 0, 'truncated_images': 0, 'removed_detections': 0, 'examples': []} for name in methods}
    seen = set()
    for line in log.open(encoding='utf-8'):
        row = json.loads(line)
        image, method = row['image'], row['method']
        if (image, method) in seen or image not in gts or method not in metrics:
            raise RuntimeError('Unexpected or duplicate identity')
        seen.add((image, method))
        gt = gts[image]
        pred = prediction_rows(row['predictions'])
        if len(pred) > 1000:
            raise RuntimeError('Saved cap exceeded')
        full, retained = match_pair(gt, pred, 500)
        fixed = full[:500]
        gain, loss = retained & ~fixed, fixed & ~retained
        c = counts[method]
        c['images'] += 1
        c['truncated_images'] += int(len(pred) > 500)
        c['removed_detections'] += max(0, len(pred)-500)
        c['retained_fp_to_tp_by_iou'] += gain.sum(axis=0)
        c['retained_tp_to_fp_by_iou'] += loss.sum(axis=0)
        c['removed_tp_by_iou'] += full[500:].sum(axis=0)
        changed = np.any(gain | loss, axis=1)
        c['images_changed'] += int(changed.any())
        c['retained_detections_changed_at_any_iou'] += int(changed.sum())
        if len(pred) <= 500 and changed.any():
            raise RuntimeError('Untruncated outputs unexpectedly changed assignment')
        if changed.any() and len(c['examples']) < 4:
            index = int(np.flatnonzero(changed)[0])
            c['examples'].append({'image': image, 'zero_based_rank': index,
                                  'prediction': pred[index].tolist(),
                                  'full_match_tp': fixed[index].tolist(),
                                  'rematched_tp': retained[index].tolist()})
        for key, p, t in [('all_saved', pred, full), ('retained_fixed_matches', pred[:500], fixed),
                           ('retained_rematched', pred[:500], retained)]:
            update(metrics[method][key], gt, p, t)
    if seen != {(image, method) for image in entries for method in methods}:
        raise RuntimeError('Missing declared outputs')
    results = {}
    for method in methods:
        statistics = {key: finish(value) for key, value in metrics[method].items()}
        expected_full = 100*summary[method]['scoring']['metrics']['metrics/mAP50-95(B)']
        expected_cap = previous['methods'][method]['converted_cap500_ap_percent']
        if not np.isclose(statistics['all_saved']['ap50_95_percent'], expected_full, atol=1e-10, rtol=0):
            raise RuntimeError('Full-set AP not reproduced')
        if not np.isclose(statistics['retained_rematched']['ap50_95_percent'], expected_cap, atol=1e-10, rtol=0):
            raise RuntimeError('Cap500 AP not reproduced')
        c = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in counts[method].items()}
        full_ap = statistics['all_saved']['ap50_95_percent']
        fixed_ap = statistics['retained_fixed_matches']['ap50_95_percent']
        rematched_ap = statistics['retained_rematched']['ap50_95_percent']
        results[method] = {'statistics': statistics, 'counts': c,
                           'fixed_assignment_deletion_delta_pp': fixed_ap-full_ap,
                           'rematching_on_retained_delta_pp': rematched_ap-fixed_ap,
                           'total_observed_cap_change_pp': rematched_ap-full_ap,
                           'ap_grid_area_beyond_attained_recall_change_pp': statistics['retained_rematched']['ap_grid_area_beyond_attained_recall_percent']-statistics['all_saved']['ap_grid_area_beyond_attained_recall_percent'],
                           'ap_grid_area_at_or_below_attained_recall_change_pp': statistics['retained_rematched']['ap_grid_area_at_or_below_attained_recall_percent']-statistics['all_saved']['ap_grid_area_at_or_below_attained_recall_percent']}
        print(json.dumps({'method': method, 'full_ap': full_ap, 'fixed_ap': fixed_ap,
                          'rematched_ap': rematched_ap, 'changed_images': c['images_changed']}), flush=True)
    output = {'utc': datetime.now(timezone.utc).isoformat(), 'constructed_controls': control,
              'iou_thresholds': torch.linspace(.5, .95, 10).tolist(), 'methods': results,
              'hashes': {'prediction_record': digest(record_path), 'prediction_log': log_hash,
                         'summary': digest(summary_path), 'cap_analysis': digest(previous_path),
                         'manifest': digest(manifest_path), 'analysis_source': digest(source_path),
                         'plan': digest(plan), 'validator_source': digest(Path(inspect.getfile(BaseValidator))),
                         'metrics_source': digest(Path(inspect.getfile(compute_ap)))},
              'limits': ['Post-observation diagnostic on one selected checkpoint and development set',
                         'Fixed-assignment statistic is artificial and is not a normal benchmark score',
                         'Path-specific AP contrasts; not a universal causal decomposition',
                         'No inference change, new method, recall improvement, toolkit-bug or novelty claim',
                         'Original per-view and final1000 caps cannot be undone from saved outputs']}
    (dest/'summary.json').write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
    print('Completed saved-output cap/matching diagnostic: '+str(dest), flush=True)


if __name__ == '__main__':
    main()
