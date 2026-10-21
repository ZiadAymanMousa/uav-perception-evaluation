# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Execute the unchanged four inference settings on explicit test-dev inputs."""
import os
from pathlib import Path
import shutil
import time
from common import (require, digest, write_json, read_json, now, snapshot_sources,
                    source_hashes, offline_environment, runtime_context, prediction_arguments, parameter_device)
from data import verify
from checkpoints import checkpoint_context
from records import methods_for_index, windows_for, score_log
from inference_geometry import to_image_coordinates, merge_predictions

def infer(raw_root, prepared, output, config_path, task, weights, *, mode, seed=None,
          smoke=False, device='cpu', baseline_run=None, training_data=None,
          training_config=None, training_binding=None):
    output, prepared, raw_root = Path(output).resolve(), Path(prepared).resolve(), Path(raw_root).resolve()
    require(not output.exists(), 'Run destination exists; preserve it')
    require(not output.is_relative_to(raw_root) and not output.is_relative_to(prepared), 'Output must be separate from raw/prepared data')
    require(not smoke or device == 'cpu', 'One-image smoke requires CPU')
    parameter_device(device)
    context = verify(raw_root, prepared, config_path, task)
    options = dict(seed=seed, baseline_run=baseline_run, training_data=training_data,
                   training_config=training_config, training_binding=training_binding)
    checkpoint = checkpoint_context(task, weights, mode, **options)
    output.mkdir(parents=True)
    offline_environment(output)
    import torch
    torch.set_num_threads(2)
    runtime = runtime_context(task, exact_inference=(mode == 'frozen-replay' and not smoke))
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO, settings
    settings.update({key: False for key in ('sync','hub','wandb','comet','clearml','mlflow','neptune','tensorboard','dvc')})
    torch.set_num_threads(2)
    gpu = device != 'cpu'
    require(not gpu or torch.cuda.is_available(), 'Requested CUDA unavailable')
    model = YOLO(str(Path(weights).resolve()))
    require(dict(model.names) == dict(enumerate(task['class_names'])), 'Checkpoint class schema differs')
    sync = lambda: torch.cuda.synchronize() if gpu else None
    policy = task['inference']
    predict_args = prediction_arguments(task,device)
    entries = read_json(prepared/'manifest.json')['entries']
    entries = entries[:1] if smoke else entries
    selection = [e['image'] for e in entries]
    sources = snapshot_sources(output/'source_snapshot')
    shutil.copyfile(config_path, output/'task_config.json')
    write_json(output/'selection.json', selection)
    record = {'schema': 'uvd1-portable-testdev-run-v1', 'status': 'running',
              'scope': 'one-image-cpu-smoke' if smoke else 'complete-testdev',
              'started_utc': now(), 'pid': os.getpid(), 'prepared_root': str(prepared),
              'data_context': context, 'checkpoint_context': checkpoint,
              'config_sha256': digest(Path(config_path)), 'source_hashes': sources,
              'selection': selection, 'selection_sha256': digest(output/'selection.json'),
              'environment': runtime, 'device_requested': device, 'predict_arguments': predict_args,
              'limits': ['Same pinned scorer; rescoring is not an independent evaluator',
                         'Converted ten-class AP; no native ignore masking',
                         'Publicly labelled held-out split; not blind, OOD or independent-scene proof',
                         'Exact checkpoint identity does not promise identical floating-point results across hardware',
                         policy['timing_limits']]}
    write_json(output/'run_record.json', record)
    completed, warmups, rows_written = 0, 0, 0
    started = time.perf_counter()
    try:
        for size in sorted({m['input_size'] for m in policy['methods']}):
            for _ in range(policy['warmup_per_input_size']):
                model.predict(np.zeros((640,640,3), dtype=np.uint8), imgsz=size, **predict_args)
                warmups += 1
        sync()
        record['actual_parameter_device'] = str(next(model.model.parameters()).device)
        # Framework device selection can adjust CPU threads during the first prediction.
        record['environment']['torch_num_threads'] = torch.get_num_threads()
        with (output/'predictions_and_timings.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
            for index, entry in enumerate(entries):
                image_path = prepared/'images/testdev'/entry['image']
                for method in methods_for_index(task, index):
                    require(digest(image_path) == entry['image_sha256'], 'Inference image bytes changed')
                    sync()
                    t0 = time.perf_counter()
                    image = cv2.imread(str(image_path))
                    require(image is not None and image.shape[:2] == (entry['height'],entry['width']), 'Image decode geometry differs')
                    t1 = time.perf_counter()
                    windows, parts, views = windows_for(entry, method), [], []
                    stats = {'clipped': 0, 'zero_area_removed': 0}
                    for window in windows:
                        view = image[window.y1:window.y2,window.x1:window.x2]
                        result = model.predict(view, imgsz=method['input_size'], **predict_args)[0]
                        predictions, counts = to_image_coordinates(result.boxes.data.cpu().numpy(), window)
                        parts.append(predictions)
                        views.append({'window_xyxy': [window.x1,window.y1,window.x2,window.y2],
                                      'output_before_geometry': len(result.boxes), 'output_after_geometry': len(predictions)})
                        for key, value in counts.items():
                            stats[key] += value
                    sync()
                    t2 = time.perf_counter()
                    if len(parts) == 1:
                        predictions = parts[0][np.argsort(-parts[0][:,4],kind='stable')][:policy['max_detections']]
                        merge = {'before_merge':len(parts[0]), 'after_nms_before_cap':len(parts[0]),
                                 'after_cap':len(predictions), 'removed_nms':0, 'removed_cap':len(parts[0])-len(predictions)}
                    else:
                        predictions, merge = merge_predictions(parts, policy['merge_nms_iou'], policy['max_detections'], return_counts=True)
                    sync()
                    t3 = time.perf_counter()
                    timing = {'image':entry['image'], 'method':method['name'], 'decode_ms':(t1-t0)*1000,
                              'views_ms':(t2-t1)*1000, 'merge_ms':(t3-t2)*1000, 'end_to_end_ms':(t3-t0)*1000,
                              'detector_passes':len(windows), 'processed_input_pixels':len(windows)*method['input_size']**2,
                              'prediction_count':len(predictions), 'geometry':stats, 'per_view_counts':views, 'merge_counts':merge}
                    import json
                    stream.write(json.dumps({'image':entry['image'],'method':method['name'],
                                             'image_sha256':entry['image_sha256'], 'timing':timing,
                                             'predictions':predictions.tolist()}, allow_nan=False)+'\n')
                    stream.flush()
                    rows_written += 1
                completed += 1
                write_json(output/'progress.json', {'status':'inference','images_completed':completed,
                           'images_planned':len(entries),'records_written':rows_written,'utc':now()})
                if completed % 25 == 0 or completed == len(entries):
                    print(f'Compared {completed}/{len(entries)} images',flush=True)
        summary, accounting = score_log(output/'predictions_and_timings.jsonl', prepared, task, smoke=smoke)
        require(verify(raw_root,prepared,config_path,task) == context, 'Dataset/config/source changed during execution')
        require(checkpoint_context(task,weights,mode,**options) == checkpoint, 'Checkpoint/training changed during execution')
        require(source_hashes() == sources, 'Source changed during execution')
        write_json(output/'summary.json', summary)
        record.update(status='completed', images_compared=completed, warmup_calls=warmups,
                      accounting=accounting, prediction_log_sha256=digest(output/'predictions_and_timings.jsonl'),
                      summary_sha256=digest(output/'summary.json'))
    except BaseException as exc:
        record.update(status='failed',error=repr(exc),images_compared=completed,warmup_calls=warmups)
        raise
    finally:
        record.update(finished_utc=now(),seconds=time.perf_counter()-started)
        write_json(output/'run_record.json',record)
        write_json(output/'progress.json',{'status':record['status'],'images_completed':completed,
                   'images_planned':len(entries),'records_written':rows_written,'utc':now()})
    return record
