# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Save controlled full-frame/fixed-slicing predictions and explicit timing records.

Pipeline-smoke mode never establishes detector quality or a runtime benchmark.
Development-study mode requires a completed, checked full-baseline checkpoint.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import platform

PROJECT=Path(__file__).resolve().parents[1]
CONFIG_DIR=PROJECT/'artifacts/ultralytics_config'
(CONFIG_DIR/'Ultralytics').mkdir(parents=True,exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(CONFIG_DIR)
os.environ['YOLO_OFFLINE']='true'
os.environ['YOLO_AUTOINSTALL']='false'
os.environ['MPLCONFIGDIR']=str(PROJECT/'artifacts/matplotlib_config')

from audit_dataset import digest
from inference_geometry import Window,fixed_windows,to_image_coordinates,merge_predictions
from prepare_training_pilot import NAMES
from score_converted_predictions import label_rows_as_xyxy,score_records
from run_records import canonical_output_hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--purpose',choices=['pipeline-smoke','development-study'],required=True)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--device',default='0')
    parser.add_argument('--image-limit',type=int)
    parser.add_argument('--baseline-run',default='b001_uvd1_yolo11n640_seed0')
    args=parser.parse_args()
    for value in (args.run_name,args.baseline_run):
        if Path(value).name!=value or ':' in value or '\\' in value:
            raise ValueError('Unsafe run id')
    if args.image_limit is not None and args.image_limit<1:
        raise ValueError('Positive image limit required')
    if args.purpose=='pipeline-smoke' and args.device!='cpu':
        raise ValueError('Initial pipeline smoke uses CPU; keep training GPU unoccupied')
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO,settings
    settings.update({key:False for key in ('sync','hub','wandb','comet','clearml','mlflow','neptune','tensorboard','dvc')})
    torch.set_num_threads(2)
    config_path=PROJECT/'configs/inference_comparison_development_v1.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    if ultralytics.__version__!=config['expected_ultralytics']:
        raise RuntimeError('Prediction policy must be re-reviewed for a different framework version')
    manifest_path=PROJECT/'data/yolo_full_uvd1/manifest.json'
    manifest_hash=digest(manifest_path)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    entries=manifest['splits']['val']['entries']
    if manifest['status']!='complete' or manifest_hash!=config['expected_training_manifest_sha256']:
        raise RuntimeError('Dataset manifest differs from the reviewed frozen task')
    if len(entries)!=config['expected_validation_images'] or len({x['image'] for x in entries})!=len(entries):
        raise RuntimeError('Development selection must contain the declared unique validation images')
    method_ids=[method['name'] for method in config['methods']]
    if len(set(method_ids))!=len(method_ids):
        raise RuntimeError('Duplicate comparison method id')
    if config['precision']!='FP32' or config['batch']!=1 or config['rectangular_padding']:
        raise RuntimeError('Unsupported precision/batch/padding change; revise the runner explicitly')
    weights=args.weights.resolve()
    if not weights.is_relative_to(PROJECT) or not weights.exists():
        raise ValueError('Use an existing project-local checkpoint')
    weights_hash=digest(weights)
    if args.purpose=='development-study':
        baseline=PROJECT/'artifacts'/args.baseline_run
        record=json.loads((baseline/'run_record.json').read_text(encoding='utf-8'))
        progress=json.loads((baseline/'progress.json').read_text(encoding='utf-8'))
        if record['status']!='completed' or progress['epochs_completed']!=progress['epochs_planned']:
            raise RuntimeError('Baseline training is incomplete; do not treat an intermediate checkpoint as final')
        if weights_hash!=canonical_output_hashes(record['output_hashes'])['fit/weights/best.pt']:
            raise RuntimeError('Study checkpoint differs from the completed baseline selection')
        if manifest_hash!=record['manifest_sha256']:
            raise RuntimeError('Study data differ from the completed baseline task')
        if args.image_limit is not None:
            raise ValueError('Full development sweep must use all declared images')
    if shutil.disk_usage(PROJECT).free<2*1024**3:
        raise RuntimeError('Insufficient disk reserve')
    dest=PROJECT/'artifacts'/args.run_name
    if dest.exists():
        raise RuntimeError('Run exists; preserve it rather than overwrite')
    dest.mkdir()
    snapshot=dest/'source_snapshot'
    snapshot.mkdir()
    for source in [Path(__file__),PROJECT/'src/inference_geometry.py',PROJECT/'src/score_converted_predictions.py',PROJECT/'src/run_records.py',config_path]:
        shutil.copyfile(source,snapshot/source.name)
    if args.image_limit is not None:
        entries=entries[:args.image_limit]
    names=dict(enumerate(NAMES))
    model=YOLO(str(weights))
    if dict(model.names)!=names:
        raise RuntimeError('Checkpoint names are not the declared ten-class task')
    gpu=args.device!='cpu'
    if gpu and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA unavailable')
    sync=lambda:torch.cuda.synchronize() if gpu else None
    predict_args=dict(device=args.device,conf=config['confidence'],iou=config['detector_nms_iou'],
                      max_det=config['max_detections'],half=False,rect=False,batch=1,
                      verbose=False,save=False,save_txt=False,save_json=False)
    model_record={'status':'running','purpose':args.purpose,'started_utc':datetime.now(timezone.utc).isoformat(),
                  'weights':str(weights),'weights_sha256':weights_hash,'config_sha256':digest(config_path),
                  'dataset_manifest_sha256':digest(manifest_path),'device_requested':args.device,
                  'ultralytics':ultralytics.__version__,'torch':torch.__version__,
                  'candidate_policy':config['candidate_policy'],
                  'predict_arguments':predict_args,
                  'environment':{'python':platform.python_version(),'platform':platform.platform(),
                                 'numpy':np.__version__,'opencv':cv2.__version__,
                                 'torchvision':importlib.metadata.version('torchvision'),
                                 'cuda_runtime':torch.version.cuda,
                                 'gpu_name':torch.cuda.get_device_name(0) if gpu else None},
                  'selection':[x['image'] for x in entries],
                  'source_hashes':{x.name:digest(x) for x in snapshot.iterdir()},
                  'framework_source_hashes':{name:digest(Path(ultralytics.__file__).parent/path) for name,path in {
                      'metrics':'utils/metrics.py','matching':'engine/validator.py','predictor':'models/yolo/detect/predict.py',
                      'nms':'utils/nms.py','detection_validator':'models/yolo/detect/val.py'}.items()},
                  'limits':['Converted ten-class scoring, no native ignore semantics',
                            'No final test evaluation or novelty claim',
                            'Timing from pipeline-smoke is not benchmark evidence',
                            config['timing_limits']]}
    (dest/'run_record.json').write_text(json.dumps(model_record,indent=2)+'\n',encoding='utf-8')
    method_records={method['name']:[] for method in config['methods']}
    all_timings=[]
    started=time.perf_counter()
    try:
        for size in sorted({m['input_size'] for m in config['methods']}):
            for _ in range(config['warmup_per_input_size']):
                model.predict(np.zeros((640,640,3),dtype=np.uint8),imgsz=size,**predict_args)
        sync()
        model_record['actual_parameter_device']=str(next(model.model.parameters()).device)
        with (dest/'predictions_and_timings.jsonl').open('w',encoding='utf-8') as stream:
            for index,entry in enumerate(entries):
                image_path=PROJECT/'data/yolo_full_uvd1/images/val'/entry['image']
                label_path=PROJECT/'data/yolo_full_uvd1/labels/val'/f"{image_path.stem}.txt"
                if digest(image_path)!=entry['image_sha256'] or digest(label_path)!=entry['label_sha256']:
                    raise RuntimeError('Evaluation input differs from manifest')
                methods=config['methods'][index%len(config['methods']):]+config['methods'][:index%len(config['methods'])]
                for method in methods:
                    sync()
                    t0=time.perf_counter()
                    image=cv2.imread(str(image_path))
                    if image is None:
                        raise RuntimeError('Image decoding failed')
                    height,width=image.shape[:2]
                    t1=time.perf_counter()
                    windows=[Window(0,0,width,height)]
                    if method['kind']=='fixed':
                        windows+=fixed_windows(width,height,method['tile_size'],method['overlap'])
                    parts=[]
                    view_counts=[]
                    stats={'clipped':0,'zero_area_removed':0}
                    for window in windows:
                        view=image[window.y1:window.y2,window.x1:window.x2]
                        result=model.predict(view,imgsz=method['input_size'],**predict_args)[0]
                        rows,counts=to_image_coordinates(result.boxes.data.cpu().numpy(),window)
                        parts.append(rows)
                        view_counts.append({'window_xyxy':[window.x1,window.y1,window.x2,window.y2],
                                            'output_before_geometry':len(result.boxes),'output_after_geometry':len(rows)})
                        for key,value in counts.items():
                            stats[key]+=value
                    sync()
                    t2=time.perf_counter()
                    if len(parts)==1:
                        predictions=parts[0][np.argsort(-parts[0][:,4],kind='stable')][:config['max_detections']]
                        merge_counts={'before_merge':len(parts[0]),'after_nms_before_cap':len(parts[0]),
                                      'after_cap':len(predictions),'removed_nms':0,
                                      'removed_cap':len(parts[0])-len(predictions)}
                    else:
                        predictions,merge_counts=merge_predictions(parts,config['merge_nms_iou'],config['max_detections'],return_counts=True)
                    sync()
                    t3=time.perf_counter()
                    timing={'image':entry['image'],'method':method['name'],'decode_ms':(t1-t0)*1000,
                            'views_ms':(t2-t1)*1000,'merge_ms':(t3-t2)*1000,'end_to_end_ms':(t3-t0)*1000,
                            'detector_passes':len(windows),'processed_input_pixels':len(windows)*method['input_size']**2,
                            'prediction_count':len(predictions),'geometry':stats,
                            'per_view_counts':view_counts,'merge_counts':merge_counts}
                    all_timings.append(timing)
                    stream.write(json.dumps({'image':entry['image'],'method':method['name'],
                                              'image_sha256':entry['image_sha256'],'timing':timing,
                                              'predictions':predictions.tolist()})+'\n')
                    stream.flush()
                    gt=label_rows_as_xyxy(label_path,width,height)
                    method_records[method['name']].append({'gt':gt,'predictions':predictions})
                print(f'Compared {index+1}/{len(entries)} images; purpose={args.purpose}',flush=True)
                if shutil.disk_usage(PROJECT).free<1024**3:
                    raise RuntimeError('Disk reserve below 1 GiB')
        summary={}
        for method in config['methods']:
            values=[r['end_to_end_ms'] for r in all_timings if r['method']==method['name']]
            summary[method['name']]={'scoring':score_records(method_records[method['name']],names),
                                     'diagnostic_median_ms':float(np.median(values)),
                                     'diagnostic_p95_ms':float(np.percentile(values,95))}
        (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
        model_record.update(status='completed',seconds=time.perf_counter()-started,
                            images_compared=len(entries),prediction_log_sha256=digest(dest/'predictions_and_timings.jsonl'))
    except BaseException as exc:
        model_record.update(status='failed',error=repr(exc),seconds=time.perf_counter()-started)
        raise
    finally:
        model_record['finished_utc']=datetime.now(timezone.utc).isoformat()
        (dest/'run_record.json').write_text(json.dumps(model_record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':model_record['status'],'purpose':args.purpose,'images':len(entries),
                      'note':'Pipeline-smoke scores/timings are not study results'},indent=2),flush=True)


if __name__=='__main__':
    main()
