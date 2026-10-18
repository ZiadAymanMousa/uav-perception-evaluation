# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Own saved-output input validation and coordinate bridge; no reference evaluator."""
import json
from pathlib import Path
import numpy as np
from audit_dataset import digest
from inference_geometry import prediction_rows
PROJECT=Path(__file__).resolve().parents[1]

def native_detections(value):
    """xyxy/conf/class0..9 -> xywh/conf/class1..10/-1/-1, no coordinate shift."""
    rows=prediction_rows(value).astype(np.float64)
    if np.any(rows[:,2:4]<=rows[:,:2]):
        raise ValueError('Native bridge requires positive-area predictions')
    if len(rows)>1 and np.any(np.diff(rows[:,4])>0):
        raise ValueError('Preserve a globally score-sorted prediction list')
    output=np.full((len(rows),8),-1.,dtype=np.float64)
    output[:,:2]=rows[:,:2]
    output[:,2:4]=rows[:,2:4]-rows[:,:2]
    output[:,4]=rows[:,4]
    output[:,5]=rows[:,5]+1
    return output

def safe_id(value):
    if Path(value).name!=value or ':' in value or '\\' in value or value in ('.','..'):
        raise ValueError('Unsafe artifact id')
    return value

def load_cases(run_name):
    run=PROJECT/'artifacts'/safe_id(run_name)
    record=json.loads((run/'run_record.json').read_text(encoding='utf-8'))
    if record['status']!='completed':
        raise RuntimeError('Prediction run is incomplete')
    manifest_path=PROJECT/'data/yolo_full_uvd1/manifest.json'
    if digest(manifest_path)!=record['dataset_manifest_sha256']:
        raise RuntimeError('Converted dataset manifest changed')
    config_path=run/'source_snapshot/inference_comparison_development_v1.json'
    if digest(config_path)!=record['config_sha256']:
        raise RuntimeError('Inference configuration snapshot changed')
    config=json.loads(config_path.read_text(encoding='utf-8'))
    methods=[x['name'] for x in config['methods']]
    if len(set(methods))!=len(methods):
        raise RuntimeError('Duplicate method ids')
    log_path=run/'predictions_and_timings.jsonl'
    if digest(log_path)!=record['prediction_log_sha256']:
        raise RuntimeError('Saved predictions changed')
    selected=record['selection']
    if not selected or len(set(selected))!=len(selected):
        raise RuntimeError('Invalid image selection')
    audit_path=PROJECT/'artifacts/train_val_audit_20261016.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if digest(audit_path)!=manifest['audit_file_sha256']:
        raise RuntimeError('Original annotation audit changed')
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    original={Path(x['image']).name:x for x in audit['splits']['val']['manifest']}
    if len(original)!=len(audit['splits']['val']['manifest']):
        raise RuntimeError('Ambiguous original image basenames')
    converted={x['image']:x for x in manifest['splits']['val']['entries']}
    rows={}
    for line in log_path.read_text(encoding='utf-8').splitlines():
        item=json.loads(line)
        key=(item['image'],item['method'])
        if key in rows or item['image'] not in selected or item['method'] not in methods:
            raise RuntimeError('Unexpected/duplicate prediction record')
        if item['image_sha256']!=converted[item['image']]['image_sha256']:
            raise RuntimeError('Prediction image identity differs from converted manifest')
        rows[key]=item
    if len(rows)!=len(selected)*len(methods):
        raise RuntimeError('Missing image/method output')
    output={method:[] for method in methods}
    annotations=[]
    for image_name in selected:
        if Path(image_name).name!=image_name or '\\' in image_name:
            raise RuntimeError('Unsafe image name')
        item=original[image_name]
        source_image=PROJECT/'data/VisDrone2019-DET-val/images'/image_name
        annotation=PROJECT/'data/VisDrone2019-DET-val/annotations'/f'{Path(image_name).stem}.txt'
        if digest(source_image)!=item['image_sha256'] or digest(annotation)!=item['annotation_sha256']:
            raise RuntimeError('Original validation inputs changed')
        if item['image_sha256']!=converted[image_name]['image_sha256']:
            raise RuntimeError('Original and converted images differ')
        gt=np.loadtxt(annotation,delimiter=',',dtype=np.float64,ndmin=2)
        annotations.append({'image':image_name,'annotation_sha256':item['annotation_sha256']})
        for method in methods:
            output[method].append({'name':Path(image_name).stem,'height':item['height'],'width':item['width'],
                                   'gt':gt.tolist(),'detections':native_detections(rows[(image_name,method)]['predictions']).tolist()})
    return output,{'prediction_run':run_name,'prediction_purpose':record['purpose'],
                   'prediction_record_sha256':digest(run/'run_record.json'),
                   'prediction_log_sha256':digest(log_path),'dataset_manifest_sha256':digest(manifest_path),
                   'audit_sha256':digest(audit_path),'source_annotations':annotations,
                   'weights_sha256':record['weights_sha256'],'config_sha256':record['config_sha256']}
