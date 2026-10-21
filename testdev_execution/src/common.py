# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Explicit paths, stable task identity and source continuity."""
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PORTABLE = ROOT.parent / 'portable_execution' / 'src'
sys.path.insert(0, str(PORTABLE))
from task_pin import TASK_SHA256
from audit_dataset import digest

def require(ok, message):
    if not ok:
        raise ValueError(message)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')

def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_bytes((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
    temporary.replace(path)

def now():
    return datetime.now(timezone.utc).isoformat()

def load_task(path):
    path = Path(path)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == TASK_SHA256, 'Task/config differs from the reviewed release-time content pin')
    task = json.loads(raw.decode('utf-8'))
    require(task['schema'] == 'uvd1-portable-testdev-task-v1', 'Unsupported task schema')
    validate_helpers(task)
    return task

def validate_helpers(task):
    actual = {p.name: digest(p) for p in sorted(PORTABLE.glob('*.py'))}
    require(actual == task['helper_sha256'], 'Pinned portable helper source changed')

def source_files():
    return {**{'src/' + p.name: p for p in sorted((ROOT/'src').glob('*.py'))},
            **{'portable_src/' + p.name: p for p in sorted(PORTABLE.glob('*.py'))}}

def source_hashes():
    return {name: digest(path) for name, path in source_files().items()}

def snapshot_sources(dest):
    import shutil
    dest = Path(dest)
    hashes = source_hashes()
    for name, path in source_files().items():
        target = dest/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return hashes

def verify_sources(dest, expected):
    require(source_hashes() == expected, 'Current source changed since this artefact was bound')
    actual = {str(p.relative_to(dest)).replace('\\', '/'): digest(p) for p in Path(dest).rglob('*.py')}
    require(actual == expected, 'Source snapshot changed or has missing/extra files')

def offline_environment(output):
    # Must run before importing Ultralytics; never download models or install dependencies.
    output = Path(output)
    os.environ['YOLO_CONFIG_DIR'] = str(output/'runtime_settings')
    os.environ['MPLCONFIGDIR'] = str(output/'matplotlib_settings')
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_AUTOINSTALL'] = 'false'

def runtime_context(task, *, exact_inference=False):
    import cv2
    import numpy as np
    import torch
    import ultralytics
    versions = {'ultralytics': ultralytics.__version__, 'torch': torch.__version__,
                'numpy': np.__version__, 'opencv': cv2.__version__,
                'torchvision': importlib.metadata.version('torchvision')}
    validate_versions(versions,task,exact_inference=exact_inference)
    paths = {'metrics': 'utils/metrics.py', 'matching': 'engine/validator.py',
             'predictor': 'models/yolo/detect/predict.py', 'nms': 'utils/nms.py',
             'detection_validator': 'models/yolo/detect/val.py'}
    hashes = {name: digest(Path(ultralytics.__file__).parent/path) for name, path in paths.items()}
    require(hashes == task['framework_source_hashes'], 'Framework AP/matching/predictor/NMS source differs')
    return {'versions': versions, 'framework_source_hashes': hashes,
            'cuda_runtime': torch.version.cuda, 'torch_num_threads': torch.get_num_threads()}


def validate_versions(versions, task, *, exact_inference=False):
    expected = task['runtime_versions']
    require(versions['ultralytics'] == expected['ultralytics'], 'Unsupported Ultralytics version')
    # Scoring on CPU can use CPU wheel suffixes; exact inference replay pins the observed wheel versions.
    for key in ('torch', 'torchvision'):
        wanted = expected[key] if exact_inference else expected[key].split('+')[0]
        actual = versions[key] if exact_inference else versions[key].split('+')[0]
        require(actual == wanted, 'Unsupported ' + key + ' version')
    for key in ('numpy', 'opencv'):
        require(versions[key] == expected[key], 'Unsupported ' + key + ' version')


def parameter_device(requested):
    require(isinstance(requested,str), 'Explicit string device required')
    if requested == 'cpu':
        return 'cpu'
    match = re.fullmatch(r'(?:cuda:)?([0-9]+)',requested)
    require(match is not None, 'Use cpu or one explicit CUDA device index')
    # The pinned Ultralytics single-device selector sets visibility, then returns logical cuda:0.
    return 'cuda:0'


def prediction_arguments(task, device):
    parameter_device(device)
    policy = task['inference']
    return dict(device=device, conf=policy['confidence'], iou=policy['detector_nms_iou'],
                max_det=policy['max_detections'], half=False, rect=False, batch=1,
                verbose=False, save=False, save_txt=False, save_json=False)
