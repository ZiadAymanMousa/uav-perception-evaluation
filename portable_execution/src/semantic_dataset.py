# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Private task-identity adapter; extracted reviewed functions, no experiments."""
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

def label_identity(path, class_count):
    lines = [line.split() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not lines:
        return (hashlib.sha256(b'').hexdigest(), 0)
    if any((len(row) != 5 for row in lines)):
        raise ValueError('Expected five columns')
    a = np.asarray(lines, dtype=np.float32)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite label')
    if not ((a[:, 0] >= 0) & (a[:, 0] < class_count) & (a[:, 0] == np.floor(a[:, 0]))).all():
        raise ValueError('Invalid class')
    if not ((a[:, 1:] >= 0) & (a[:, 1:] <= 1)).all() or not (a[:, 3:] > 0).all():
        raise ValueError('Invalid normalized box')
    rows = [row.tobytes() for row in a.astype('<f4', copy=False)]
    if len(set(rows)) != len(rows):
        raise ValueError('Duplicate converted float32 labels')
    return (hashlib.sha256(a.astype('<f4', copy=False).tobytes(order='C')).hexdigest(), len(a))

def identity(root, manifest=None, image_mode='rehash'):
    if image_mode not in {'rehash', 'cached'}:
        raise ValueError('Unknown image verification mode')
    m = manifest or load(root / 'manifest.json')
    if m['status'] != 'complete' or set(m['splits']) != {'train', 'val'}:
        raise ValueError('Incomplete or different split set')
    policy = {k: m[k] for k in ['protocol', 'class_names', 'policy', 'exceptions']}
    policy['schema'] = 'converted-task-semantic-identity-prototype-v1'
    policy['splits'] = {}
    images = targets = 0
    for split in ['train', 'val']:
        entries = m['splits'][split]['entries']
        names = [e['image'] for e in entries]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate image identifier')
        if any((Path(n).name != n or not n.lower().endswith('.jpg') for n in names)):
            raise ValueError('Unsafe image name')
        expected_labels = {Path(n).stem + '.txt' for n in names}
        expected_sides = {Path(n).stem + '.json' for n in names}
        for folder, expected in [('images', set(names)), ('labels', expected_labels), ('excluded_annotations', expected_sides)]:
            actual = {p.name for p in (root / folder / split).iterdir() if p.is_file()}
            if actual != expected:
                raise ValueError('Missing or extra ' + folder)
        records = []
        for e in entries:
            name = e['image']
            stem = Path(name).stem
            image = root / 'images' / split / name
            label = root / 'labels' / split / (stem + '.txt')
            side = root / 'excluded_annotations' / split / (stem + '.json')
            if image_mode == 'rehash' and sha(image) != e['image_sha256']:
                raise ValueError('Image bytes changed')
            if sha(label) != e['label_sha256'] or sha(side) != e['sidecar_sha256']:
                raise ValueError('Label/sidecar byte provenance differs')
            label_sha, count = label_identity(label, len(m['class_names']))
            if count != e['targets']:
                raise ValueError('Label count differs')
            side_sem = hashlib.sha256(canonical(load(side))).hexdigest()
            records.append({'image': name, 'image_sha256': e['image_sha256'], 'source_annotation_sha256': e['source_annotation_sha256'], 'label_float32_ordered_sha256': label_sha, 'sidecar_semantic_sha256': side_sem, 'targets': count})
            images += 1
            targets += count
        policy['splits'][split] = {'entries': records, 'counts': m['splits'][split]['counts'], 'class_counts': m['splits'][split]['class_counts']}
    return {'identity_sha256': hashlib.sha256(canonical(policy)).hexdigest(), 'images': images, 'targets': targets, 'image_mode': image_mode, 'label_sidecar_bytes_rehashed': True, 'image_bytes_rehashed': image_mode == 'rehash', 'raw_source_annotations_rehashed': False, 'float_dtype': 'little-endian float32', 'numpy_version': np.__version__}
