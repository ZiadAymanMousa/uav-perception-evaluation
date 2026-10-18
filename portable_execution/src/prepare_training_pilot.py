# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Prepare a fixed small training/development pilot, preserving ignore metadata.

This is a YOLO-labelled pipeline pilot, not the native VisDrone scoring protocol.
Ignored regions remain in the image pixels and are NOT masked in the loss.
"""
from collections import Counter
import csv
import json
from pathlib import Path
import random
import shutil

from PIL import Image

from audit_dataset import digest, parse_annotation

PROJECT = Path(__file__).resolve().parents[1]
NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
         'tricycle', 'awning-tricycle', 'bus', 'motor']


def convert_rows(rows, width, height):
    """Keep scored classes 1..10, clip to image extent and retain all exclusions."""
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image extent')
    labels, excluded = [], []
    clipped = 0
    for row in rows:
        item = parse_annotation(row)
        if not (1 <= item['category'] <= 10 and item['score'] == 1):
            excluded.append({'reason': 'not_scored_target', 'annotation': item})
            continue
        raw = (item['left'], item['top'], item['left'] + item['width'], item['top'] + item['height'])
        left, top = max(0, raw[0]), max(0, raw[1])
        right, bottom = min(width, raw[2]), min(height, raw[3])
        if right <= left or bottom <= top:
            raise ValueError('Scored target has no image intersection')
        clipped += (left, top, right, bottom) != raw
        labels.append((item['category'] - 1, (left + right) / (2 * width),
                       (top + bottom) / (2 * height), (right - left) / width,
                       (bottom - top) / height))
    return labels, excluded, clipped


def prepare(train_count=128, val_count=32, seed=20261016):
    dest = PROJECT / 'data/yolo_training_pilot'
    if (dest / 'manifest.json').exists():
        raise RuntimeError('Pilot manifest exists; preserve this run rather than overwrite')
    record = {'purpose': 'small pipeline pilot; not a scientific baseline',
              'seed': seed, 'class_names': NAMES,
              'policy': 'score=1 classes 1..10 mapped to 0..9; clip scored boxes to image extent; exclude 0/11 and score=0 from labels; preserve exclusion sidecars; no ignore-loss masking',
              'val_usage': 'official validation exposed development subset; not held-out final evaluation',
              'test_dev': 'not accessed', 'splits': {}}
    for split, count in [('train', train_count), ('val', val_count)]:
        source = PROJECT / f'data/VisDrone2019-DET-{split}'
        images = sorted((source / 'images').glob('*.jpg'))
        chosen = sorted(random.Random(seed).sample(images, count))
        image_dest = dest / 'images' / split
        label_dest = dest / 'labels' / split
        sidecar_dest = dest / 'excluded_annotations' / split
        for folder in (image_dest, label_dest, sidecar_dest):
            folder.mkdir(parents=True, exist_ok=True)
        entries, classes = [], Counter()
        for image in chosen:
            annotation = source / 'annotations' / f'{image.stem}.txt'
            with Image.open(image) as value:
                width, height = value.size
            with annotation.open(newline='', encoding='utf-8') as stream:
                labels, excluded, clipped = convert_rows(csv.reader(stream), width, height)
            classes.update(str(row[0]) for row in labels)
            label = label_dest / f'{image.stem}.txt'
            label.write_text(''.join(f'{row[0]} ' + ' '.join(f'{v:.9f}' for v in row[1:]) + '\n' for row in labels), encoding='utf-8')
            (sidecar_dest / f'{image.stem}.json').write_text(json.dumps(excluded, indent=2) + '\n', encoding='utf-8')
            shutil.copyfile(image, image_dest / image.name)
            entries.append({'image': image.name, 'source_image_sha256': digest(image),
                            'annotation_sha256': digest(annotation), 'label_sha256': digest(label),
                            'width': width, 'height': height, 'target_boxes': len(labels),
                            'excluded_rows': len(excluded), 'clipped_boxes': clipped})
        record['splits'][split] = {'image_count': count, 'class_counts': dict(classes), 'entries': entries}
    hashes = [{x['source_image_sha256'] for x in record['splits'][s]['entries']} for s in ('train', 'val')]
    if hashes[0] & hashes[1]:
        raise RuntimeError('Byte-identical train/validation overlap')
    manifest = dest / 'manifest.json'
    manifest.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    data = {'path': str(dest.resolve()).replace('\\', '/'), 'train': 'images/train',
            'val': 'images/val', 'names': dict(enumerate(NAMES))}
    import yaml
    (dest / 'dataset.yaml').write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    print(json.dumps({'manifest_sha256': digest(manifest), 'splits': {
        k: {'images': v['image_count'], 'class_counts': v['class_counts'],
            'targets': sum(x['target_boxes'] for x in v['entries']),
            'excluded': sum(x['excluded_rows'] for x in v['entries']),
            'clipped': sum(x['clipped_boxes'] for x in v['entries'])}
        for k, v in record['splits'].items()}}, indent=2))


if __name__ == '__main__':
    prepare()
