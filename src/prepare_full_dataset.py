# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Strict, logged full-data YOLO conversion; original VisDrone files stay intact.

Protocol UVD-1: ten scored classes, known exact zero-height exceptions, retained
exclusion sidecars, float32 duplicate removal, no native ignore-loss masking.
"""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import yaml

from audit_dataset import digest
from prepare_training_pilot import convert_rows, NAMES

PROJECT = Path(__file__).resolve().parents[1]
KNOWN = {
    ('train', '0000293_03401_d_0000939.txt', 130): '1008,374,3,0,0,0,0,0',
    ('train', '9999985_00000_d_0000020.txt', 12): '611,158,4,0,1,4,0,0',
    ('train', '9999999_00590_d_0000267.txt', 89): '545,414,10,0,0,0,0,0',
}


def curate_lines(lines, split, filename):
    rows, exceptions = [], []
    for number, line in enumerate(lines, 1):
        raw = line.rstrip('\r\n')
        key = (split, filename, number)
        if key in KNOWN:
            if raw != KNOWN[key]:
                raise ValueError(f'Known exception changed: {key}')
            exceptions.append({'split': split, 'file': filename, 'line': number,
                               'raw': raw, 'reason': 'known_zero_height_original'})
        elif raw.strip():
            rows.append(next(csv.reader([raw])))
    return rows, exceptions


def deduplicate_labels(labels):
    unique, duplicates, seen = [], [], set()
    for index, row in enumerate(labels):
        key = tuple(np.asarray(row, dtype=np.float32).tolist())
        if key in seen:
            duplicates.append({'converted_row_index': index, 'label': row,
                               'reason': 'duplicate_in_framework_float32_labels'})
        else:
            seen.add(key)
            unique.append(row)
    return unique, duplicates


def prepare():
    started = time.perf_counter()
    audit_path = PROJECT / 'artifacts/train_val_audit_20261016.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    expected_issues = {(s, f'{name}:{number}', 'nonpositive box dimensions') for s,name,number in KNOWN}
    actual_issues = {(x['split'], x['path'], x['detail']) for x in audit['issues']}
    if actual_issues != expected_issues or audit['cross_split_duplicates']:
        raise RuntimeError('Audit differs from reviewed exceptions or contains cross-split duplicates')
    dest = PROJECT / 'data/yolo_full_uvd1'
    if dest.exists():
        raise RuntimeError('Destination exists; preserve/inspect rather than overwrite')
    if shutil.disk_usage(PROJECT).free < 5 * 1024**3:
        raise RuntimeError('At least 5 GiB free required before full conversion')
    dest.mkdir()
    record = {'protocol': 'UVD-1', 'status': 'converting',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'class_names': NAMES, 'audit_file_sha256': digest(audit_path),
              'source_manifest_sha256': audit['manifest_sha256'],
              'converter_sha256': digest(Path(__file__)),
              'row_converter_sha256': digest(PROJECT / 'src/prepare_training_pilot.py'),
              'policy': 'score=1 classes 1..10 -> 0..9; known three exact zero-height exceptions; clip valid scored boxes if required; remove float32 duplicate labels; all exclusions retained; original files untouched',
              'limitations': ['ignored pixels remain in loss and built-in metric',
                              'not native VisDrone evaluation',
                              'validation is exposed development; test-dev untouched',
                              'byte duplicate audit is not scene-independence verification'],
              'exceptions': [], 'splits': {}}
    for split, expected_count in [('train',6471), ('val',548)]:
        source = PROJECT / f'data/VisDrone2019-DET-{split}'
        items = audit['splits'][split]['manifest']
        if len(items) != expected_count:
            raise RuntimeError('Unexpected split size')
        image_dest, label_dest = dest / 'images' / split, dest / 'labels' / split
        sidecar_dest = dest / 'excluded_annotations' / split
        for folder in (image_dest,label_dest,sidecar_dest):
            folder.mkdir(parents=True)
        counts, classes, entries = Counter(), Counter(), []
        for index,item in enumerate(items,1):
            image = source / item['image']
            annotation = source / 'annotations' / f'{image.stem}.txt'
            if digest(annotation) != item['annotation_sha256']:
                raise RuntimeError(f'Source annotation changed: {annotation.name}')
            with Image.open(image) as value:
                width,height = value.size
            if (width,height) != (item['width'],item['height']):
                raise RuntimeError('Source image dimensions changed')
            rows, exceptions = curate_lines(annotation.read_text(encoding='utf-8-sig').splitlines(), split, annotation.name)
            labels, excluded, clipped = convert_rows(rows,width,height)
            labels, duplicates = deduplicate_labels(labels)
            record['exceptions'].extend(exceptions)
            label = label_dest / annotation.name
            label.write_text(''.join(f'{row[0]} ' + ' '.join(f'{x:.9f}' for x in row[1:]) + '\n' for row in labels), encoding='utf-8')
            sidecar = sidecar_dest / f'{image.stem}.json'
            sidecar.write_text(json.dumps({'not_scored_targets':excluded, 'known_exceptions':exceptions,
                                          'duplicates':duplicates},indent=2) + '\n',encoding='utf-8')
            target = image_dest / image.name
            shutil.copyfile(image,target)
            if digest(target) != item['image_sha256']:
                raise RuntimeError(f'Copied image differs from audited source: {image.name}')
            classes.update(str(row[0]) for row in labels)
            counts.update(images=1, targets=len(labels), excluded=len(excluded),
                          known_exceptions=len(exceptions), duplicates=len(duplicates), clipped=clipped)
            entries.append({'image':image.name, 'image_sha256':item['image_sha256'],
                            'source_annotation_sha256':item['annotation_sha256'],
                            'label_sha256':digest(label), 'sidecar_sha256':digest(sidecar),
                            'targets':len(labels)})
            if index % 500 == 0:
                print(f'{split}: converted and verified {index}/{len(items)} images',flush=True)
        record['splits'][split] = {'counts':dict(counts),'class_counts':dict(classes),'entries':entries}
    observed = {(x['split'],x['file'],x['line']) for x in record['exceptions']}
    if observed != set(KNOWN):
        raise RuntimeError('Not all exact exceptions were accounted for')
    record.update(status='complete',seconds=time.perf_counter()-started)
    manifest = dest / 'manifest.json'
    manifest.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    config = {'path':dest.resolve().as_posix(),'train':'images/train','val':'images/val',
              'names':dict(enumerate(NAMES))}
    (dest / 'dataset.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    print(json.dumps({'status':record['status'],'manifest_sha256':digest(manifest),
                      'seconds':record['seconds'],'splits':{k:v['counts'] for k,v in record['splits'].items()}},indent=2),flush=True)


if __name__ == '__main__':
    prepare()
