# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Strict all-image conversion with path-independent release-time identity."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil

from common import (require, digest, object_sha, read_json, write_json, now,
                    source_hashes, snapshot_sources, verify_sources, validate_helpers)
from audit_dataset import parse_annotation
from prepare_training_pilot import convert_rows, NAMES
from prepare_full_dataset import deduplicate_labels
from score_converted_predictions import label_rows_as_xyxy

IDENTITY_KEYS = ('image', 'image_sha256', 'source_annotation_sha256', 'width',
                 'height', 'label_sha256', 'targets')

def semantic_identity(entries):
    return object_sha({'schema': 'uvd1-testdev-content-v1', 'class_names': NAMES,
                       'entries': [{k: e[k] for k in IDENTITY_KEYS} for e in entries]})

def file_members(folder):
    folder = Path(folder)
    require(folder.is_dir() and not folder.is_symlink(), 'Missing or linked input directory: ' + str(folder))
    values = list(folder.iterdir())
    require(all(p.is_file() and not p.is_symlink() for p in values), 'Unexpected nested/linked entry in ' + str(folder))
    return {p.name for p in values}

def encode_labels(labels):
    return ''.join(f'{r[0]} ' + ' '.join(f'{v:.9f}' for v in r[1:]) + '\n' for r in labels).encode('utf-8')

def convert_entry(image, annotation):
    from PIL import Image
    image, annotation = Path(image), Path(annotation)
    with Image.open(image) as value:
        width, height = value.size
        value.verify()
    raw = annotation.read_bytes()
    # Every nonempty original row is parsed, including native ignored rows.
    source_rows = []
    for line_number, row in enumerate(csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline='')), 1):
        if not row or not any(v.strip() for v in row):
            continue
        parsed = parse_annotation(row)
        source_rows.append((line_number, row, parsed))
    labels, excluded, clipped = convert_rows([v[1] for v in source_rows], width, height)
    labels, duplicates = deduplicate_labels(labels)
    require(len(labels) + len(excluded) + len(duplicates) == len(source_rows), 'Row accounting mismatch')
    ignored = [{'line_number': line, 'original_fields': fields, 'annotation': parsed,
                'reason': 'not_scored_target'} for line, fields, parsed in source_rows
               if not (1 <= parsed['category'] <= 10 and parsed['score'] == 1)]
    label_bytes = encode_labels(labels)
    sidecar = {'source_annotation_sha256': hashlib.sha256(raw).hexdigest(),
               'not_scored_targets': ignored, 'duplicates': duplicates,
               'known_exceptions': [], 'raw_rows': len(source_rows)}
    entry = {'image': image.name, 'image_sha256': digest(image),
             'source_annotation_sha256': sidecar['source_annotation_sha256'],
             'width': width, 'height': height,
             'label_sha256': hashlib.sha256(label_bytes).hexdigest(), 'targets': len(labels)}
    counts = dict(images=1, raw_rows=len(source_rows), targets=len(labels),
                  excluded=len(ignored), duplicates=len(duplicates), clipped=clipped, malformed_exceptions=0)
    return entry, label_bytes, sidecar, counts

def scan_raw(raw_root, task, *, destination=None):
    validate_helpers(task)
    raw_root = Path(raw_root)
    images = sorted(file_members(raw_root/'images'))
    require(all(Path(n).suffix.lower() == '.jpg' for n in images), 'Only reviewed JPEG image members accepted')
    require(len(images) == task['expected_counts']['images'], 'Raw split image count differs')
    require(len({Path(n).stem for n in images}) == len(images), 'Duplicate raw image stems')
    expected_annotations = {Path(n).stem + '.txt' for n in images}
    require(file_members(raw_root/'annotations') == expected_annotations, 'Raw image/annotation membership differs')
    entries, total, classes = [], Counter(), Counter()
    for name in images:
        ann_name = Path(name).stem + '.txt'
        entry, labels, sidecar, counts = convert_entry(raw_root/'images'/name, raw_root/'annotations'/ann_name)
        entry['sidecar_content_sha256'] = object_sha(sidecar)
        entry['sidecar_sha256'] = hashlib.sha256((json.dumps(sidecar,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')).hexdigest()
        total.update(counts)
        for line in labels.decode().splitlines():
            classes[line.split()[0]] += 1
        entries.append(entry)
        if destination is not None:
            target = Path(destination)
            shutil.copyfile(raw_root/'images'/name, target/'images/testdev'/name)
            shutil.copyfile(raw_root/'annotations'/ann_name, target/'source_annotations/testdev'/ann_name)
            (target/'labels/testdev'/ann_name).write_bytes(labels)
            write_json(target/'excluded_annotations/testdev'/(Path(name).stem + '.json'), sidecar)
    require(dict(total) == task['expected_counts'], 'Strict split/annotation conversion totals differ')
    require(semantic_identity(entries) == task['expected_identity_sha256'], 'Reviewed test-dev content identity differs')
    return entries, dict(total), dict(classes)

def prepare(raw_root, prepared, config_path, task):
    prepared = Path(prepared)
    require(not prepared.exists(), 'Prepared destination exists; preserve it')
    require(not prepared.resolve().is_relative_to(Path(raw_root).resolve()), 'Preparation must be outside raw input tree')
    starting_sources, starting_config = source_hashes(), digest(Path(config_path))
    # Read-only full preflight before creating output, then convert with a second complete check.
    expected, _, _ = scan_raw(raw_root, task)
    prepared.mkdir(parents=True)
    for folder in ('images/testdev', 'labels/testdev', 'excluded_annotations/testdev', 'source_annotations/testdev'):
        (prepared/folder).mkdir(parents=True)
    record = {'schema': 'uvd1-testdev-preparation-v1', 'status': 'converting', 'started_utc': now()}
    write_json(prepared/'manifest.json', record)
    try:
        entries, counts, classes = scan_raw(raw_root, task, destination=prepared)
        require(entries == expected, 'Raw input changed during conversion')
        require(source_hashes() == starting_sources and digest(Path(config_path)) == starting_config, 'Source/config changed during conversion')
        sources = snapshot_sources(prepared/'source_snapshot')
        shutil.copyfile(config_path, prepared/'task_config.json')
        record.update(status='complete', config_sha256=digest(config_path), class_names=NAMES,
                      entries=entries, counts=counts, class_counts=classes,
                      content_identity_sha256=semantic_identity(entries), source_hashes=sources, finished_utc=now())
        write_json(prepared/'manifest.json', record)
        binding = {'schema': 'uvd1-testdev-binding-v1', 'manifest_sha256': digest(prepared/'manifest.json'),
                   'config_sha256': digest(config_path), 'content_identity_sha256': semantic_identity(entries),
                   'source_identity_sha256': object_sha(sources)}
        write_json(prepared/'binding.json', binding)
        verify(raw_root, prepared, config_path, task)
        return binding
    except BaseException as exc:
        record.update(status='failed', error=repr(exc), finished_utc=now())
        write_json(prepared/'manifest.json', record)
        raise

def verify(raw_root, prepared, config_path, task):
    prepared = Path(prepared)
    binding, manifest = read_json(prepared/'binding.json'), read_json(prepared/'manifest.json')
    require(binding['schema'] == 'uvd1-testdev-binding-v1' and manifest['schema'] == 'uvd1-testdev-preparation-v1', 'Unknown preparation schema')
    require(manifest['status'] == 'complete', 'Preparation is not complete')
    require(digest(prepared/'manifest.json') == binding['manifest_sha256'], 'Prepared manifest bytes changed')
    require(digest(config_path) == digest(prepared/'task_config.json') == binding['config_sha256'] == manifest['config_sha256'], 'Prepared task/config changed')
    require(manifest['class_names'] == NAMES == task['class_names'], 'Class names changed')
    verify_sources(prepared/'source_snapshot', manifest['source_hashes'])
    require(object_sha(manifest['source_hashes']) == binding['source_identity_sha256'], 'Preparation source binding differs')
    entries, counts, classes = scan_raw(raw_root, task)
    require(entries == manifest['entries'] and counts == manifest['counts'] and classes == manifest['class_counts'], 'Prepared data differs from complete raw reconstruction')
    require(semantic_identity(entries) == binding['content_identity_sha256'] == manifest['content_identity_sha256'], 'Prepared semantic binding differs')
    for folder, suffix in [('images/testdev', None), ('labels/testdev', '.txt'),
                            ('excluded_annotations/testdev', '.json'), ('source_annotations/testdev', '.txt')]:
        expected = {e['image'] if suffix is None else Path(e['image']).stem + suffix for e in entries}
        require(file_members(prepared/folder) == expected, 'Prepared membership changed: ' + folder)
    for entry in entries:
        stem = Path(entry['image']).stem
        require(digest(prepared/'images/testdev'/entry['image']) == entry['image_sha256'], 'Prepared image changed')
        require(digest(prepared/'source_annotations/testdev'/(stem+'.txt')) == entry['source_annotation_sha256'], 'Retained original annotation changed')
        label = prepared/'labels/testdev'/(stem+'.txt')
        require(digest(label) == entry['label_sha256'], 'Converted label bytes changed')
        sidecar_path = prepared/'excluded_annotations/testdev'/(stem+'.json')
        require(digest(sidecar_path) == entry['sidecar_sha256'], 'Exclusion sidecar bytes changed')
        sidecar = read_json(sidecar_path)
        require(object_sha(sidecar) == entry['sidecar_content_sha256'], 'Exclusion sidecar changed')
        gt = label_rows_as_xyxy(label, entry['width'], entry['height'])
        require(len(gt) == entry['targets'], 'Converted target count differs')
    return {'binding_sha256': digest(prepared/'binding.json'), 'manifest_sha256': digest(prepared/'manifest.json'),
            'config_sha256': digest(config_path), 'content_identity_sha256': semantic_identity(entries),
            'source_identity_sha256': object_sha(manifest['source_hashes']), 'counts': counts}
