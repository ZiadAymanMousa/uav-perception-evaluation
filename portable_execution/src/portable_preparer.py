# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Standalone UVD-1 data preparation and strict prospective dataset binding.

This component does not train a model, run inference, score AP or modify a legacy
run. Expected source/task identities come from a separately reviewed config.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

from PIL import Image
import yaml

from audit_dataset import audit, digest
from prepare_full_dataset import KNOWN, curate_lines, deduplicate_labels
from prepare_training_pilot import NAMES, convert_rows
from semantic_dataset import identity

SCHEMA = 'uvd1-portable-reviewed-task-v1'
BINDING_SCHEMA = 'uvd1-prepared-data-binding-v1'
POLICY = ('score=1 classes 1..10 -> 0..9; known three exact zero-height exceptions; '
          'clip valid scored boxes if required; remove float32 duplicate labels; '
          'all exclusions retained; original files untouched')
SPLITS = ('train', 'val')
ROOT_FILES = {'manifest.json', 'dataset.yaml', 'source_audit.json',
              'binding.json', 'preparation_record.json'}
HELPERS = ('portable_preparer.py', 'audit_dataset.py', 'prepare_full_dataset.py',
           'prepare_training_pilot.py', 'semantic_dataset.py')
BINDING_FIELDS = ('schema', 'scope', 'config_sha256', 'dataset_manifest_sha256',
                  'source_audit_sha256', 'dataset_yaml_sha256',
                  'source_manifest_sha256', 'semantic_identity_sha256',
                  'image_count', 'target_count', 'implementation_sha256')


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def implementation_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in HELPERS}


def load_reviewed_config(path, *, allow_synthetic=False):
    config = read_json(path)
    require(config.get('schema') == SCHEMA, 'Unknown reviewed-task schema')
    scope = config.get('scope')
    require(scope == 'visdrone-uvd1-reviewed' or
            (allow_synthetic and scope == 'synthetic-control-only'), 'Unsupported task scope')
    require(config.get('protocol') == 'UVD-1', 'Unsupported curation protocol')
    require(config.get('policy') == POLICY, 'Unsupported curation policy')
    require(config.get('class_names') == NAMES, 'Unexpected class order')
    require(config.get('semantic_schema') == 'converted-task-semantic-identity-prototype-v1',
            'Unknown semantic schema')
    require(set(config['splits']) == set(SPLITS), 'Unexpected split set')
    for key in ('expected_source_manifest_sha256', 'expected_semantic_identity_sha256'):
        value = config[key]
        require(isinstance(value, str) and len(value) == 64 and
                all(c in '0123456789abcdef' for c in value), 'Invalid expected identity')
    if scope == 'visdrone-uvd1-reviewed':
        expected = [{'split': s, 'file': name, 'line': number, 'raw': raw,
                     'reason': 'known_zero_height_original'}
                    for (s, name, number), raw in sorted(KNOWN.items())]
        require(config['expected_exceptions'] == expected, 'Reviewed exact exceptions changed')
        require(config['splits']['train']['counts']['images'] == 6471 and
                config['splits']['val']['counts']['images'] == 548, 'Reviewed split sizes changed')
    return config


def _link(path):
    return path.is_symlink() or getattr(path, 'is_junction', lambda: False)()


def _source_identity(report):
    # This intentionally matches audit_dataset.audit's relative-content identity.
    manifests = {name: item['manifest'] for name, item in report['splits'].items()}
    return hashlib.sha256(json.dumps(manifests, sort_keys=True).encode()).hexdigest()


def validate_audit(report, config):
    require(set(report['splits']) == set(SPLITS), 'Audit split set differs')
    source_identity = _source_identity(report)
    require(report['manifest_sha256'] == source_identity == config['expected_source_manifest_sha256'],
            'Original source content differs from the reviewed task')
    expected_issues = {(e['split'], e['file'] + ':' + str(e['line']),
                        'nonpositive box dimensions') for e in config['expected_exceptions']}
    actual_issues = {(x['split'], x['path'], x['detail']) for x in report['issues']}
    require(actual_issues == expected_issues and len(report['issues']) == len(expected_issues),
            'Raw audit differs from the exact reviewed exceptions')
    require(report['error_count'] == len(expected_issues), 'Incorrect audit error accounting')
    require(not report['cross_split_duplicates'], 'Cross-split byte-identical images')
    for split in SPLITS:
        entries = report['splits'][split]['manifest']
        require(len(entries) == config['splits'][split]['counts']['images'], 'Unexpected source image count')
        names = [Path(e['image']).name for e in entries]
        require(names == sorted(names) and len(set(names)) == len(names), 'Invalid source image selection')
    return source_identity


def _validate_layout(root, manifest):
    require(not _link(root), 'Dataset root cannot be a link or junction')
    require(root.is_dir(), 'Prepared dataset is missing')
    expected_dirs = {'images', 'labels', 'excluded_annotations'}
    require({p.name for p in root.iterdir() if p.is_dir()} == expected_dirs, 'Extra/missing root directory')
    require({p.name for p in root.iterdir() if p.is_file()} <= ROOT_FILES, 'Unexpected root metadata file')
    for path in root.rglob('*'):
        require(not _link(path), 'Links/junctions are not supported in prepared data')
        require(path.resolve().is_relative_to(root.resolve()), 'Prepared path escapes dataset root')
    for kind in expected_dirs:
        base = root / kind
        require({p.name for p in base.iterdir()} == set(SPLITS), 'Extra/missing split folder')
        for split in SPLITS:
            folder = base / split
            require(folder.is_dir() and all(p.is_file() for p in folder.iterdir()),
                    'Unexpected nested data directory')
    for split in SPLITS:
        names = [entry['image'] for entry in manifest['splits'][split]['entries']]
        require(len(names) == len(set(names)), 'Duplicate image identifiers')
        require(all(isinstance(name, str) and Path(name).name == name and
                    not any(c in name for c in ('/', '\\', ':')) for name in names), 'Unsafe image identifier')


def _validate_yaml(root, config):
    path = root / 'dataset.yaml'
    value = yaml.safe_load(path.read_text(encoding='utf-8'))
    require(isinstance(value, dict) and set(value) == {'path', 'train', 'val', 'names'},
            'Dataset YAML contains unsupported fields')
    require(isinstance(value['path'], str) and Path(value['path']).is_absolute() and
            Path(value['path']).resolve() == root.resolve(), 'Dataset YAML redirects its root')
    require(value['train'] == 'images/train' and value['val'] == 'images/val',
            'Dataset YAML redirects a split')
    require(value['names'] == dict(enumerate(config['class_names'])), 'Dataset YAML class order differs')


def verify_prepared_dataset(data_root, config_path, *, expected_binding=None,
                            source_roots=None, allow_synthetic=False):
    """Rehash all converted images/labels/sidecars; return actual binding.

    expected_binding is an independently retained record/dict. If provided,
    byte-level provenance and implementation hashes must also agree. Without it,
    semantic task verification remains possible after fresh preparation.
    source_roots optionally triggers a new raw image/annotation audit; otherwise
    the retained raw-audit content is validated but originals are not re-read.
    """
    started = time.perf_counter()
    root = Path(data_root).absolute()
    config_path = Path(config_path).resolve()
    config = load_reviewed_config(config_path, allow_synthetic=allow_synthetic)
    manifest = read_json(root / 'manifest.json')
    require(manifest['status'] == 'complete', 'Dataset conversion is incomplete')
    for key in ('protocol', 'class_names', 'policy'):
        require(manifest[key] == config[key], 'Prepared task policy differs: ' + key)
    require(manifest['exceptions'] == config['expected_exceptions'], 'Exception policy differs')
    for split in SPLITS:
        for key in ('counts', 'class_counts'):
            require(manifest['splits'][split][key] == config['splits'][split][key],
                    'Prepared split accounting differs: ' + split + '/' + key)
    _validate_layout(root, manifest)
    _validate_yaml(root, config)
    audit_path = root / 'source_audit.json'
    source_audit = read_json(audit_path)
    require(digest(audit_path) == manifest['audit_file_sha256'], 'Retained source audit bytes changed')
    source_identity = validate_audit(source_audit, config)
    require(manifest['source_manifest_sha256'] == source_identity, 'Manifest source identity differs')
    for split in SPLITS:
        raw = source_audit['splits'][split]['manifest']
        entries = manifest['splits'][split]['entries']
        require([Path(e['image']).name for e in raw] == [e['image'] for e in entries],
                'Original and converted selections differ')
        for original, converted in zip(raw, entries):
            require(original['image_sha256'] == converted['image_sha256'] and
                    original['annotation_sha256'] == converted['source_annotation_sha256'],
                    'Original and converted source provenance differs')
    signature = identity(root, image_mode='rehash')
    require(signature['identity_sha256'] == config['expected_semantic_identity_sha256'],
            'Converted semantic task identity differs')
    if source_roots is not None:
        require(set(source_roots) == set(SPLITS), 'Explicit train and val source roots required')
        fresh = audit({s: Path(source_roots[s]) for s in SPLITS})
        validate_audit(fresh, config)
        require(_source_identity(fresh) == source_identity, 'Fresh raw source differs')
    result = {'schema': BINDING_SCHEMA, 'scope': config['scope'], 'verified_utc': now(),
              'config_sha256': digest(config_path), 'dataset_manifest_sha256': digest(root / 'manifest.json'),
              'source_audit_sha256': digest(audit_path), 'dataset_yaml_sha256': digest(root / 'dataset.yaml'),
              'source_manifest_sha256': source_identity,
              'semantic_identity_sha256': signature['identity_sha256'],
              'image_count': signature['images'], 'target_count': signature['targets'],
              'implementation_sha256': implementation_hashes(),
              'converted_images_labels_sidecars_fully_rehashed': True,
              'raw_sources_reaudited_now': source_roots is not None,
              'expected_binding_checked': expected_binding is not None,
              'verification_seconds': time.perf_counter() - started,
              'limits': ['No training/inference/AP execution or clean environment installation',
                         'Semantic equivalence does not relabel historical checkpoint provenance',
                         'Retained raw audit is not a new read of originals unless raw_sources_reaudited_now is true']}
    if expected_binding is not None:
        expected = read_json(expected_binding) if isinstance(expected_binding, (str, Path)) else expected_binding
        require(set(BINDING_FIELDS) <= set(expected), 'Incomplete or unknown saved binding')
        for key in BINDING_FIELDS:
            require(expected[key] == result[key], 'Saved binding is forged/stale or inputs changed: ' + key)
    return result


def prepare_dataset(source_roots, output_root, config_path, *, allow_synthetic=False):
    """Audit original files and create a new preparation; never overwrite."""
    config_path = Path(config_path).resolve()
    config = load_reviewed_config(config_path, allow_synthetic=allow_synthetic)
    require(set(source_roots) == set(SPLITS), 'Explicit train and val source roots required')
    roots = {s: Path(source_roots[s]).resolve() for s in SPLITS}
    dest = Path(output_root).absolute()
    require(not dest.exists(), 'Output exists; preserve rather than overwrite')
    require(not any(dest.resolve().is_relative_to(root) or root.is_relative_to(dest.resolve())
                    for root in roots.values()), 'Output and raw source roots must not overlap')
    dest.mkdir(parents=True)
    rec = {'schema': 'uvd1-portable-preparation-run-v1', 'status': 'running',
           'phase': 'raw_source_audit', 'started_utc': now(),
           'config_sha256': digest(config_path), 'implementation_sha256': implementation_hashes(),
           'raw_source_roots': {s: str(p) for s, p in roots.items()},
           'output_root': str(dest.resolve()),
           'model_execution_performed': False, 'scope': config['scope']}
    write_json(dest / 'preparation_record.json', rec)
    started = time.perf_counter()
    try:
        report = audit(roots)
        write_json(dest / 'source_audit.json', report)
        validate_audit(report, config)
        rec.update(phase='converting', audit_seconds=time.perf_counter() - started,
                   raw_source_pairs_audited=sum(len(report['splits'][s]['manifest']) for s in SPLITS))
        write_json(dest / 'preparation_record.json', rec)
        m = {'protocol': config['protocol'], 'status': 'converting', 'created_utc': now(),
             'class_names': config['class_names'], 'audit_file_sha256': digest(dest / 'source_audit.json'),
             'source_manifest_sha256': report['manifest_sha256'],
             'converter_sha256': digest(Path(__file__)),
             'row_converter_sha256': digest(Path(__file__).with_name('prepare_training_pilot.py')),
             'implementation_sha256': implementation_hashes(), 'policy': config['policy'],
             'limitations': config['limitations'], 'exceptions': [], 'splits': {}}
        conversion_started = time.perf_counter()
        for split in SPLITS:
            for kind in ('images', 'labels', 'excluded_annotations'):
                (dest / kind / split).mkdir(parents=True)
            entries, counts, classes = [], Counter(), Counter()
            for index, item in enumerate(report['splits'][split]['manifest'], 1):
                image = roots[split] / item['image']
                ann = roots[split] / 'annotations' / (image.stem + '.txt')
                require(digest(ann) == item['annotation_sha256'], 'Original annotation changed after audit')
                with Image.open(image) as value:
                    width, height = value.size
                require((width, height) == (item['width'], item['height']), 'Original image dimensions changed')
                rows, exceptions = curate_lines(ann.read_text(encoding='utf-8-sig').splitlines(), split, ann.name)
                labels, excluded, clipped = convert_rows(rows, width, height)
                labels, duplicates = deduplicate_labels(labels)
                m['exceptions'].extend(exceptions)
                label = dest / 'labels' / split / ann.name
                label.write_text(''.join(f'{row[0]} ' + ' '.join(f'{v:.9f}' for v in row[1:]) + '\n'
                                         for row in labels), encoding='utf-8')
                side = dest / 'excluded_annotations' / split / (image.stem + '.json')
                write_json(side, {'not_scored_targets': excluded, 'known_exceptions': exceptions,
                                  'duplicates': duplicates})
                target = dest / 'images' / split / image.name
                shutil.copyfile(image, target)
                require(digest(target) == item['image_sha256'], 'Copied image differs from audited source')
                entries.append({'image': image.name, 'image_sha256': item['image_sha256'],
                                'source_annotation_sha256': item['annotation_sha256'],
                                'label_sha256': digest(label), 'sidecar_sha256': digest(side), 'targets': len(labels)})
                classes.update(str(row[0]) for row in labels)
                counts.update(images=1, targets=len(labels), excluded=len(excluded),
                              known_exceptions=len(exceptions), duplicates=len(duplicates), clipped=clipped)
                if index % 500 == 0:
                    print(f'PREPARE {split}: {index} images', flush=True)
            m['splits'][split] = {'counts': dict(counts), 'class_counts': dict(classes), 'entries': entries}
        m.update(status='complete', seconds=time.perf_counter() - conversion_started)
        write_json(dest / 'manifest.json', m)
        (dest / 'dataset.yaml').write_text(yaml.safe_dump({'path': dest.resolve().as_posix(),
            'train': 'images/train', 'val': 'images/val', 'names': dict(enumerate(config['class_names']))},
            sort_keys=False), encoding='utf-8')
        rec.update(phase='full_converted_verification', conversion_seconds=m['seconds'])
        write_json(dest / 'preparation_record.json', rec)
        binding = verify_prepared_dataset(dest, config_path, allow_synthetic=allow_synthetic)
        write_json(dest / 'binding.json', binding)
        rec.update(status='completed', phase='completed', completed_utc=now(),
                   binding_sha256=digest(dest / 'binding.json'),
                   dataset_manifest_sha256=binding['dataset_manifest_sha256'],
                   semantic_identity_sha256=binding['semantic_identity_sha256'],
                   seconds=time.perf_counter() - started)
        write_json(dest / 'preparation_record.json', rec)
        return binding
    except BaseException as exc:
        rec.update(status='failed', phase=rec['phase'], failed_utc=now(),
                   error_type=type(exc).__name__, error=str(exc), seconds=time.perf_counter() - started)
        write_json(dest / 'preparation_record.json', rec)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Fresh source audit and conversion; output must not exist')
    prepare.add_argument('--train-root', type=Path, required=True)
    prepare.add_argument('--val-root', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--config', type=Path, required=True)
    verify = commands.add_parser('verify', help='Full converted-data rehash and saved-binding check')
    verify.add_argument('--data-root', type=Path, required=True)
    verify.add_argument('--config', type=Path, required=True)
    verify.add_argument('--binding', type=Path, help='Default: DATA_ROOT/binding.json')
    verify.add_argument('--train-root', type=Path)
    verify.add_argument('--val-root', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare_dataset({'train': args.train_root, 'val': args.val_root}, args.output, args.config)
    else:
        if (args.train_root is None) != (args.val_root is None):
            parser.error('Supply both original source roots, or neither')
        sources = None if args.train_root is None else {'train': args.train_root, 'val': args.val_root}
        result = verify_prepared_dataset(args.data_root, args.config,
                    expected_binding=args.binding or args.data_root / 'binding.json', source_roots=sources)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
