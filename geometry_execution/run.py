# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Numeric-only adapter with explicit inputs and an execution gate."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
sys.dont_write_bytecode = True
import cv2
import numpy as np
from PIL import Image
if __package__:
    from . import core
else:
    import core

ROLES = {'posthoc_target', 'known_exact_byte_control', 'known_transform_control',
         'visually_different_natural_control_not_verified_geography'}
TRANSFORMS = {'scale_crop', 'rotation', 'perspective'}
FIELDS = {'case_id', 'role', 'left', 'right', 'transform', 'left_sha256', 'right_sha256'}
SETTINGS = {
    'long_edge_max': 1280, 'upscale': False,
    'rgb_decode': 'Pillow RGB without EXIF transpose',
    'resize': 'INTER_AREA; rounded dimensions, actual independent sx/sy',
    'coordinates': 'pixel centres: x_resized=sx*(x_original+0.5)-0.5; same for y',
    'sift': {'nfeatures': 4000, 'nOctaveLayers': 3, 'contrastThreshold': .04,
             'edgeThreshold': 10., 'sigma': 1.6},
    'matcher': 'exact BF L2, two nearest neighbours in both directions',
    'ratio_strict_less_than': .75, 'mutual': True,
    'ransac': {'threshold_resized_right_pixels': 3., 'max_iterations': 2000,
               'confidence': .995, 'seed_each_case': 20261027},
    'opencv_threads': 1, 'opencl': False, 'inverse_condition_max': 1e12,
    'known_transform': {'scale_crop': 'centred 10% floor crop to original dimensions',
                        'rotation': '8 degrees about pixel-centre image centre',
                        'perspective': 'fixed corner fractions (.04,.03),(.94,.02),(.97,.94),(.03,.97)'},
    'known_grid': '9x9 interior source fractions .1 to .9, expected output-mask overlap',
    'no_binary_truth_threshold': True,
}
SUMMARY_FIELDS = ['case_id', 'role', 'status', 'mutual_matches_n', 'ransac_inliers_n',
                  'ransac_unique_inlier_xy_pairs_n', 'inlier_fraction_denominator_mutual',
                  'inlier_forward_median_resized_right_pixels',
                  'all_mutual_forward_p95_resized_right_pixels']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def numeric_json(value):
    """Nonfinite numeric values become null, never JSON NaN/Infinity."""
    if isinstance(value, dict):
        return {k: numeric_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [numeric_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError('Unsupported output value type')


def relative_source(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Sources must be nonempty relative POSIX paths')
    parts = value.split('/')
    if (any(p in ('', '.', '..') for p in parts) or ':' in value or
            PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive):
        raise ValueError('Absolute, drive, empty or traversing sources are forbidden')
    return value


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON keys are forbidden')
        result[key] = value
    return result


def load_cases(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            if (reader.fieldnames is None or set(reader.fieldnames) - FIELDS or
                    len(reader.fieldnames) != len(set(reader.fieldnames))):
                raise ValueError('Invalid CSV case columns')
            raw = list(reader)
    elif path.suffix.lower() == '.json':
        raw = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_json_object)
        if isinstance(raw, dict):
            if set(raw) - {'schema', 'cases', 'selection'} or raw.get('schema') != 1:
                raise ValueError('Invalid case manifest schema')
            raw = raw.get('cases')
    else:
        raise ValueError('Case manifest must be CSV or JSON')
    if not isinstance(raw, list) or not raw:
        raise ValueError('A nonempty case list is required')
    result, seen = [], set()
    for row in raw:
        if not isinstance(row, dict) or set(row) - FIELDS:
            raise ValueError('Invalid case fields')
        cid, role = row.get('case_id'), row.get('role')
        if not isinstance(cid, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', cid) or cid in seen:
            raise ValueError('Case identifiers must be unique simple names')
        if role not in ROLES:
            raise ValueError('Unsupported descriptive case role')
        left = relative_source(row.get('left'))
        transform = row.get('transform') or None
        if transform is not None:
            if transform not in TRANSFORMS or role != 'known_transform_control':
                raise ValueError('Only fixed known-transform controls can generate a right image')
            if row.get('right') not in (None, '', 'generated:' + transform):
                raise ValueError('Transformed cases must not also name a right source')
            right = 'generated:' + transform
        else:
            if role == 'known_transform_control':
                raise ValueError('Transform control requires a fixed transform')
            right = relative_source(row.get('right'))
        out = {'case_id': cid, 'role': role, 'left': left, 'right': right, 'transform': transform}
        for key in ('left_sha256', 'right_sha256'):
            expected = row.get(key) or None
            if expected is not None and (not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected)):
                raise ValueError('Expected hashes must be lower-case SHA256')
            if key == 'right_sha256' and expected is not None and transform is not None:
                raise ValueError('Generated right images have no source-file hash')
            if expected is not None:
                out[key] = expected
        seen.add(cid)
        result.append(out)
    return result


def preflight(data_root, case_file, output):
    root = Path(data_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Data root must be a directory')
    output = Path(output)
    if os.path.lexists(output):
        raise ValueError('Output must not already exist, including a symlink')
    output = output.resolve(strict=False)
    if output == root or root in output.parents:
        raise ValueError('Output must be outside the data root')
    if not output.parent.is_dir():
        raise ValueError('Output parent must already exist')
    cases = load_cases(case_file)
    paths, hashes = {}, {}
    for case in cases:
        for side in ('left', 'right'):
            if side == 'right' and case['transform'] is not None:
                continue
            name = case[side]
            source = (root / name).resolve(strict=True)
            if root not in source.parents or not source.is_file():
                raise ValueError('Source escapes the data root or is not a file')
            if name not in hashes:
                paths[name], hashes[name] = source, sha256(source)
            if case.get(side + '_sha256') not in (None, hashes[name]):
                raise ValueError('Selected source hash does not match expected value')
        if case['role'] == 'known_exact_byte_control' and hashes[case['left']] != hashes[case['right']]:
            raise ValueError('Known exact-byte control sources differ')
    return cases, paths, hashes, output


def public_result(result):
    # No per-feature rows, descriptor arrays, RANSAC mask or path-bearing cv2 errors.
    return numeric_json({k: v for k, v in result.items()
                         if k not in {'opencv_error', 'ransac_mask'}})


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(numeric_json(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def run(data_root, case_file, output, *, execute=False):
    cases, paths, before, out = preflight(data_root, case_file, output)
    if not execute:
        return {'state': 'validated_no_execution', 'cases_n': len(cases), 'sources_n': len(paths)}
    # Everything above, including every selected source, is validated before mkdir.
    out.mkdir(exist_ok=False)
    core.configure_cpu()
    features = {}
    results = []
    for case in cases:
        try:
            for side in ('left', 'right'):
                if side == 'right' and case['transform'] is not None:
                    continue
                name = case[side]
                if name not in features:
                    with Image.open(paths[name]) as image:
                        rgb = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
                    features[name] = (rgb, core.extract_features(rgb))
            left_rgb, left = features[case['left']]
            truth = None
            if case['transform'] is not None:
                generated, truth, mask = core.create_transform(left_rgb, case['transform'])
                right = core.extract_features(generated)
            else:
                right = features[case['right']][1]
            result = public_result(core.analyse_pair(left, right)[0])
            result['images'] = {
                side: {'original_shape_hw': feature['original_shape_hw'],
                       'resized_shape_hw': feature['resized_shape_hw'],
                       'T_original_to_resized': feature['T_original_to_resized'].tolist()}
                for side, feature in (('left', left), ('right', right))}
            if truth is not None:
                grid, _ = core.synthetic_grid_check(result['homography_left_to_right_original'], truth,
                                                     left['original_shape_hw'], mask)
                result['known_transform'] = {'kind': case['transform'],
                                             'true_homography_left_to_right_original': truth.tolist(),
                                             'grid_summary': grid}
            result.update({k: case[k] for k in ('case_id', 'role', 'left', 'right', 'transform')})
        except Exception as exc:
            result = {'case_id': case['case_id'], 'role': case['role'],
                      'status': 'case_execution_error', 'error_type': type(exc).__name__,
                      'truth_classification': None, 'no_binary_truth_threshold': True}
        results.append(numeric_json(result))
    after = {}
    for name, path in paths.items():
        try:
            after[name] = sha256(path)
        except OSError:
            after[name] = None
    changed = [name for name in before if before[name] != after[name]]
    errors = sum(r['status'] in {'case_execution_error', 'opencv_estimation_error', 'homography_nonfinite'}
                 for r in results)
    state = 'completed' if not changed and not errors else 'completed_with_errors_or_changed_inputs'
    write_json(out / 'input_manifest.json', {'schema': 1, 'cases': cases,
               'selected_sources': [{'path': name, 'sha256_before': before[name],
                                     'sha256_after': after[name]} for name in sorted(paths)],
               'selected_inputs_unchanged': not changed})
    write_json(out / 'settings.json', {'schema': 1, 'fixed_method': SETTINGS,
               'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                            'opencv': cv2.__version__, 'pillow': Image.__version__},
               'source_sha256': {name: sha256(Path(__file__).with_name(name))
                                 for name in ('core.py', 'run.py')}})
    write_json(out / 'results.json', {'schema': 1, 'state': state, 'cases_n': len(cases),
               'case_errors_n': errors, 'changed_input_paths': changed, 'results': results,
               'scope': 'Fixed descriptive geometry; no geography, prevalence or detector-score claim'})
    with (out / 'summary.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for result in results:
            stats = result.get('residuals_ransac_inliers', {}).get('resized', {}).get('forward', {})
            all_stats = result.get('residuals_all_mutual', {}).get('resized', {}).get('forward', {})
            row = {k: result.get(k) for k in SUMMARY_FIELDS[:6]}
            row.update(inlier_fraction_denominator_mutual=result.get('ransac_inlier_fraction_denominator_mutual'),
                       inlier_forward_median_resized_right_pixels=stats.get('median'),
                       all_mutual_forward_p95_resized_right_pixels=all_stats.get('p95_linear'))
            writer.writerow(row)
    write_json(out / 'output_manifest.json', {'schema': 1, 'state': state,
               'files': {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
               'note': 'This manifest excludes its own hash; outputs contain no images or feature dumps'})
    return {'state': state, 'cases_n': len(cases), 'case_errors_n': errors,
            'selected_inputs_unchanged': not changed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--cases', required=True, help='Relative-source case CSV or JSON')
    parser.add_argument('--output', required=True, help='New directory outside data root')
    parser.add_argument('--run', action='store_true', help='Explicitly allow feature extraction and fitting')
    args = parser.parse_args(argv)
    try:
        result = run(args.data_root, args.cases, args.output, execute=args.run)
    except Exception as exc:
        # Do not emit absolute source paths, decoder exceptions or tracebacks.
        print('Input/execution failure: ' + type(exc).__name__, file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0 if result['state'] in ('completed', 'validated_no_execution') else 2


if __name__ == '__main__':
    raise SystemExit(main())
