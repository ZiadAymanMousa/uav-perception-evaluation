# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Generated-only geometry and I/O controls; real SIFT/BF/RANSAC are blocked."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry_execution import core
from geometry_execution import run as adapter
import numpy as np
from PIL import Image


def fixture_rgb(h=36, w=50):
    y, x = np.indices((h, w))
    return np.stack([(x * 13 + y * 7) % 256, (x * 3 + y * 11) % 256,
                     ((x // 3 + y // 4) % 2) * 255], axis=2).astype(np.uint8)


def feature(points, T=None, shape=(40, 60)):
    points = np.asarray(points, np.float64).reshape(-1, 2)
    T = np.eye(3) if T is None else T
    return {'points_resized': points, 'points_original': core.project(points, np.linalg.inv(T)),
            'T_original_to_resized': T, 'original_shape_hw': list(shape),
            'resized_shape_hw': list(shape), 'keypoints': [{} for _ in points],
            'descriptors': np.zeros((len(points), 128), np.float32)}


def neighbours(n):
    return [{'query_index': i, 'first_train_index': i, 'first_distance': 0.,
             'ratio': 0., 'passes_strict_ratio_075': True} for i in range(n)]


class AlgorithmBlockedBase(unittest.TestCase):
    def setUp(self):
        # An accidental real feature/matching/fitting call fails this entire suite.
        for name in ('SIFT_create', 'BFMatcher', 'findHomography'):
            patch = mock.patch.object(core.cv2, name,
                                      side_effect=AssertionError('Real algorithm blocked in generated controls'))
            patch.start()
            self.addCleanup(patch.stop)



class GeometryControls(AlgorithmBlockedBase):
    def test_rounded_resize_pixel_centres_and_no_upscale(self):
        rgb = np.zeros((1301, 2001, 3), np.uint8)
        resized, T = core.resize_rgb(rgb)
        self.assertEqual(resized.shape, (832, 1280, 3))
        sx, sy = 1280 / 2001, 832 / 1301
        np.testing.assert_allclose(T, [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
        points = np.array([[0, 0], [2000, 1300], [123.4, 567.8]])
        np.testing.assert_allclose(core.project(core.project(points, T), np.linalg.inv(T)), points, atol=1e-12)
        small = fixture_rgb()
        copied, identity = core.resize_rgb(small)
        np.testing.assert_array_equal(copied, small)
        np.testing.assert_array_equal(identity, np.eye(3))
        self.assertIsNot(copied, small)
        with self.assertRaises(ValueError):
            core.resize_rgb(np.zeros((4, 4), np.uint8))

    def test_original_resized_homography_contract_and_units(self):
        TL = np.array([[.5, 0, -.25], [0, .4, -.3], [0, 0, 1.]])
        TR = np.array([[.7, 0, -.15], [0, .6, -.2], [0, 0, 1.]])
        original = np.array([[1.1, .05, 3], [.03, .9, -2], [.0001, -.0002, 1.]])
        resized = TR @ original @ np.linalg.inv(TL)
        np.testing.assert_allclose(np.linalg.inv(TR) @ resized @ TL, original, atol=1e-15)
        pts = np.array([[0., 0.], [20, 10], [100, 80]])
        np.testing.assert_allclose(core.project(core.project(pts, TL), resized),
                                   core.project(core.project(pts, original), TR), atol=1e-13)
        left_pts = np.array([[1, 1], [20, 1], [20, 20], [1, 20]], np.float64)
        right_pts = core.project(left_pts, resized)
        left, right = feature(left_pts, TL), feature(right_pts, TR)
        with mock.patch.object(core, 'knn_rows', side_effect=[neighbours(4), neighbours(4)]), \
             mock.patch.object(core.cv2, 'findHomography', return_value=(resized, np.ones((4, 1), np.uint8))):
            result = core.analyse_pair(left, right)[0]
        np.testing.assert_allclose(result['homography_left_to_right_original'], original, atol=1e-15)
        residuals, _, _ = core.calculate_residuals(np.array([[1., 2.]]), np.array([[3., 5.]]),
                                                  np.diag([2., 3., 1.]))
        self.assertAlmostEqual(residuals['forward'][0], np.sqrt(2))
        self.assertAlmostEqual(residuals['backward'][0], np.sqrt(.5**2 + (1/3)**2))
        self.assertAlmostEqual(residuals['symmetric'][0],
                               np.sqrt((2 + .5**2 + (1/3)**2) / 2))

    def test_invalid_projection_summary_and_serialization(self):
        H = np.array([[1., 0, 0], [0, 1, 0], [1, 0, 0]])
        pts = core.project([[0, 2], [1, 2]], H)
        self.assertTrue(np.isnan(pts[0]).all())
        np.testing.assert_allclose(pts[1], [1, 2])
        residuals, _, status = core.calculate_residuals([[1, 2]], [[1, 2]], np.zeros((3, 3)))
        self.assertEqual(status, 'singular_or_ill_conditioned')
        self.assertTrue(np.isnan(residuals['forward']).all())
        self.assertEqual(core.summary([np.nan, np.inf], 2),
                         {'total_n': 2, 'finite_n': 0, 'median': None, 'p95_linear': None})
        value = adapter.numeric_json({'projection': pts, 'bad': [np.inf, -np.inf, np.nan]})
        self.assertEqual(value['projection'][0], [None, None])
        self.assertEqual(value['bad'], [None, None, None])
        json.dumps(value, allow_nan=False)

    def test_fixed_transforms_and_known_grid_on_generated_texture(self):
        rgb = fixture_rgb()
        for kind in ('scale_crop', 'rotation', 'perspective'):
            generated, H, mask = core.create_transform(rgb, kind)
            self.assertEqual(generated.shape, rgb.shape)
            self.assertEqual(mask.shape, rgb.shape[:2])
            grid, rows = core.synthetic_grid_check(H, H, rgb.shape[:2], mask)
            self.assertEqual(grid['all_grid_n'], 81)
            self.assertGreater(grid['valid_overlap_n'], 20)
            self.assertEqual(grid['error_on_valid_expected_overlap']['median'], 0.)
            self.assertEqual(len(rows), 81)
        with self.assertRaises(ValueError):
            core.create_transform(rgb, 'tuned_transform')
        with self.assertRaises(ValueError):
            core.create_transform(fixture_rgb(8, 8), 'rotation')

    def test_unique_xy_count_does_not_deduplicate_fit(self):
        pts = np.array([[1, 1], [10, 1], [10, 10], [1, 10], [1, 1]], np.float64)
        f = feature(pts)
        with mock.patch.object(core, 'knn_rows', side_effect=[neighbours(5), neighbours(5)]), \
             mock.patch.object(core.cv2, 'findHomography', return_value=(np.eye(3), np.ones((5, 1), np.uint8))) as fitted:
            result, _, _, matches = core.analyse_pair(f, f)
        self.assertEqual(len(fitted.call_args.args[0]), 5)
        self.assertEqual(len(matches), 5)
        self.assertEqual(result['ransac_inliers_n'], 5)
        self.assertEqual(result['ransac_unique_inlier_xy_pairs_n'], 4)
        self.assertEqual(result['residuals_ransac_inliers']['resized']['forward']['median'], 0.)
        self.assertIsNone(result['truth_classification'])
        self.assertEqual(fitted.call_args.kwargs,
                         {'method': core.cv2.RANSAC, 'ransacReprojThreshold': 3., 'maxIters': 2000, 'confidence': .995})

    def test_strict_ratio_missing_neighbours_and_mutual(self):
        def match(q, t, distance):
            return SimpleNamespace(queryIdx=q, trainIdx=t, distance=distance)
        returned = [[match(0, 0, 3.), match(0, 1, 4.)],
                    [match(1, 1, 2.), match(1, 0, 4.)],
                    [match(2, 0, 0.), match(2, 1, 0.)], [match(3, 1, 1.)]]
        matcher = SimpleNamespace(knnMatch=mock.Mock(return_value=returned))
        with mock.patch.object(core.cv2, 'BFMatcher', return_value=matcher) as factory:
            rows = core.knn_rows(np.zeros((4, 128), np.float32), np.zeros((2, 128), np.float32), 'left_to_right')
        factory.assert_called_once_with(core.cv2.NORM_L2, crossCheck=False)
        self.assertEqual([r['passes_strict_ratio_075'] for r in rows], [False, True, False, False])
        reverse = neighbours(2)
        self.assertEqual(len(core.mutual_rows(rows, reverse)), 1)
        reverse[1]['passes_strict_ratio_075'] = False
        self.assertEqual(core.mutual_rows(rows, reverse), [])
        self.assertEqual(core.knn_rows(np.empty((0, 128)), np.empty((0, 128)), 'left_to_right'), [])

    def test_sift_settings_with_fake_detector_only(self):
        kp = SimpleNamespace(pt=(3., 5.), size=2., angle=60., response=.5, octave=1, class_id=-1)
        detector = SimpleNamespace(detectAndCompute=mock.Mock(return_value=([kp], np.ones((1, 128), np.float32))))
        with mock.patch.object(core.cv2, 'SIFT_create', return_value=detector) as factory:
            result = core.extract_features(fixture_rgb())
        factory.assert_called_once_with(nfeatures=4000, nOctaveLayers=3,
                                        contrastThreshold=.04, edgeThreshold=10., sigma=1.6)
        np.testing.assert_allclose(result['points_original'], [[3, 5]])
        self.assertEqual(result['descriptors'].shape, (1, 128))

    def test_insufficient_match_retained_without_fit(self):
        f = feature([[1, 1], [2, 2], [3, 3]])
        with mock.patch.object(core, 'knn_rows', side_effect=[neighbours(3), neighbours(3)]):
            result, _, _, matches = core.analyse_pair(f, f)
        self.assertEqual(result['status'], 'insufficient_mutual_matches')
        self.assertEqual(result['mutual_matches_n'], 3)
        self.assertEqual(result['ransac_unique_inlier_xy_pairs_n'], 0)
        self.assertIsNone(result['homography_left_to_right_original'])
        self.assertIsNone(result['residuals_all_mutual']['resized']['forward']['median'])

    def test_public_summary_removes_raw_mask_and_error(self):
        result = adapter.public_result({'status': 'opencv_estimation_error',
                    'opencv_error': '/private/machine/path', 'ransac_mask': [1, 0],
                    'matrix': [[np.nan]], 'truth_classification': None})
        self.assertNotIn('opencv_error', result)
        self.assertNotIn('ransac_mask', result)
        self.assertEqual(result['matrix'], [[None]])


class PathAndExecutionControls(AlgorithmBlockedBase):
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.data = self.base / 'data'
        self.data.mkdir()
        Image.fromarray(fixture_rgb()).save(self.data / 'one.png')
        Image.fromarray(fixture_rgb()).save(self.data / 'two.png')
        self.case_file = self.base / 'cases.json'
        self.out = self.base / 'new-output'
        self.cases = [{'case_id': 'C_ONE', 'role': 'known_exact_byte_control',
                       'left': 'one.png', 'right': 'two.png', 'transform': None}]
        self.save_cases()
        self.initial = {p.name: p.read_bytes() for p in self.data.iterdir()}

    def save_cases(self):
        self.case_file.write_text(json.dumps({'schema': 1, 'cases': self.cases}), encoding='utf-8')

    def assert_sources_unchanged(self):
        self.assertEqual(self.initial, {p.name: p.read_bytes() for p in self.data.iterdir() if p.is_file()})

    def test_validate_only_has_no_output_or_algorithms(self):
        result = adapter.run(self.data, self.case_file, self.out)
        self.assertEqual(result['state'], 'validated_no_execution')
        self.assertFalse(self.out.exists())
        with contextlib.redirect_stdout(io.StringIO()):
            code = adapter.main(['--data-root', str(self.data), '--cases', str(self.case_file), '--output', str(self.out)])
        self.assertEqual(code, 0)
        self.assertFalse(self.out.exists())
        self.assert_sources_unchanged()

    def test_existing_output_preserved(self):
        self.out.mkdir()
        sentinel = self.out / 'existing.bin'
        sentinel.write_bytes(b'keep this exact content')
        for execute in (False, True):
            with self.assertRaises(ValueError):
                adapter.run(self.data, self.case_file, self.out, execute=execute)
        self.assertEqual(sentinel.read_bytes(), b'keep this exact content')
        self.assertEqual(list(self.out.iterdir()), [sentinel])
        self.assert_sources_unchanged()

    def test_broken_output_symlink_preserved(self):
        try:
            self.out.symlink_to(self.base / 'missing-destination', target_is_directory=True)
        except OSError:
            self.skipTest('Symlink creation unavailable on this platform/account')
        with self.assertRaises(ValueError):
            adapter.run(self.data, self.case_file, self.out, execute=True)
        self.assertTrue(os.path.lexists(self.out))
        self.assertFalse((self.base / 'missing-destination').exists())

    def test_traversal_absolute_drive_unc_backslash_rejected(self):
        for value in ('../one.png', '/one.png', 'C:/one.png', 'C:one.png', '//server/share.png',
                      r'folder\\one.png', r'folder\one.png', './one.png', 'a//one.png'):
            with self.subTest(value=value):
                self.cases[0]['left'] = value
                self.save_cases()
                with self.assertRaises(ValueError):
                    adapter.run(self.data, self.case_file, self.out, execute=True)
                self.assertFalse(self.out.exists())
        self.assert_sources_unchanged()

    def test_source_symlink_escape_rejected(self):
        external = self.base / 'external.png'
        external.write_bytes((self.data / 'one.png').read_bytes())
        try:
            (self.data / 'escape.png').symlink_to(external)
        except OSError:
            self.skipTest('Symlink creation unavailable on this platform/account')
        self.cases[0]['left'] = 'escape.png'
        self.save_cases()
        with self.assertRaises(ValueError):
            adapter.run(self.data, self.case_file, self.out, execute=True)
        self.assertFalse(self.out.exists())
        self.assertEqual(external.read_bytes(), self.initial['one.png'])

    def test_output_inside_data_rejected_before_creation(self):
        with self.assertRaises(ValueError):
            adapter.run(self.data, self.case_file, self.data / 'results', execute=True)
        self.assertFalse((self.data / 'results').exists())
        self.assert_sources_unchanged()

    def test_invalid_later_case_and_wrong_hash_leave_no_output(self):
        self.cases.append(dict(self.cases[0], case_id='LATER', left='../bad.png'))
        self.save_cases()
        with self.assertRaises(ValueError):
            adapter.run(self.data, self.case_file, self.out, execute=True)
        self.assertFalse(self.out.exists())
        self.cases.pop()
        self.cases[0]['left_sha256'] = '0' * 64
        self.save_cases()
        with self.assertRaises(ValueError):
            adapter.run(self.data, self.case_file, self.out, execute=True)
        self.assertFalse(self.out.exists())
        self.assert_sources_unchanged()

    def test_csv_json_equivalence_and_known_transform_schema(self):
        csv_path = self.base / 'cases.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['case_id', 'role', 'left', 'right', 'transform'])
            writer.writeheader()
            writer.writerows(self.cases)
        self.assertEqual(adapter.load_cases(csv_path), adapter.load_cases(self.case_file))
        self.cases[0].update(role='known_transform_control', transform='rotation', right='generated:rotation')
        self.save_cases()
        normalized = adapter.load_cases(self.case_file)[0]
        self.assertEqual(normalized['right'], 'generated:rotation')
        self.cases[0]['right'] = 'two.png'
        self.save_cases()
        with self.assertRaises(ValueError):
            adapter.load_cases(self.case_file)

    def test_generated_fixture_stubbed_run_numeric_outputs_and_hashes(self):
        fake = feature([[1., 1.], [10, 1], [10, 10]])
        numeric = {'status': 'insufficient_mutual_matches', 'mutual_matches_n': 3,
                   'ransac_inliers_n': 0, 'ransac_unique_inlier_xy_pairs_n': 0,
                   'homography_left_to_right_original': None,
                   'truth_classification': None, 'ransac_mask': [0, 0, 0],
                   'opencv_error': str(self.base / 'must-not-leak'),
                   'residuals_all_mutual': {'resized': {'forward': {'p95_linear': None}}},
                   'residuals_ransac_inliers': {'resized': {'forward': {'median': None}}}}
        with mock.patch.object(core, 'extract_features', return_value=fake), \
             mock.patch.object(core, 'analyse_pair', return_value=(numeric, [], [], [])):
            result = adapter.run(self.data, self.case_file, self.out, execute=True)
        self.assertEqual(result['state'], 'completed')
        self.assert_sources_unchanged()
        self.assertEqual({p.name for p in self.out.iterdir()},
                         {'results.json', 'settings.json', 'input_manifest.json', 'summary.csv', 'output_manifest.json'})
        all_text = '\n'.join(p.read_text(encoding='utf-8') for p in self.out.iterdir())
        self.assertNotIn(str(self.base), all_text)
        self.assertNotIn('must-not-leak', all_text)
        self.assertNotIn('ransac_mask', all_text)
        manifest = json.loads((self.out / 'output_manifest.json').read_text())
        for name, expected in manifest['files'].items():
            self.assertEqual(adapter.sha256(self.out / name), expected)
        inputs = json.loads((self.out / 'input_manifest.json').read_text())
        self.assertTrue(inputs['selected_inputs_unchanged'])
        for row in inputs['selected_sources']:
            self.assertEqual(row['sha256_before'], row['sha256_after'])
        written = json.loads((self.out / 'results.json').read_text())['results'][0]
        self.assertIsNone(written['truth_classification'])

    def test_duplicate_csv_headers_and_json_keys_rejected(self):
        bad_csv = self.base / 'duplicate.csv'
        bad_csv.write_text('case_id,role,left,left,right\nC,posthoc_target,one.png,two.png,two.png\n')
        with self.assertRaises(ValueError):
            adapter.load_cases(bad_csv)
        self.case_file.write_text('{"schema":1,"cases":[],"cases":[]}')
        with self.assertRaises(ValueError):
            adapter.run(self.data, self.case_file, self.out, execute=True)
        self.assertFalse(self.out.exists())

    def test_estimator_error_is_not_success(self):
        fake = feature([[1, 1], [10, 1], [10, 10]])
        result = {'status': 'opencv_estimation_error', 'opencv_error': str(self.base),
                  'truth_classification': None}
        with mock.patch.object(core, 'extract_features', return_value=fake), \
             mock.patch.object(core, 'analyse_pair', return_value=(result, [], [], [])):
            state = adapter.run(self.data, self.case_file, self.out, execute=True)
        self.assertEqual(state['state'], 'completed_with_errors_or_changed_inputs')
        self.assertEqual(state['case_errors_n'], 1)
        self.assertNotIn(str(self.base), (self.out / 'results.json').read_text())
        self.assert_sources_unchanged()

    def test_external_source_disappearance_is_recorded_as_failure(self):
        def external_change(rgb):
            (self.data / 'one.png').unlink()
            raise ValueError('Simulated external fixture change')
        with mock.patch.object(core, 'extract_features', side_effect=external_change):
            state = adapter.run(self.data, self.case_file, self.out, execute=True)
        self.assertFalse(state['selected_inputs_unchanged'])
        self.assertEqual(state['state'], 'completed_with_errors_or_changed_inputs')
        inputs = json.loads((self.out / 'input_manifest.json').read_text())
        changed = next(row for row in inputs['selected_sources'] if row['path'] == 'one.png')
        self.assertIsNone(changed['sha256_after'])

    def test_case_error_retained_without_path_traceback(self):
        with mock.patch.object(core, 'extract_features', side_effect=ValueError(str(self.data / 'one.png'))):
            result = adapter.run(self.data, self.case_file, self.out, execute=True)
        self.assertEqual(result['state'], 'completed_with_errors_or_changed_inputs')
        written = (self.out / 'results.json').read_text()
        self.assertNotIn(str(self.data), written)
        self.assertIn('case_execution_error', written)
        self.assertNotIn('Traceback', written)
        self.assert_sources_unchanged()


if __name__ == '__main__':
    unittest.main()
