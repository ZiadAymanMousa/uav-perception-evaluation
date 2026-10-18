# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Synthetic source -> conversion -> binding controls; never touch real data."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'src'))
from audit_dataset import audit, digest
from semantic_dataset import identity
from portable_preparer import (prepare_dataset, verify_prepared_dataset,
                               read_json, write_json, load_reviewed_config)


class PortablePreparationTests(unittest.TestCase):
    def setUp(self):
        # Preserve all attempts, including failed fixtures, under this owned directory.
        parent = HERE / 'test_artifacts'
        parent.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix=self._testMethodName + '_', dir=parent))
        self.raw = {s: self.root / ('raw_' + s) for s in ('train', 'val')}
        for split, count in (('train', 2), ('val', 1)):
            for folder in ('images', 'annotations'):
                (self.raw[split] / folder).mkdir(parents=True)
            for index in range(count):
                name = f'{split}{index}'
                Image.new('RGB', (20, 20), color=(30 + index * 30, 40 if split == 'train' else 80, 50)).save(
                    self.raw[split] / 'images' / (name + '.jpg'))
                (self.raw[split] / 'annotations' / (name + '.txt')).write_text(
                    '2,2,4,4,1,1,0,0\n', encoding='utf-8')
        report = audit(self.raw)
        self.config = read_json(HERE / 'reviewed_task.json')
        self.config['scope'] = 'synthetic-control-only'
        self.config['expected_exceptions'] = []
        self.config['expected_source_manifest_sha256'] = report['manifest_sha256']
        self.config['splits'] = {}
        # Independently specify expected converted fixture bytes/geometry. Do not
        # derive expected labels by calling the converter under test.
        expected = self.root / 'expected_material'
        manifest = {'status': 'complete', 'protocol': 'UVD-1', 'policy': self.config['policy'],
                    'class_names': self.config['class_names'], 'exceptions': [], 'splits': {}}
        for split, count in (('train', 2), ('val', 1)):
            for kind in ('images', 'labels', 'excluded_annotations'):
                (expected / kind / split).mkdir(parents=True)
            entries = []
            for source in report['splits'][split]['manifest']:
                name = Path(source['image']).name
                stem = Path(name).stem
                image = expected / 'images' / split / name
                shutil.copyfile(self.raw[split] / source['image'], image)
                label = expected / 'labels' / split / (stem + '.txt')
                label.write_text('0 0.200000000 0.200000000 0.200000000 0.200000000\n', encoding='utf-8')
                side = expected / 'excluded_annotations' / split / (stem + '.json')
                write_json(side, {'not_scored_targets': [], 'known_exceptions': [], 'duplicates': []})
                entries.append({'image': name, 'image_sha256': digest(image),
                                'source_annotation_sha256': source['annotation_sha256'],
                                'label_sha256': digest(label), 'sidecar_sha256': digest(side), 'targets': 1})
            counts = {'images': count, 'targets': count, 'excluded': 0,
                      'known_exceptions': 0, 'duplicates': 0, 'clipped': 0}
            class_counts = {'0': count}
            manifest['splits'][split] = {'counts': counts, 'class_counts': class_counts, 'entries': entries}
            self.config['splits'][split] = {'counts': counts, 'class_counts': class_counts}
        write_json(expected / 'manifest.json', manifest)
        self.config['expected_semantic_identity_sha256'] = identity(expected)['identity_sha256']
        self.config_path = self.root / 'fixture_config.json'
        write_json(self.config_path, self.config)
        self.dest = self.root / 'prepared'
        self.binding = prepare_dataset(self.raw, self.dest, self.config_path, allow_synthetic=True)

    def verify(self, **kwargs):
        return verify_prepared_dataset(self.dest, self.config_path, allow_synthetic=True, **kwargs)

    def manifest(self):
        return read_json(self.dest / 'manifest.json')

    def save_manifest(self, value):
        write_json(self.dest / 'manifest.json', value)

    def rewrite_label(self, text):
        label = self.dest / 'labels/train/train0.txt'
        label.write_text(text, encoding='utf-8')
        m = self.manifest()
        m['splits']['train']['entries'][0]['label_sha256'] = digest(label)
        self.save_manifest(m)

    def test_standalone_prepare_and_full_source_reaudit(self):
        result = self.verify(expected_binding=self.dest / 'binding.json', source_roots=self.raw)
        self.assertEqual(result['image_count'], 3)
        self.assertEqual(result['target_count'], 3)
        self.assertTrue(result['raw_sources_reaudited_now'])
        self.assertTrue(result['converted_images_labels_sidecars_fully_rehashed'])
        self.assertTrue(result['expected_binding_checked'])
        self.assertEqual(read_json(self.dest / 'preparation_record.json')['status'], 'completed')

    def test_default_verification_does_not_claim_raw_reread(self):
        result = self.verify()
        self.assertFalse(result['raw_sources_reaudited_now'])
        self.assertFalse(result['expected_binding_checked'])

    def test_existing_output_is_preserved(self):
        before = digest(self.dest / 'manifest.json')
        with self.assertRaisesRegex(ValueError, 'Output exists'):
            prepare_dataset(self.raw, self.dest, self.config_path, allow_synthetic=True)
        self.assertEqual(digest(self.dest / 'manifest.json'), before)

    def test_synthetic_config_rejected_by_default_public_entry(self):
        with self.assertRaisesRegex(ValueError, 'scope'):
            load_reviewed_config(self.config_path)

    def test_unchanged_saved_binding_checks_actual_bytes(self):
        result = self.verify(expected_binding=self.binding)
        self.assertEqual(result['dataset_manifest_sha256'], self.binding['dataset_manifest_sha256'])

    def test_forged_signature_rejected(self):
        forged = copy.deepcopy(self.binding)
        forged['semantic_identity_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'forged/stale'):
            self.verify(expected_binding=forged)

    def test_missing_binding_fields_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.verify(expected_binding={'schema': self.binding['schema']})

    def test_unknown_binding_schema_rejected(self):
        value = copy.deepcopy(self.binding)
        value['schema'] = 'future-unknown'
        with self.assertRaisesRegex(ValueError, 'forged/stale'):
            self.verify(expected_binding=value)

    def test_nonsemantic_manifest_metadata_accepted_with_new_binding(self):
        m = self.manifest()
        m.update(created_utc='2099-01-01T00:00:00+00:00', seconds=99.0, converter_sha256='different-generation')
        self.save_manifest(m)
        result = self.verify()
        self.assertEqual(result['semantic_identity_sha256'], self.binding['semantic_identity_sha256'])
        self.assertNotEqual(result['dataset_manifest_sha256'], self.binding['dataset_manifest_sha256'])
        with self.assertRaisesRegex(ValueError, 'forged/stale'):
            self.verify(expected_binding=self.binding)

    def test_nonsemantic_audit_serialization_accepted_after_explicit_rebinding(self):
        report = read_json(self.dest / 'source_audit.json')
        report['created_utc'] = '2099-01-01T00:00:00+00:00'
        report['splits']['train']['root'] = 'another-machine/source/train'
        (self.dest / 'source_audit.json').write_text(json.dumps(report, sort_keys=True), encoding='utf-8')
        m = self.manifest()
        m['audit_file_sha256'] = digest(self.dest / 'source_audit.json')
        self.save_manifest(m)
        result = self.verify()
        self.assertEqual(result['semantic_identity_sha256'], self.binding['semantic_identity_sha256'])
        self.assertNotEqual(result['source_audit_sha256'], self.binding['source_audit_sha256'])

    def test_relocation_accepts_new_correct_yaml_and_preserves_semantics(self):
        relocated = self.root / 'relocated'
        shutil.copytree(self.dest, relocated)
        data = yaml.safe_load((relocated / 'dataset.yaml').read_text())
        data['path'] = relocated.resolve().as_posix()
        (relocated / 'dataset.yaml').write_text(yaml.safe_dump(data), encoding='utf-8')
        result = verify_prepared_dataset(relocated, self.config_path, allow_synthetic=True)
        self.assertEqual(result['semantic_identity_sha256'], self.binding['semantic_identity_sha256'])
        self.assertNotEqual(result['dataset_yaml_sha256'], self.binding['dataset_yaml_sha256'])
        with self.assertRaisesRegex(ValueError, 'forged/stale'):
            verify_prepared_dataset(relocated, self.config_path, expected_binding=self.binding, allow_synthetic=True)

    def test_yaml_root_redirection_rejected(self):
        data = yaml.safe_load((self.dest / 'dataset.yaml').read_text())
        data['path'] = self.root.resolve().as_posix()
        (self.dest / 'dataset.yaml').write_text(yaml.safe_dump(data), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'redirects its root'):
            self.verify()

    def test_yaml_train_redirection_rejected(self):
        data = yaml.safe_load((self.dest / 'dataset.yaml').read_text())
        data['train'] = str(self.raw['train'] / 'images')
        (self.dest / 'dataset.yaml').write_text(yaml.safe_dump(data), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'redirects a split'):
            self.verify()

    def test_yaml_extra_field_rejected(self):
        data = yaml.safe_load((self.dest / 'dataset.yaml').read_text())
        data['download'] = 'external-command'
        (self.dest / 'dataset.yaml').write_text(yaml.safe_dump(data), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unsupported fields'):
            self.verify()

    def test_yaml_class_order_rejected(self):
        data = yaml.safe_load((self.dest / 'dataset.yaml').read_text())
        data['names'][0], data['names'][1] = data['names'][1], data['names'][0]
        (self.dest / 'dataset.yaml').write_text(yaml.safe_dump(data), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'class order'):
            self.verify()

    def test_missing_image_rejected(self):
        (self.dest / 'images/train/train0.jpg').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or extra images'):
            self.verify()

    def test_extra_image_rejected(self):
        shutil.copyfile(self.dest / 'images/train/train0.jpg', self.dest / 'images/train/extra.jpg')
        with self.assertRaisesRegex(ValueError, 'Missing or extra images'):
            self.verify()

    def test_missing_exclusion_sidecar_rejected(self):
        (self.dest / 'excluded_annotations/train/train0.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or extra excluded_annotations'):
            self.verify()

    def test_extra_data_directory_rejected(self):
        (self.dest / 'images/train/nested').mkdir()
        with self.assertRaisesRegex(ValueError, 'nested'):
            self.verify()

    def test_extra_root_metadata_rejected(self):
        (self.dest / 'untracked.txt').write_text('unexpected', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Unexpected root metadata'):
            self.verify()

    def test_image_bytes_mutation_rejected(self):
        path = self.dest / 'images/train/train0.jpg'
        path.write_bytes(path.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'Image bytes changed'):
            self.verify()

    def test_one_float32_step_geometry_mutation_rejected_despite_new_byte_hash(self):
        moved = float(np.nextafter(np.float32(.2), np.float32(1)))
        self.rewrite_label(f'0 {moved!r} 0.2 0.2 0.2\n')
        with self.assertRaisesRegex(ValueError, 'semantic task identity differs'):
            self.verify()

    def test_equivalent_decimal_and_newline_labels_preserve_semantics(self):
        self.rewrite_label('0 2e-1 .2000000000 0.200 .2\n\n')
        result = self.verify()
        self.assertEqual(result['semantic_identity_sha256'], self.binding['semantic_identity_sha256'])
        with self.assertRaisesRegex(ValueError, 'forged/stale'):
            self.verify(expected_binding=self.binding)

    def test_class_mutation_rejected_despite_new_byte_hash(self):
        self.rewrite_label('1 .2 .2 .2 .2\n')
        with self.assertRaisesRegex(ValueError, 'semantic task identity differs'):
            self.verify()

    def test_sidecar_semantic_mutation_rejected_despite_new_byte_hash(self):
        side = self.dest / 'excluded_annotations/train/train0.json'
        value = read_json(side)
        value['duplicates'] = [{'reason': 'invented'}]
        write_json(side, value)
        m = self.manifest()
        m['splits']['train']['entries'][0]['sidecar_sha256'] = digest(side)
        self.save_manifest(m)
        with self.assertRaisesRegex(ValueError, 'semantic task identity differs'):
            self.verify()

    def test_equivalent_sidecar_serialization_preserves_semantics(self):
        side = self.dest / 'excluded_annotations/train/train0.json'
        value = read_json(side)
        side.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')
        m = self.manifest()
        m['splits']['train']['entries'][0]['sidecar_sha256'] = digest(side)
        self.save_manifest(m)
        result = self.verify()
        self.assertEqual(result['semantic_identity_sha256'], self.binding['semantic_identity_sha256'])

    def test_duplicate_manifest_image_rejected(self):
        m = self.manifest()
        m['splits']['train']['entries'][1] = copy.deepcopy(m['splits']['train']['entries'][0])
        self.save_manifest(m)
        with self.assertRaisesRegex(ValueError, 'Duplicate image'):
            self.verify()

    def test_image_order_change_rejected(self):
        m = self.manifest()
        m['splits']['train']['entries'].reverse()
        self.save_manifest(m)
        with self.assertRaisesRegex(ValueError, 'selections differ'):
            self.verify()

    def test_changed_policy_rejected(self):
        m = self.manifest()
        m['policy'] = 'different policy'
        self.save_manifest(m)
        with self.assertRaisesRegex(ValueError, 'policy differs'):
            self.verify()

    def test_changed_exception_policy_rejected(self):
        m = self.manifest()
        m['exceptions'] = [{'reason': 'invented'}]
        self.save_manifest(m)
        with self.assertRaisesRegex(ValueError, 'Exception policy'):
            self.verify()

    def test_incomplete_conversion_rejected(self):
        m = self.manifest()
        m['status'] = 'converting'
        self.save_manifest(m)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.verify()

    def test_changed_retained_audit_rejected(self):
        value = read_json(self.dest / 'source_audit.json')
        value['created_utc'] = 'changed'
        write_json(self.dest / 'source_audit.json', value)
        with self.assertRaisesRegex(ValueError, 'audit bytes changed'):
            self.verify()

    def test_forged_source_content_hash_rejected(self):
        report = read_json(self.dest / 'source_audit.json')
        report['splits']['train']['manifest'][0]['annotation_sha256'] = '0' * 64
        write_json(self.dest / 'source_audit.json', report)
        m = self.manifest()
        m['audit_file_sha256'] = digest(self.dest / 'source_audit.json')
        self.save_manifest(m)
        with self.assertRaisesRegex(ValueError, 'source content differs'):
            self.verify()

    def test_original_annotation_change_detected_by_optional_reaudit(self):
        ann = self.raw['train'] / 'annotations/train0.txt'
        ann.write_text('3,2,4,4,1,1,0,0\n', encoding='utf-8')
        # Converted files remain unchanged; this distinction is explicit.
        self.assertFalse(self.verify()['raw_sources_reaudited_now'])
        with self.assertRaisesRegex(ValueError, 'source content differs'):
            self.verify(source_roots=self.raw)

    def test_failed_source_preparation_is_preserved(self):
        ann = self.raw['train'] / 'annotations/train0.txt'
        ann.write_text('changed-invalid-original\n', encoding='utf-8')
        output = self.root / 'failed_source_preparation'
        with self.assertRaises(ValueError):
            prepare_dataset(self.raw, output, self.config_path, allow_synthetic=True)
        rec = read_json(output / 'preparation_record.json')
        self.assertEqual(rec['status'], 'failed')
        self.assertEqual(rec['phase'], 'raw_source_audit')
        self.assertTrue((output / 'source_audit.json').exists())

    def test_config_unknown_schema_rejected(self):
        config = copy.deepcopy(self.config)
        config['schema'] = 'unknown'
        write_json(self.config_path, config)
        with self.assertRaisesRegex(ValueError, 'Unknown reviewed-task'):
            self.verify()

    def test_changed_config_is_not_silently_accepted_as_same_byte_binding(self):
        config = copy.deepcopy(self.config)
        config['trust_note'] = 'new nonsemantic explanation'
        write_json(self.config_path, config)
        self.assertEqual(self.verify()['semantic_identity_sha256'], self.binding['semantic_identity_sha256'])
        with self.assertRaisesRegex(ValueError, 'config_sha256'):
            self.verify(expected_binding=self.binding)


if __name__ == '__main__':
    unittest.main(verbosity=2)
