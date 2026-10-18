# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Synthetic checks of data-integrity failure modes; not research benchmarks."""
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_dataset import audit, parse_annotation


class AuditTests(unittest.TestCase):
    def temporary(self):
        project = Path(__file__).resolve().parents[1]
        base = (project / "artifacts" / "test_tmp").resolve()
        self.assertTrue(base.is_relative_to(project))
        base.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=base)
        self.assertTrue(Path(directory.name).resolve().is_relative_to(base))
        return directory

    def fixture(self, root, colour="red", rows="1,2,10,10,1,1,0,0\n"):
        (root / "images").mkdir(parents=True)
        (root / "annotations").mkdir()
        Image.new("RGB", (64, 64), colour).save(root / "images" / "sample.png")
        (root / "annotations" / "sample.txt").write_text(rows, encoding="utf-8")

    def test_ignore_and_other_are_preserved_and_not_scored_as_targets(self):
        with self.temporary() as d:
            root = Path(d)
            self.fixture(root, rows="0,0,20,20,0,0,0,0\n1,2,10,10,1,1,0,1,\n3,4,5,5,1,11,0,0\n")
            r = audit({"train": root})
            counts = r["splits"]["train"]["counts"]
            self.assertEqual(r["error_count"], 0)
            self.assertEqual(counts["annotation_rows"], 3)
            self.assertEqual(counts["ignore_regions"], 1)
            self.assertEqual(counts["other_category_rows"], 1)
            self.assertEqual(counts["scored_target_objects"], 1)

    def test_rejects_malformed_or_impossible_ground_truth(self):
        for row in ["1,2,3", "1,2,0,4,1,1,0,0", "1,2,3,4,1,12,0,0",
                    "1,2,3,4,0.9,1,0,0", "1,2,3,4,1,1,0,3"]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                parse_annotation(row.split(","))

    def test_identical_images_across_splits_are_errors(self):
        with self.temporary() as d:
            a, b = Path(d) / "train", Path(d) / "val"
            self.fixture(a)
            self.fixture(b)
            r = audit({"train": a, "val": b})
            self.assertEqual(len(r["cross_split_duplicates"]), 1)
            self.assertGreater(r["error_count"], 0)

    def test_bounds_are_flagged_without_silent_clipping(self):
        with self.temporary() as d:
            root = Path(d)
            self.fixture(root, rows="60,60,10,10,1,1,0,0\n")
            r = audit({"train": root})
            self.assertEqual(r["error_count"], 0)
            self.assertEqual(r["warning_count"], 1)

    def test_missing_annotation_is_an_error(self):
        with self.temporary() as d:
            root = Path(d)
            self.fixture(root)
            (root / "annotations" / "sample.txt").unlink()
            self.assertGreater(audit({"train": root})["error_count"], 0)

    def test_manifest_is_stable_across_roots_and_changes_with_labels(self):
        with self.temporary() as d:
            a, b = Path(d) / "a", Path(d) / "b"
            self.fixture(a)
            self.fixture(b)
            first, second = audit({"train": a}), audit({"train": b})
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            (b / "annotations" / "sample.txt").write_text("1,2,11,10,1,1,0,0\n")
            self.assertNotEqual(first["manifest_sha256"], audit({"train": b})["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
