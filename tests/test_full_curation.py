# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare_full_dataset import curate_lines, deduplicate_labels
from prepare_training_pilot import convert_rows


class FullCurationTests(unittest.TestCase):
    def test_exact_known_exception_is_preserved(self):
        lines=['0,0,1,1,1,1,0,0']*11+['611,158,4,0,1,4,0,0']
        rows, exceptions=curate_lines(lines,'train','9999985_00000_d_0000020.txt')
        self.assertEqual(len(rows),11)
        self.assertEqual(exceptions[0]['line'],12)
        self.assertEqual(exceptions[0]['raw'],lines[11])

    def test_changed_exception_fails_closed(self):
        lines=['0,0,1,1,1,1,0,0']*11+['611,158,5,0,1,4,0,0']
        with self.assertRaisesRegex(ValueError,'changed'):
            curate_lines(lines,'train','9999985_00000_d_0000020.txt')

    def test_same_bad_row_in_other_file_is_not_allowed(self):
        rows, exceptions=curate_lines(['611,158,4,0,1,4,0,0'],'train','different.txt')
        self.assertEqual(exceptions,[])
        with self.assertRaisesRegex(ValueError,'nonpositive'):
            convert_rows(rows,1000,1000)

    def test_known_training_exception_never_applies_to_validation(self):
        lines=['0,0,1,1,1,1,0,0']*11+['611,158,4,0,1,4,0,0']
        rows, exceptions=curate_lines(lines,'val','9999985_00000_d_0000020.txt')
        self.assertEqual(exceptions,[])
        with self.assertRaises(ValueError):
            convert_rows(rows,1000,1000)

    def test_duplicates_match_framework_float32_precision(self):
        labels=[(0,.5,.5,.1,.1),(0,.5+1e-12,.5,.1,.1),(1,.5,.5,.1,.1)]
        unique, removed=deduplicate_labels(labels)
        self.assertEqual(len(unique),2)
        self.assertEqual(len(removed),1)
        self.assertEqual(removed[0]['converted_row_index'],1)


if __name__=='__main__':
    unittest.main()
