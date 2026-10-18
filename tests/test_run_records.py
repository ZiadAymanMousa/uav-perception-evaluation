# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
import unittest
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

from run_records import canonical_output_hashes


class OutputHashPaths(unittest.TestCase):
    def test_windows_completion_record_selects_same_weight_as_posix(self):
        posix = {'fit/weights/best.pt':'best','fit/results.csv':'csv'}
        windows = {'fit\\weights\\best.pt':'best','fit\\results.csv':'csv'}
        self.assertEqual(canonical_output_hashes(windows),posix)
        self.assertEqual(canonical_output_hashes(posix),posix)

    def test_aliases_cannot_silently_replace_weight_identity(self):
        with self.assertRaisesRegex(ValueError,'Ambiguous'):
            canonical_output_hashes({'fit/weights/best.pt':'a','fit\\weights\\best.pt':'b'})

    def test_paths_cannot_resolve_outside_recorded_run(self):
        for name in ('../best.pt','C:/best.pt','/best.pt','fit/../best.pt','fit//best.pt'):
            with self.subTest(name=name),self.assertRaises(ValueError):
                canonical_output_hashes({name:'hash'})


if __name__=='__main__':
    unittest.main()
