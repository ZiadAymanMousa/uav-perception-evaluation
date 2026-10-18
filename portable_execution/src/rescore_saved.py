# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Rescore strict saved portable outputs; preserve smoke/development purpose."""
import argparse
import json
import os
from pathlib import Path

# Configure package-owned runtime paths before any Ultralytics import.
PROJECT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT / 'artifacts/ultralytics_config'
(CONFIG_DIR / 'Ultralytics').mkdir(parents=True, exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(CONFIG_DIR)
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['MPLCONFIGDIR'] = str(PROJECT / 'artifacts/matplotlib_config')

from portable_preparer import write_json, read_json, require
from audit_dataset import digest
from study_inputs import load_converted_cases
from score_converted_predictions import score_records
from prepare_training_pilot import NAMES


FRAMEWORK_FILES = {'metrics':'utils/metrics.py','matching':'engine/validator.py',
                   'predictor':'models/yolo/detect/predict.py','nms':'utils/nms.py',
                   'detection_validator':'models/yolo/detect/val.py'}


def validate_scoring_runtime(record, *, observed=None):
    """Reject a different installed scoring runtime before recomputing saved AP."""
    if observed is None:
        import torch
        import ultralytics
        base = Path(ultralytics.__file__).parent
        observed = {'ultralytics':ultralytics.__version__,'torch':torch.__version__,
                    'framework_source_hashes':{name:digest(base/path) for name,path in FRAMEWORK_FILES.items()}}
    for name in ('ultralytics','torch'):
        require(observed[name] == record[name], 'Installed scoring version changed: ' + name)
    require(set(record['framework_source_hashes']) == set(FRAMEWORK_FILES),
            'Missing/unknown recorded framework source identities')
    require(observed['framework_source_hashes'] == record['framework_source_hashes'],
            'Installed scoring framework source changed')
    return {'installed_versions_and_framework_sources_match_prediction_run':True,**observed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('run-dir', 'data-root', 'task-config', 'binding', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing rescore output')
    runtime = validate_scoring_runtime(read_json(args.run_dir/'run_record.json'))
    cases, provenance = load_converted_cases(args.run_dir, data_root=args.data_root,
                            task_config=args.task_config, binding=args.binding)
    result = {'purpose': provenance['purpose'], 'provenance': provenance,
              'runtime_verification':runtime,'rescorer_sha256':digest(Path(__file__)),
              'runtime_configuration':{name:os.environ[name] for name in
                  ('YOLO_CONFIG_DIR','YOLO_OFFLINE','YOLO_AUTOINSTALL','MPLCONFIGDIR')},
              'scores': {method: score_records(rows, dict(enumerate(NAMES))) for method, rows in cases.items()},
              'limits': ['Pipeline-smoke scores are integration checks, not detector quality evidence',
                         'Converted scoring only; no native reference/current-matcher branch']}
    write_json(args.output, result)
    print(json.dumps({'status': 'rescored', 'purpose': result['purpose'], 'methods': len(cases)}, indent=2))


if __name__ == '__main__':
    main()
