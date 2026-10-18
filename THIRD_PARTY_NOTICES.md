# Third-party software and source notices

Original study code and documentation: Copyright (c) 2026 Ziad Ismail. Candidate code licence: AGPL-3.0-only; full text in LICENSE. Existing upstream rights are preserved.

## Ultralytics

Ultralytics YOLO 8.3.253 and ultralytics-thop 2.2.2 use AGPL-3.0. Upstream notice: “Ultralytics — AGPL-3.0 License — https://ultralytics.com/license”. Source: https://github.com/ultralytics/ultralytics/tree/v8.3.253 and https://github.com/ultralytics/thop/tree/v2.2.2 . Matching pair extraction in analyse_size_errors.py follows the inspected non-SciPy matcher in Ultralytics engine/validator.py; upstream attribution and the AGPL licence apply. The package does not vendor the installed framework, AP source snapshot or model weights.

Changes to study source for this candidate are licence headers, extraction of the study's own input/coordinate-validation helpers into study_inputs.py, and the corresponding size-analysis import. Original scientific outputs are retained; the candidate is not the upstream Ultralytics package. Do not treat this as a permissively licensed substitute for Ultralytics or its models. Publisher guidance: https://www.ultralytics.com/license .

## Other dependencies

Dependencies are installed separately and keep their own licences: torch/torchvision (BSD-family), NumPy (multiple licences including BSD-3-Clause), OpenCV (Apache-2.0 and bundled notices), Pillow (MIT-CMU), PyYAML (MIT), psutil (BSD-3-Clause), Polars (MIT and runtime notices), Matplotlib and SciPy (their distributed licence/third-party notices). See each installed distribution's licence files. No wheels, CUDA runtime, Python distribution or library binaries are bundled. This inventory is not a clearance for later binary redistribution.

## Related methods and withheld material

SAHI and its author benchmark are MIT licensed; their code is not bundled. Cite the SAHI paper and retain MIT notices if later copying substantial source. MRU author code (AGPL-3.0) and PPAL source (Apache-2.0) are not bundled. Related-method attribution does not make this an exact reproduction.

The VisDrone2018 DET toolkit README says research-purpose only; a general redistribution licence was not established. Captured MATLAB source, its source-mirroring port and reference evaluator branch are excluded pending rights review. No licence in this candidate is applied to that code or to VisDrone data. Publisher: https://github.com/VisDrone/VisDrone2018-DET-toolkit .

The upstream AP boundary change is documented at https://github.com/ultralytics/ultralytics/pull/24175 . Aggregate reporting of that known behavior is not a claim of original authorship of the change. Its executable publisher-source control is not bundled here.


## Version 0.2 aggregate rendering and matching history

Version 0.2 adds aggregate JSON/CSV and an original rendering script. It changes no installed framework or historical experimental source. The captured current matcher is credited to Ultralytics contributors: https://github.com/ultralytics/ultralytics/pull/26616 and immutable validator commit abc2066dd6bd87f6806f73a170cf8f764ca2ebbe. The captured matcher/AP function/module source is not bundled; only its numerical aggregates and identifying hashes are included. No licence notice grants rights over withheld data, weights or native toolkit source.


## Version0.3 prospective execution and label-budget core

The new original adapters retain AGPL-3.0-only and upstream rights. The selection core calls the separately installed Ultralytics8.3.253 AP function; captured framework/native/current-matcher implementation is not vendored. Two study-owned converted-score/geometry helpers retain their implementation with licence headers. Dataset filenames and query-index masks describe the experiment; no image/annotation data or labels are redistributed.
