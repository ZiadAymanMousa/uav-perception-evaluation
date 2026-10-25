# Third-party software and source notices

Original study code and documentation: Copyright (c) 2026 Ziad Ismail. Code licence: AGPL-3.0-only; full text in LICENSE. Existing upstream rights are preserved.

## Ultralytics

Ultralytics YOLO 8.3.253 and ultralytics-thop 2.2.2 use AGPL-3.0. Upstream notice: “Ultralytics — AGPL-3.0 License — https://ultralytics.com/license”. Source: https://github.com/ultralytics/ultralytics/tree/v8.3.253 and https://github.com/ultralytics/thop/tree/v2.2.2 . Matching pair extraction in analyse_size_errors.py follows the inspected non-SciPy matcher in Ultralytics engine/validator.py; upstream attribution and the AGPL licence apply. The package does not vendor the installed framework, AP source snapshot or model weights.

The original public-source preparation added licence headers, extracted the study's own input/coordinate-validation functions into study_inputs.py, and updated the corresponding size-analysis import. Original scientific outputs are retained; this repository is not the upstream Ultralytics package. Do not treat this as a permissively licensed substitute for Ultralytics or its models. Publisher guidance: https://www.ultralytics.com/license .

## Other dependencies

Dependencies are installed separately and keep their own licences: torch/torchvision (BSD-family), NumPy (multiple licences including BSD-3-Clause), OpenCV (Apache-2.0 and bundled notices), Pillow (MIT-CMU), PyYAML (MIT), psutil (BSD-3-Clause), Polars (MIT and runtime notices), Matplotlib and SciPy (their distributed licence/third-party notices). See each installed distribution's licence files. No wheels, CUDA runtime, Python distribution or library binaries are bundled. This inventory is not a clearance for later binary redistribution.

## Related methods and withheld material

SAHI and its author benchmark are MIT licensed; their code is not bundled. Cite the SAHI paper and retain MIT notices if later copying substantial source. MRU author code (AGPL-3.0) and PPAL source (Apache-2.0) are not bundled. Related-method attribution does not make this an exact reproduction.

The VisDrone2018 DET toolkit README says research-purpose only; a general redistribution licence was not established. Captured MATLAB source, its source-mirroring port and reference evaluator branch are not bundled because redistribution permission was not established. No licence in this release is applied to that code or to VisDrone data. Publisher: https://github.com/VisDrone/VisDrone2018-DET-toolkit .

The upstream AP boundary change is documented at https://github.com/ultralytics/ultralytics/pull/24175 . Aggregate reporting of that known behavior is not a claim of original authorship of the change. Its executable publisher-source control is not bundled here.


## Version 0.2 aggregate rendering and matching history

Version 0.2 adds aggregate JSON/CSV and an original rendering script. It changes no installed framework or historical experimental source. The captured current matcher is credited to Ultralytics contributors: https://github.com/ultralytics/ultralytics/pull/26616 and immutable validator commit abc2066dd6bd87f6806f73a170cf8f764ca2ebbe. The captured matcher/AP function/module source is not bundled; only its numerical aggregates and identifying hashes are included. No licence notice grants rights over withheld data, weights or native toolkit source.


## Version0.3 prospective execution and label-budget core

The new original adapters retain AGPL-3.0-only and upstream rights. The selection core calls the separately installed Ultralytics8.3.253 AP function; captured framework/native/current-matcher implementation is not vendored. Two study-owned converted-score/geometry helpers retain their implementation with licence headers. Dataset filenames and query-index masks describe the experiment; no image/annotation data or labels are redistributed.


## Version 0.5 test-dev execution and rendering

The original test-dev execution adapters, synthetic controls and aggregate renderer retain AGPL-3.0-only, copyright 2026 Ziad Ismail, and existing upstream rights. The new component imports the unchanged study-owned conversion, geometry, training-context and converted-score modules from portable_execution/src; their source notices remain intact. Its inference and scoring call the separately installed Ultralytics framework and dependencies under their applicable licences. No captured framework/native/current-matcher source, runtime binary or model weight is newly redistributed.

The release-time task configuration contains content/checkpoint/source identities, class names and fixed numerical settings. Those hashes grant no rights over the independently obtained VisDrone inputs or checkpoints. Original data, converted labels, exclusion records and raw predictions remain unbundled. Tables and figures are rendered from aggregate evidence; reconstruction is not an official challenge evaluation.


## Version 0.6 native-policy numerical evidence

This update adds study aggregates and figures for the separate post hoc native-policy test-dev check. The VisDrone toolkit source, source-mirroring port, execution wrappers and raw reference inputs remain unbundled. Existing toolkit redistribution limitations and upstream rights are unchanged. Aggregate evidence grants no rights over the source code, data, model weights or runtime dependencies. No new native-scoring implementation or dependency binary is included.
