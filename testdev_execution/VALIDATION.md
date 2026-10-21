# Portable test-dev validation

Checked 21 October 2026. By Ziad Ismail.

This record separates complete data preparation, saved-prediction rescoring and a bounded CPU inference check. The historical full test-dev experiment and this later portable execution component are separate executions.

## Completed checks

| Check | Observed scope | Result |
| --- | --- | --- |
| Synthetic and CPU controls | 62 controls, including an actual pinned-scorer positive control and 15 targeted controls for config consistency, recorded policy/runtime/device provenance, relocation, per-view caps and safe write ordering | Passed |
| Real-data preparation | All 1,610 official test-dev image/annotation pairs, read from the raw source and copied to a separate new preparation | Completed |
| Prepared-data verification | Full raw content rehash/reconstruction, exact membership, converted labels, original annotation copies, exclusion sidecars, task/source snapshots and binding | Passed |
| Retained seed0 predictions | 1,610 images and 6,440 image/method rows | All four complete method summaries exactly match saved historical summaries |
| Retained seed1 predictions | 1,610 images and 6,440 image/method rows | All four complete method summaries exactly match saved historical summaries |
| Retained seed2 predictions | 1,610 images and 6,440 image/method rows | All four complete method summaries exactly match saved historical summaries |
| New seed0 CPU inference | First sorted image, all four settings, explicit matching historical checkpoint, actual model parameter device `cpu` | Completed |
| New CPU run verification | The same one-image/four-row output, checked against inputs, checkpoint, source/config snapshots, policy/runtime metadata and freshly recomputed summary | Passed |

The complete preparation contains 77,547 original nonempty annotation rows: 75,102 scored targets and 2,445 excluded rows. Clipping, duplicate removal and malformed-row exceptions are all zero. Every excluded native annotation is retained with its original fields and source line number, and the complete raw annotation bytes are copied separately. The converted metric does not apply native ignore-region masking.

Each retained complete checkpoint log has one row for every image/method pair. Validation checks rotating method order, image identity, original-image geometry, confidence sorting, per-view and merged caps, windows, timing arithmetic and detector/input-pixel accounting before scoring. Across the three logs, 19,320 retained rows were checked. For each checkpoint there are 1,610 scored detector passes for each full-frame setting and 11,448 passes for the fixed-slice/global setting.

Rescoring used the same pinned converted scorer and framework source implementation. All twelve complete method-summary objects matched exactly, including the AP results and diagnostic cost/timing summaries. The timing summaries are calculated from the retained timing rows; this comparison is not a new latency measurement. These twelve matches are method-level parity checks, not twelve independent experiments or an independent evaluator.

The new CPU inference check used exactly one image and all four methods, with 15 warm-up calls recorded separately. It confirms a real bounded path from verified preparation and an actual checkpoint through prediction, saved outputs, scoring and run verification. It does not confirm the new runner's entire 1,610-image inference path. Its CPU timings are not performance benchmark evidence.

## Identified implementation

The task/config pin is unchanged by these checks. It identifies a release-time verification configuration, not an external preregistration or a backdated portable execution.

| Identity | SHA256 |
| --- | --- |
| Task/config bytes | `4ffe21a2cf0de4cb8bdda831537e951fc763d4f6d89b068e61fea05cd6f2f4c8` |
| Path- and timestamp-independent test-dev content | `b50b8d2f31831a22ccc1c80580c65902e443308e958eecc18eab037732f83ac2` |
| Execution source set, including the unchanged imported portable modules | `98dc6962db70c6257348d5c9ed6af2fd34017346c53a7779f6ae94296ef05037` |

The checked runtime used Ultralytics 8.3.253, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, NumPy 2.4.4 and OpenCV 4.13.0. The five configured framework source hashes matched. A CUDA-capable installed wheel does not mean the CPU smoke used a GPU: its actual parameter device was CPU. This is an observed installed environment, not evidence of a successful clean installation or a complete transitive dependency lock.

## What remains unverified

- No new complete test-dev inference sweep was executed through this component; complete retained outputs were rescored instead.
- No new model training or actual completed prospective-baseline execution was performed. Unit controls check routing to the separate training-context guard and propagation of incomplete-training rejection.
- There was no fresh environment installation, network acquisition test or independently supplied checkpoint/data reproduction.
- Exact saved-output parity does not establish an independent evaluator, native VisDrone challenge scoring, official COCO equivalence, stable latency or peer review.
- A matching checkpoint hash identifies its bytes; rescoring an externally supplied log does not by itself prove that those weights generated the log or certify the original inference runtime.
- The publicly labelled test-dev split was held out from this study. It does not establish a blind benchmark, out-of-distribution performance or independent scenes.

The commands in [README.md](README.md) expose the verified interfaces. Matching historical checkpoint bytes are still required for historical replay; new completed training produces a prospective result rather than recovering the earlier table by association.
