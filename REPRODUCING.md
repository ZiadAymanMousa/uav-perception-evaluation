# Reproduction in version 0.6

Use [portable_execution/README.md](portable_execution/README.md) for prospective training/development execution and [testdev_execution/README.md](testdev_execution/README.md) for explicit test-dev preparation, inference and retained-output scoring. [PORTABLE_VALIDATION.md](PORTABLE_VALIDATION.md) and the [test-dev validation record](testdev_execution/VALIDATION.md) separate actual checked scopes. The historical manifest-bound code remains in root src/ to preserve the original source. The separately versioned adapter does not relabel existing weights as trained on fresh data. Clean installation, full prospective retraining and complete public reproduction remain unverified.

The label-budget [core/controls/protocol](label_budget/READ_FIRST.md) and aggregates are supplied. Its real full-grid orchestration and raw sufficient statistics are not bundled, so aggregates are inspectable numerical evidence rather than a standalone AP recomputation.

The following 0.2 reproduction notes describe the preserved historical branch; their manifest guard remains deliberate and is not a restriction on the new explicit-path prospective adapter.

---

# Reproduction scope

## Verified scope

The experiment used Windows 11, Python 3.13.13, torch 2.10.0+cu128, torchvision 0.25.0+cu128 and an RTX 5070 Ti. [Observed versions](environment/observed_versions.json) include Pillow 12.2.0. The provided requirement lists describe observed versions; they are not a complete transitive lock or a verified fresh CPU/GPU installation. No runtime is vendored.

The packaged synthetic tests and own-loader extraction are checked with the existing environment. Clean installation, fresh GPU training, checkpoint equivalence across hardware and full package reproduction have not been executed. Seed/deterministic settings do not promise bitwise training determinism.

## Optional staged research workflow

1. Obtain the original train/val datasets independently under the publisher's terms. [DATA.md](DATA.md) gives scope. These historical steps concern training/development; the separate completed test-dev confirmation and its new explicit-input component are documented below. Create this package's `data/VisDrone2019-DET-train` and `data/VisDrone2019-DET-val` layouts only after obtaining the data lawfully.
2. Audit inputs with `python -B src/audit_dataset.py --split train=data/VisDrone2019-DET-train --split val=data/VisDrone2019-DET-val --output artifacts/train_val_audit_20261016.json`. Inspect the documented malformed-training-row policy before conversion; originals remain unchanged.
3. `prepare_full_dataset.py` requires that audit and handles the three exact known exceptions plus float32 duplicate labels. It generates curated inputs/manifests. Ignored pixels remain in the converted training loss/metric.
4. Training uses an independently obtained, licence-compliant original YOLO11n checkpoint at the expected local path, with matching acquisition/hash metadata. `train_full_baseline.py --config configs/baseline_yolo11n_640_seed0.json` is a real 150-epoch GPU workload, not a test command. It requires adequate local resources and reviewed settings. No weights are supplied.
5. Review terminal completion/checkpoint provenance before running `compare_inference.py` in its declared modes. Study mode requires a completed baseline and matching checkpoint/task hashes. It refuses incomplete runs.
6. Converted analyses require saved prediction/run/data records in their documented layout. The size loader validates original image/annotation identity and all image/method pairs; it performs no reference scoring or model inference.

## Deliberate gaps

The exact generated dataset-manifest SHA binds timestamps/serialization and upstream audit metadata as well as content. Re-running conversion on another machine does not automatically regenerate the original full hash. The frozen inference config retains that guard; it is not weakened in this package. A portability adaptation must explicitly verify semantic image/label identity and carry its own version/proof. Do not edit expected hashes merely to suppress a failed check.

Original selected weights, raw predictions, per-object catalogues and private run metadata are withheld. Bundled aggregates can regenerate the displayed tables and plots, but cannot recompute AP from raw detections or independently reproduce detector results. Some scripts retain original artifact names and require those staged inputs. There is no claim of a one-command public benchmark reproduction.

Native evaluator code and captured publisher AP source are not bundled. Native/boundary numerical controls are aggregate evidence only in this package; the complete reference pipeline is not part of the executable package. Optional source additions require their exact licence, notices and hash verification rather than a latest-version substitution.


## Version 0.2 aggregate rendering and completed seed IDs

`python -B src/render_aggregate_results.py` reads only the two bundled aggregate JSON files and writes the 72-row protocol CSV, 96-row matcher CSV, 12-row first-sweep cost CSV and three-panel PNG/SVG. It performs no scoring, detector loading, network access or dataset reconstruction. This rendering and the synthetic suite have been checked in the observed environment; fresh installation is still unverified.

All three completed training configurations are supplied: seed0, seed1_fresh and seed2. `review_completed_baseline.py --run RUN_ID` and `compare_inference.py --baseline-run RUN_ID` already accept each staged completed run; no fixed seed0-only guard was relaxed. Expected IDs are b001_uvd1_yolo11n640_seed0, b004_uvd1_yolo11n640_seed1_fresh and b003_uvd1_yolo11n640_seed2. Training/inference require original staged data and checkpoint metadata; no new GPU run was performed to validate this package. The failed seed1 configuration remains for transparent history and is not a completed replicate.

The unchanged own loader is locally checked against all 6,576 original image/method cases and provenance across the three saved sweeps. This is loader extraction equivalence on retained inputs, not a fresh public reproduction. `analyse_size_errors.py` retains its seed0-specific study input/output IDs and supplies no seed1/2 size result. `analyse_cap_matching.py` additionally needs a prior cap-analysis summary that is not bundled; it is retained diagnostic source, not a complete standalone cap pipeline.

Native source scoring, published-boundary execution and the isolated current-matcher control are **aggregate-only** in this package. Their original inputs and executable reference branch are not supplied. The figure script can render their numbers but cannot independently reproduce them. The exact captured source hashes/URLs identify what was used; replacing them with a current pip package would change the experiment. Do not claim all current-control code is executable from this package.


## Version 0.5 test-dev execution and rendering

The completed [confirmation](TESTDEV_RESULTS.md) retains the twelve AP rows, six contrasts, twelve diagnostic cost rows, original PNG/SVG, derived integrity/validation summaries, later presentation of the earlier internal plan and technical-report PDF. Their scientific values and file bytes are unchanged by this software addition.

The [test-dev component](testdev_execution/README.md) supplies explicit raw/prepared/output/config/checkpoint inputs for preparation, verification, inference and saved-output scoring. Its release-time content identity is independent of paths and timestamps, with exact raw membership/content checks, nine-decimal converted labels, retained original annotations/exclusion fields, source snapshots and launch bindings. It is not a backdated preregistration or a claim that this new component executed the historical full inference runs.

Two checkpoint routes stay separate. Exact frozen replay requires the user-supplied historical seed0/1/2 checkpoint hash. Prospective mode requires an actual completed baseline and its separate training data/config/binding; it cannot relabel fresh training as historical results. The older portable historical-certificate route remains CPU-smoke-only and is unchanged.

On 21 October, complete preparation/verification passed for all 1,610 real pairs and 75,102 targets. All three retained 6,440-row prediction logs were validated and rescored with the same pinned scorer; all twelve complete method-summary objects matched exactly. A new seed-0 CPU inference/verification covered one sorted image and four settings only. The 62 synthetic/CPU controls exercise data/source/config/checkpoint changes, matrix/accounting, completion, provenance and write-path guards. These are existing-environment checks, not clean installation, an independent evaluator, a full new inference sweep or actual completed prospective training.

The separate renderer reads bundled aggregates or six explicitly hash-bound development/test-dev summaries. It validates schemas, cells, differences and workload accounting, then reconstructs 12 AP rows, six contrasts, 12 cost rows and the PNG/SVG. Its 17 controls and both input modes passed. This is table/figure reconstruction without detection or AP recomputation. Numeric agreement does not promise byte-identical graphics across Matplotlib versions, platforms or rendering metadata.

```console
python -B -m unittest discover -s testdev_execution/tests -v
python -B src/render_testdev_results.py --input-dir results/testdev --output-dir artifacts/testdev_render
```

The renderer requires a new output directory. For supplied summaries, `--summary-index` replaces `--input-dir`; the index names three seed pairs with explicit development/test-dev paths and SHA256 values. Paths resolve relative to the index. No retained raw prediction log, data, labels, weights, full private audit/orchestrator or captured native/current-matcher branch is distributed. The standalone source and data-dependent commands do not make those inputs available by association.

The original frozen test-dev confirmation is the earlier all-image, three-checkpoint primary comparison. Its development native/reference/matcher controls do not extend to test-dev by association; the later native-policy test-dev result has its own declared protocol and bounded source gate. Same-algorithm rescoring checks continuity, not evaluator independence, scene-independent generalisation, deployment timing or cross-hardware determinism. See [the execution validation](testdev_execution/VALIDATION.md) and [original confirmation validation summary](results/testdev/validation_summary.json) for the separate scopes.


## Version 0.6 native-policy aggregates

The [post hoc native check](NATIVE_TESTDEV_RESULTS.md) adds all twelve seven-metric rows, per-class AP and class-image weights, protocol/validation presentations and static figures. Existing scientific code, configs, tests and converted result exports are unchanged. The public testdev_execution component continues to implement its documented converted scorer; it does not acquire native scoring because native aggregates are included.

The native scorer, captured toolkit execution wrappers and original reference inputs are not supplied. Inspecting the new CSV/JSON and replotting its aggregates can check presentation and weighted arithmetic, but cannot recompute detector AP without the original predictions, annotations and evaluator. No new native command, clean installation, full inference sweep or prospective training is claimed.

The reference gate compares captured publisher MATLAB-language functions under Octave 11.3.0 on three annotation-selected images for every checkpoint/setting, before twelve full-split Python-port evaluations. This is 36 purposeful cases, not all 1,610 images through the publisher functions. The score-zero-target selection criterion used its fixed fallback. The [validation summary](results/testdev_native/validation_summary.json) gives tolerances and actual agreement; there is no official challenge or exhaustive MATLAB-platform equivalence claim.
