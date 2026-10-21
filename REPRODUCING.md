# Reproduction in version 0.4

Use [portable_execution/README.md](portable_execution/README.md) for the new prospective data/training/inference/scoring path and [PORTABLE_VALIDATION.md](PORTABLE_VALIDATION.md) for actual checked scope. The historical manifest-bound code remains in root src/ to preserve the original source. The separately versioned adapter does not relabel existing weights as trained on fresh data. Clean installation, full prospective retraining and complete public reproduction remain unverified.

The label-budget [core/controls/protocol](label_budget/READ_FIRST.md) and aggregates are supplied. Its real full-grid orchestration and raw sufficient statistics are not bundled, so aggregates are inspectable numerical evidence rather than a standalone AP recomputation.

The following 0.2 reproduction notes describe the preserved historical branch; their manifest guard remains deliberate and is not a restriction on the new explicit-path prospective adapter.

---

# Reproduction scope

## Verified scope

The experiment used Windows 11, Python 3.13.13, torch 2.10.0+cu128, torchvision 0.25.0+cu128 and an RTX 5070 Ti. [Observed versions](environment/observed_versions.json) include Pillow 12.2.0. The provided requirement lists describe observed versions; they are not a complete transitive lock or a verified fresh CPU/GPU installation. No runtime is vendored.

The packaged synthetic tests and own-loader extraction are checked with the existing environment. Clean installation, fresh GPU training, checkpoint equivalence across hardware and full package reproduction have not been executed. Seed/deterministic settings do not promise bitwise training determinism.

## Optional staged research workflow

1. Obtain the original train/val datasets independently under the publisher's terms. [DATA.md](DATA.md) gives scope. These historical steps concern training/development; the separate completed test-dev confirmation is documented below and its adapter is not bundled. Create this package's `data/VisDrone2019-DET-train` and `data/VisDrone2019-DET-val` layouts only after obtaining the data lawfully.
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


## Version 0.4 test-dev scope

The completed [confirmation](TESTDEV_RESULTS.md) adds all twelve development/test-dev AP rows, six contrasts, twelve diagnostic cost rows, PNG/SVG, derived integrity/validation summaries, a later presentation of the earlier internal plan and a technical-report PDF. These are inspectable aggregates and protocol evidence. No dataset, labels, weights, raw predictions, internal adapter, full audit/orchestrator or test-dev figure-building program is supplied.

The historical aggregate-rendering command remains unchanged and does not process these new test-dev files. The executable code/tests are byte-identical to version 0.3.0. Do not claim this release can independently regenerate test-dev AP or its figure. A portable executable test-dev extension would need explicit input/binding interfaces and targeted source/data/checkpoint/metric/completeness controls in a separately reviewed change.

The actual retained test-dev runs completed on all 1,610 images with the three existing checkpoints. Separate same-algorithm CPU scoring agreed exactly for twelve summaries. That result does not close clean-install/full prospective training/cross-hardware determinism gaps, supply a second evaluator implementation, extend development native/reference controls to test-dev, or prove scene-independent performance. The [validation summary](results/testdev/validation_summary.json) records the exact checked scope.
