# Prospective portable UVD-1 execution

By Ziad Ismail. Separately versioned prospective execution component.

This package connects the reviewed standalone preparation/binding API to real training, completion review, inference and saved converted scoring. Explicit paths replace the historical generated-manifest dependency for new runs. It preserves the original UVD-1 curation and pinned Ultralytics 8.3.253 inference/scoring policy. No dataset, labels, raw predictions, model weights or runtime are bundled.

The package supplements the historical study. It does not change that study's records or mean that an old checkpoint was trained on a newly prepared dataset.

## Preparation and CPU controls

Independently obtain original train/validation data under the publisher's terms and local starting weights with trustworthy acquisition/hash metadata. Use a reviewed environment; installation is not performed by these scripts. The original study's observed environment and dependency limits remain applicable.

```console
python -B -m unittest discover -s tests -v
python -B src/portable_preparer.py prepare --train-root TRAIN_ROOT --val-root VAL_ROOT --output NEW_DATA_ROOT --config reviewed_task.json
python -B src/portable_preparer.py verify --data-root NEW_DATA_ROOT --config reviewed_task.json
```

The source-content and converted semantic identities, class order, exact malformed-row policy and split counts come from reviewed_task.json. Changing that configuration changes the trusted task. Full verification rehashes converted images/labels/sidecars; supplying both original source roots to verify additionally repeats the raw audit. Dataset YAML must resolve to the verified data, split paths and classes. New outputs are never silently overwritten.

## Real prospective training and review

The supplied training config is an unexecuted 150-epoch workload with a new prospective run ID. Training uses a CUDA GPU only after strict CPU input verification. The separate --preflight-only flag stops before model loading or training:

```console
python -B src/train_full_baseline.py --config configs/baseline_portable_seed0.json --data-root DATA_ROOT --task-config reviewed_task.json --binding DATA_ROOT/binding.json --initial-weights INITIAL_WEIGHTS.pt --initial-weights-metadata ACQUISITION.json --preflight-only
```

ACQUISITION.json must contain the independently verified sha256 of the local initial checkpoint. Remove --preflight-only only to deliberately start real training with adequate resources. The runner calls the pinned framework's actual model.train path and saves epochs, outputs, source snapshots and launch binding. It does not manufacture a completed training record. No paid compute is started by the package itself.

Resume requires a terminal prior process, matching actual launch data/config/runner/helper/initial-checkpoint metadata bytes and the saved last checkpoint. Semantic equivalence alone never permits resuming with a new manifest. The last checkpoint must also match the checksum retained in progress.json; a missing checksum or changed file requires reviewing the interrupted state before resume. Completed runs cannot resume. Preserve failed runs before considering any replacement.

After actual completion:

```console
python -B src/review_completed_baseline.py --run-dir BASELINE_RUN_DIR --data-root DATA_ROOT --task-config reviewed_task.json --binding DATA_ROOT/binding.json --output NEW_REVIEW_DIR
python -B src/compare_inference.py --purpose development-study --run-name NEW_STUDY_ID --baseline-run-dir BASELINE_RUN_DIR --weights BASELINE_RUN_DIR/fit/weights/best.pt --config configs/inference_portable_v1.json --data-root DATA_ROOT --task-config reviewed_task.json --binding DATA_ROOT/binding.json --device 0
```

Development mode requires a completed prospective portable baseline, exact selected checkpoint, unchanged config/helper snapshots and matching launch binding. Contiguous CSV/epoch callbacks, finite values and the preserved final-validation event are checked. The full reviewer additionally inspects checkpoint parameters, class names, selection fitness and curves. Historical checkpoints are rejected from this prospective development route. All 548 exposed validation images remain development data.

## Bounded historical-checkpoint CPU smoke

A smoke test may use a historical completed ten-class checkpoint only with explicit original training evidence and a newly checked semantic-equivalence certificate. Certificate creation takes explicit original baseline/data/audit paths; those are optional legacy evidence for this smoke route, never dependencies of prospective preparation or training. The certificate checks original record/config/runner/checkpoint hashes and fully rehashes the historical converted task, preserving its different manifest bytes.

Use src/execution_inputs.py --help for the certificate's explicit paths and a new output. Then:

```console
python -B src/compare_inference.py --purpose pipeline-smoke --run-name NEW_SMOKE_ID --weights HISTORICAL_BEST.pt --historical-certificate CERTIFICATE.json --config configs/inference_portable_v1.json --data-root DATA_ROOT --task-config reviewed_task.json --binding DATA_ROOT/binding.json --device cpu --image-limit 1
```

Smoke is limited to one or two images on CPU. The saved record labels the historical training origin and states that the checkpoint was not trained on the fresh preparation. Smoke scores and timings are integration diagnostics, not new detector-quality or speed results.

## Saved converted scoring

```console
python -B src/rescore_saved.py --run-dir PREDICTION_RUN_DIR --data-root DATA_ROOT --task-config reviewed_task.json --binding DATA_ROOT/binding.json --output NEW_RESCORE.json
```

The loader validates exact launch binding, config/helper snapshots, prediction log hash, unique image/method pairs and image identity. Before importing Ultralytics, the standalone rescorer sets package-owned YOLO/Matplotlib configuration paths with offline mode and autoinstall disabled. It rejects a different installed Ultralytics/Torch version or different recorded framework source hashes, then records its actual own-source hash and runtime configuration. It reuses the pinned converted scorer and preserves the run's smoke/development purpose. An explicit raw-coordinate loader is also provided; no native/reference/current-matcher evaluator is bundled or executed here.

## Validation scope and limits

The synthetic suite has 69 passing CPU controls: the 37 preparation/binding controls, 26 selected-checkpoint/completed-history/exact-resume controls and 6 installed-version/framework-source controls. They include genuinely invalid inputs and rejected stale/forged bindings; they do not train a detector.

In the observed existing environment, the actual training preflight passed without model loading or training. A historical checkpoint with an explicit freshly checked equivalence certificate then completed a one-image/four-setting CPU smoke; strict saved-output converted rescoring exactly matched the inline scores. Recorded model parameters were on CPU, and protected historical files remained unchanged. Smoke scores and times are not detector-quality or performance results. The final standalone rescorer was additionally validated with inherited runtime configuration variables removed: its package-owned paths and version/source guards passed, and the same retained outputs exactly matched earlier and inline scores. Model inference was not repeated for the rescore-only corrections. Actual execution evidence is retained privately. Fresh environment installation, prospective training to completion, full portable development inference and native/reference score reproduction remain unverified. Training determinism across hardware is not promised. This engineering addition supports inspectable reproduction; it introduces no new detector or original research method.

Read READ_FIRST.md and THIRD_PARTY_NOTICES.md before extending the component. Own code is AGPL-3.0-only; see LICENSE.


Version 0.3 root review added a last-checkpoint digest guard after the bounded integration smoke. Four targeted CPU controls check exact-state acceptance and replaced/missing/unrecorded-state rejection. The original integration and rescore records retain their actual source snapshots; no prospective training or model inference was repeated for this resume-only change.
