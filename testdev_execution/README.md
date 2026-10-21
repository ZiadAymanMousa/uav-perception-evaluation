# Portable test-dev execution

By Ziad Ismail. This component prepares and verifies the complete 1,610-image VisDrone DET test-dev task, runs the four fixed inference settings, and validates and rescores saved predictions with the same pinned scorer. It does not include data, annotations, raw predictions or model weights.

The content pin is a release-time verification identity derived from retained data. It is not an external preregistration or a claim that this portable code executed the historical experiment. Earlier experiment records remain separate.

## Inputs and environment

Obtain the official VisDrone2019-DET-test-dev images and annotations independently under the publisher's terms. The input root must contain only the expected files inside `images` and `annotations`. Keep raw input read-only. Preparation writes a new separate directory and copies original image and annotation bytes; it does not alter the source.

Use this directory alongside the unchanged `portable_execution` component. Its conversion, geometry and scoring implementations are imported and their source files pinned. The surrounding repository's observed requirements describe the runtime; no dependency installation, weight download, training or cloud job is triggered here.

Scoring uses Ultralytics 8.3.253 and the recorded five AP/matching/prediction/NMS source hashes. NumPy, OpenCV, PyTorch and torchvision versions are checked. Same-scorer CPU rescoring accepts matching CPU-wheel base versions; complete frozen-checkpoint inference additionally pins the observed CUDA-wheel version suffixes. No clean installation or cross-platform inference identity is established by the controls below.

Preflight and saved-output scoring configure runtime caches only after validating write destinations. An optional `--cache-root` selects a writable location outside raw, prepared and baseline trees; rejected output paths do not configure or import the runtime.

Run commands from the repository root. Set `$RawTestdev`, `$PreparedTestdev`, `$Weights` and `$OutputRoot` to your actual local paths. The prepared and run destinations must be new, outside the raw tree, with output runs also outside the prepared tree. `$Weights` must be an existing local `.pt` file; loading it is a trust decision about that file's source.

## Prepare and verify all images

```powershell
python -B testdev_execution/src/testdev.py prepare --raw-root "$RawTestdev" --prepared "$PreparedTestdev" --config testdev_execution/configs/testdev_task.json
python -B testdev_execution/src/testdev.py verify --raw-root "$RawTestdev" --prepared "$PreparedTestdev" --config testdev_execution/configs/testdev_task.json
```

Both operations parse every original nonempty annotation row, including ignored regions. Malformed ignored rows are errors, not training exceptions. The reviewed totals are 1,610 images, 77,547 raw rows, 75,102 scored targets, 2,445 excluded rows, zero clips, zero duplicate removals and zero malformed-row exceptions.

Labels are class 0–9 with normalised geometry to nine decimal places, written as UTF-8 LF and decoded by the unchanged float32 scorer. Historical Windows line endings are normalised in this portable content identity. Sidecars retain excluded original field strings, source line numbers, parsed native annotations and accounting; complete original annotation bytes are also copied. Ignored regions remain visible in images and are not masked by the converted metric.

Verification rehashes raw images and annotations, reconstructs labels and sidecars, checks exact membership and counts, and compares the release-time content identity. It also checks prepared streams, byte and semantic sidecar hashes, task copies, source snapshots and binding. Paths and timestamps are excluded from content identity. Altered files cannot become trusted merely by editing a generated manifest.

## Exact frozen-checkpoint identity

The historical replay route requires the unchanged user-supplied checkpoint bytes for the chosen seed:

| Seed | SHA256 |
| --- | --- |
| seed0 | `4d6eda28e25673f6175fb9ca32ad329d2469965b263cb4068492c25a69fa98f3` |
| seed1 | `99a6f2263087473a48147437f2f57c8f9c0c934f89744d77e99061ba02c88c27` |
| seed2 | `15636cb841282eac605e8b2b97c469eed86b932717098abea2c6f8d4acddde3b` |

```powershell
python -B testdev_execution/src/testdev.py preflight --raw-root "$RawTestdev" --prepared "$PreparedTestdev" --config testdev_execution/configs/testdev_task.json --weights "$Weights" --mode frozen-replay --seed seed0
python -B testdev_execution/src/testdev.py infer --raw-root "$RawTestdev" --prepared "$PreparedTestdev" --config testdev_execution/configs/testdev_task.json --weights "$Weights" --mode frozen-replay --seed seed0 --smoke --device cpu --output "$OutputRoot/smoke-seed0"
```

`--smoke` always processes exactly the first sorted image with all four methods on CPU. It never counts as full confirmation. To deliberately execute a complete sweep, omit `--smoke`, use a new output directory and explicitly select an available device. No complete sweep is run by the test suite.

```powershell
python -B testdev_execution/src/testdev.py infer --raw-root "$RawTestdev" --prepared "$PreparedTestdev" --config testdev_execution/configs/testdev_task.json --weights "$Weights" --mode frozen-replay --seed seed0 --device 0 --output "$OutputRoot/complete-seed0"
python -B testdev_execution/src/testdev.py verify-run --raw-root "$RawTestdev" --prepared "$PreparedTestdev" --config testdev_execution/configs/testdev_task.json --weights "$Weights" --mode frozen-replay --seed seed0 --run "$OutputRoot/complete-seed0"
```

Repeat separately with the matching seed1/seed2 weights if all three identities are available. The pinned single-GPU selector records the requested device index while mapping the selected visible GPU to logical `cuda:0`; a logical device name alone is not proof of physical GPU identity. Run verification checks CPU/CUDA consistency, the fixed prediction arguments and the recorded runtime versions and source hashes.

Matching a checkpoint hash identifies the historical weights; it does not mean this component trained them or guarantee bit-identical GPU predictions on different hardware. Without those weights, the original numbers cannot be claimed to have been recovered.

## New prospective completed baselines

The separate `portable_execution` training and completion guards remain unchanged, including its historical-certificate smoke restrictions. A new test-dev run may use a genuinely completed prospective baseline, checked against its own original training data/config/binding. That training context must be supplied separately; test-dev data are never substituted for training provenance.

After actual baseline completion, set `$BaselineRun`, `$TrainingData`, `$TrainingTask` and `$TrainingBinding` to the verified training inputs. Use its selected `best.pt` as `$Weights`:

```powershell
python -B testdev_execution/src/testdev.py infer --raw-root "$RawTestdev" --prepared "$PreparedTestdev" --config testdev_execution/configs/testdev_task.json --weights "$Weights" --mode prospective-completed-baseline --baseline-run "$BaselineRun" --training-data "$TrainingData" --training-config "$TrainingTask" --training-binding "$TrainingBinding" --device 0 --output "$OutputRoot/prospective-testdev"
```

This route checks actual completed status, epoch history, selected checkpoint, frozen config and helper snapshots through `completed_checkpoint`. It does not accept a historical smoke certificate and cannot use historical seed labels. A fresh training run is a prospective result, even if settings match; it does not recover the earlier table by association. `verify-run` takes the same separate training arguments.

## Saved-log rescoring

For a supplied saved log, set `$SavedPredictions` to its `predictions_and_timings.jsonl` and use the applicable checkpoint context:

```powershell
python -B testdev_execution/src/testdev.py rescore-log --raw-root "$RawTestdev" --prepared "$PreparedTestdev" --config testdev_execution/configs/testdev_task.json --weights "$Weights" --mode frozen-replay --seed seed0 --predictions "$SavedPredictions" --output "$OutputRoot/seed0-rescore.json"
```

This validates every image/method row before applying the same pinned scorer. It does not independently certify that an external log was generated by those weights, verify the original inference runtime by association, or constitute an independent evaluator. `verify-run` additionally verifies this component's retained run chain and compares a fresh same-scorer summary with its saved summary. It uses the caller-verified prepared directory, so valid relocation does not depend on an obsolete launch path.

A complete checkpoint log requires exactly 6,440 rows in the rotating image/method order, one row per image/method, 1,610 passes for each full-frame setting and 11,448 sliced/global passes. The 15 warm-up calls are recorded separately from scored passes. Missing/duplicate rows, bad geometry, nonfinite or unsorted confidence, cap violations, inconsistent windows/timing/counts and false completion state are rejected. Summaries retain the original AP and diagnostic timing boundaries.

## CPU controls and limits

```powershell
python -B -m unittest discover -s testdev_execution/tests -v
```

62 synthetic/CPU controls passed, including a real pinned-scorer perfect-box positive control and 15 targeted controls for config consistency, recorded policy/runtime/device provenance, relocation, per-view caps and safe write ordering. The suite tests input/source/config/checkpoint changes, malformed ignored rows, conversion/accounting, complete versus smoke scope, matrix integrity, and completion guards. Its prospective-route control verifies use of a separate training context and propagation of the existing incomplete-baseline rejection; it does not create or run new training.

On 21 October 2026, complete real-data preparation and verification also passed for all 1,610 images and 75,102 scored targets. All three retained checkpoint logs were validated and rescored: each contained 6,440 image/method rows, and all twelve complete method-summary objects matched the historical saved summaries exactly. This was same-pinned-scorer rescoring of existing predictions, not a new inference sweep or an independent evaluator. A separate seed0 CPU integration run processed the first sorted image with all four settings, and its saved run verification passed. See [the checked scope and limitations](VALIDATION.md).

These checks do not establish a clean installation, a complete 1,610-image inference run through this new component, real prospective training, stable latency, native VisDrone ignore-region scoring, external peer review or an independent evaluator. Aggregate figures elsewhere in the repository are a presentation layer, not a substitute for those executions.
