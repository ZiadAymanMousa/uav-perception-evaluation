# Fixed local-feature geometry execution

This optional CPU adapter applies the fixed SIFT/mutual-ratio/RANSAC diagnostic to explicit image pairs. It outputs numerical summaries only. It is not a new algorithm, a same-scene classifier, a geographic crosswalk or a detector-performance experiment.

Use separately acquired data under the applicable dataset terms. The supplied `study_cases.json` names the eleven historical selected cases: four targets, a byte-identical control, three transforms of one source and three visually labelled natural controls. It contains no dataset pixels, annotations, descriptors or verified geographic labels. Selection was post hoc to the preceding image audit; two targets share an endpoint. Re-running this adapter does not turn that selection into representative sampling.

## Invocation

From the repository root, with separately acquired images arranged under `data/` and a new `geometry-results` directory as output, using Python with NumPy, OpenCV and Pillow already installed:

```sh
python -B -m geometry_execution.run --data-root data --cases geometry_execution/study_cases.json --output geometry-results
python -B -m geometry_execution.run --data-root data --cases geometry_execution/study_cases.json --output geometry-results --run
```

The first command validates all cases and selected source hashes; it does not extract features, fit a matrix or create output. `--run` explicitly starts execution. A standalone script invocation, `python -B geometry_execution/run.py`, accepts the same arguments. No download, model, training or GPU is required. Output's parent must exist, and output must be new and outside the data root. Existing files/directories/symlinks are refused. Input paths must be relative POSIX paths without `..`, absolute/drive/UNC paths or symlink escapes. Originals are read only.

CSV columns are `case_id,role,left,right,transform`, with optional `left_sha256,right_sha256`. JSON accepts the equivalent list or `{"schema":1,"cases":[...]}`. IDs must be unique simple names. Roles are `posthoc_target`, `known_exact_byte_control`, `known_transform_control`, or `visually_different_natural_control_not_verified_geography`. The last role describes a prior visual label, not verified truth. A transform case uses `scale_crop`, `rotation` or `perspective`, with a blank right field or `generated:<transform>`; the right image is generated in memory. Optional expected hashes are lower-case SHA256. An exact-byte control is checked by source-file hash.

For the supplied case file, relative sources resolve under these directories:

```text
data/
  train/images/
  val/images/
  testdev/images/
```

Official download directories may have names such as `VisDrone2019-DET-train`. You can edit the manifest's relative source paths to those existing directories under your chosen data root; there is no need to move or duplicate the originals. The supplied manifest pins SHA256 hashes for all sixteen physical source images from the historical execution. Validation rejects a different input file, including a re-encoded image. Generated right-hand controls have no source-file hash. Matching these hashes establishes selected input-byte identity; it does not establish capture location, representative sampling or identical numerical results across software builds. The frozen original execution remains separate evidence.

## Fixed method and outputs

The long edge is at most 1280 without upscaling. Actual rounded resize scales determine the pixel-centre mapping. SIFT requests 4000 features with 3 octave layers, contrast threshold 0.04, edge threshold 10, sigma 1.6; response ties can produce more features. Exact L2 two-nearest matching uses a strict 0.75 ratio in both directions and mutual agreement. A left-to-right RANSAC homography uses 3 resized destination pixels, 2000 iterations, confidence 0.995 and seed 20261027, one OpenCV thread and disabled OpenCL. These values are not CLI tuning options.

`results.json` retains every outcome, including insufficient matches and case errors; `summary.csv` provides counts and two residual summaries. `input_manifest.json` records relative names, selected source hashes before/after and cases. `settings.json` records fixed settings, source-code hashes and version numbers. `output_manifest.json` hashes the other four files. No image, annotation, descriptor, keypoint/match table, raw mask, absolute user path or machine-specific operational log is written. Invalid numeric projections become JSON null; each summary reports total and finite denominators. An execution/estimator error, nonfinite fitted matrix or changed source makes the overall state an error and CLI exit 2. Partial outputs are retained rather than silently retried.

Raw inlier rows and unique joint XY correspondence pairs are separate counts. Repeated SIFT orientations at the same locations are not deduplicated before fitting. Original-coordinate mapping is `inverse(T_right) @ H_resized @ T_left`. Forward errors use right-image pixels; backward errors use left-image pixels. Their symmetric RMS is descriptive and mixes those stated coordinate systems. Hull fraction is area divided by image width times height, not a probability. A returned matrix, high inlier fraction or tiny fitting error does not verify a scene: four point pairs can fit almost exactly. The 0.995 RANSAC parameter is not a probability of shared geography. A planar fit can fail under parallax even for related views.

Known transforms and the fixed 9x9 overlap grid check finite geometry plumbing on a single source. They are not independent acquisition scenarios. Natural visual controls are not a false-positive-rate estimate. No AP, split edits, retraining or population prevalence follows from this command.

## Validation and source credits

Generated-fixture controls are in `tests/test_geometry_execution.py`:

```sh
python -B -m unittest discover -s tests -p test_geometry_execution.py -v
```

They isolate feature extraction/matching/fitting with test doubles and check the coordinate math, prescribed transforms, unique counts, nulls, manifest/path rejection and output preservation. Passing them does not claim a fresh dataset rerun, clean installation, cross-platform numerical identity or independent human validation. Tested versions/results should be reported with the actual validation record rather than assumed from dependency names.

Methods: [Lowe's SIFT paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf), [OpenCV matching](https://docs.opencv.org/4.13.0/dc/dc3/tutorial_py_matcher.html), and [OpenCV homography](https://docs.opencv.org/4.13.0/d1/de0/tutorial_py_feature_homography.html). The original code is AGPL-3.0-only under the repository licence. OpenCV and other dependencies retain their own licences; [OpenCV 4.5+ uses Apache 2.0](https://opencv.org/license/). No upstream toolkit code is bundled here. Dataset terms are separate: the [official notice](https://aiskyeye.com/data-protection/) names VisDrone 2021 under CC BY-NC-SA 3.0; the exact local 2019 archive licence crosswalk remains qualified. This adapter redistributes no dataset image or annotation.
