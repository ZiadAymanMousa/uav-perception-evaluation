# Saved-output cap and matching diagnostic

Frozen before executing this diagnostic, 16 October 2026. This is an exploratory explanation of the already-observed development AP change, not a prospective detector-improvement test.

Use only the saved seed-0 four-method/548-image predictions and the same curated converted ground truth. Keep the pinned Ultralytics 8.3.253 matcher and AP functions unchanged, on CPU with two torch threads. Do not run inference, access final-test data, change training or substitute an alternative official score.

For each image and IoU threshold, compare the TP/FP indicators of the retained top 500 predictions under matching with all saved predictions (up to 1,000) against matching after truncation. Record retained detections changing each way, image counts, removed TP counts and bounded examples. Recompute three dataset statistics:

1. All saved predictions and their original full-set matches. Require reproduction of the completed study AP.
2. Top 500 predictions per image, retaining their full-set matches. This is an artificial fixed-assignment diagnostic, **not** a normal cap-500 evaluation or valid alternative benchmark score.
3. Top 500 predictions per image, rematched normally. Require reproduction of the earlier cap-500 analysis.

The path from (1) to (2) measures deletion with frozen assignments, including its effect on the pinned AP interpolation. The path from (2) to (3) measures recomputed assignments on identical retained predictions and ground truth. AP is nonlinear: report a path-specific descriptive contrast, not a general causal decomposition. Never interpret an AP rise as higher detector recall without separately measured counts.

Before the real-data diagnostic, execute a one-GT/two-prediction control with confidence order opposing IoU order. Choose prediction-to-prediction IoU below the study's 0.7 NMS threshold, so the illustration is not merely a duplicate pair necessarily removed by that threshold. Verify the actual pinned matcher/AP outputs; do not assume the proposed mechanism occurs in real outputs until measured.

Capture source/input hashes, installed matching/AP source hash, all outputs and failures. The result can explain our observed protocol sensitivity; it does not establish a new theorem, toolkit bug, novel finding or publishable contribution.

## Amendment after constructed controls, before full-data diagnostic

The proposed one-GT/two-prediction reassignment control failed. The pinned two-stage unique-index ordering retained the higher-confidence/earlier detection. Its test failure, original plan/source/test and exploratory outputs are preserved under `artifacts/cap_diagnostic_initial_control_20261016` and adjacent logs. Do not report the original hypothesis as a finding.

A second measured control has two ground-truth objects, one correct high-confidence detection and one low-confidence false positive. Removing the false-positive tail leaves one matched target and recall 0.5 but changes pinned AP from 62.25% to 75%. The source appends recall 1 / precision 0 and linearly interpolates a 101-point precision curve before trapezoid integration; its area beyond achieved recall depends on the terminal precision. This is the function's convention, not a detector gain or established bug.

Revise the real-output diagnostic to retain the originally planned match checks and AP reproduction, and partition that exact 101-point AP polygon at each class/IoU's maximum attained recall. Compute the area below/at that recall and the area beyond it, including a boundary point interpolated on the same polygon. Their sum must reproduce pinned AP. Compare both terms and matched-TP counts across caps. This is descriptive integration accounting, not a new official score or proof that the constructed mechanism dominates real data.
