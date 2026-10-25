# Preparation and defence

Before presenting this study, explain shared weights within each seed, different trained checkpoints across seeds, and why 548 exposed development images do not become independent final-test data after repetition. Distinguish three training seeds, three seed 0 inference sweeps, six scoring conventions and two-seed current-matcher controls.

Work through the protocol table in RESULTS.md: define the percent/percentage-point units, cap-only rematching, legacy macro versus source weighting, published AP boundary substitution and current matcher capture. Explain why a score change with identical detections is not a detector gain, why the fixed-match prefix is a counterfactual, and why the seed 0 exact-source Octave check does not certify seeds 1/2 or exhaustive MATLAB equivalence. Read the upstream references before claiming novelty.

Reconcile seed 0 fixed-threshold size recall with aggregate AP and converted precision. Explain uncalibrated confidence, all-target matching before grouping, assignment-dependent ties, cap/NMS visibility limits and why the result has no size AP, selector utility or flight-safety conclusion.

Run the synthetic suite and aggregate-rendering script; inspect source, exported units and reproduction limitations. Raw data, labels, detections, weights and executable reference controls are absent. Fresh install/retraining remains unverified. The code and aggregate package is a bounded software study, not an accepted paper or production certification.


## Portable and retrospective selection preparation

Read [portable preparation](portable_execution/READ_FIRST.md), [execution scope](PORTABLE_VALIDATION.md) and [label-budget preparation](label_budget/READ_FIRST.md). Historical scripts and new prospective adapters are different entrypoints. Never call the bounded smoke run a new detector result or the retrospective diagnostic a population guarantee.


## Version 0.4 confirmation preparation

Read [TESTDEV_RESULTS.md](TESTDEV_RESULTS.md), the [protocol presentation](protocols/testdev_confirmation.json) and the [technical report](reports/Technical%20Report.pdf). Explain why full 1280 remains highest under the primary test-dev scorer while absolute AP falls; give the 3.712–4.421-point range only for full 1280. A same-checkpoint cross-split difference is not a paired-image observation or proof of a causal mechanism.

Keep scopes separate: 548 exposed development images and six conventions; the original 1,610-image test-dev confirmation under one primary metric; and a separate later post hoc native-policy test-dev check. Development size, current-matcher and label-budget checks remain development-only. Public labels held out from this study do not prove blind/OOD or independent-scene performance. Explain all twelve primary AP rows, six primary contrasts and the timing-drift caveat. The later sanitised protocol presents an earlier internal plan without claiming public preregistration.

The completed confirmation supports the simple full-1280 comparator within this declared task. A learned selector would require a meaningful decision failure, declared cost and fair prior-work/control analysis. The test-dev material provides aggregate/protocol evidence plus a separate [explicit-input execution component](testdev_execution/README.md). Raw inputs, weights and predictions remain user-supplied. Distinguish the complete retained-output rescoring from a new full inference run.


## Version 0.5 execution and rendering preparation

Read [the test-dev commands](testdev_execution/README.md) and [checked scope](testdev_execution/VALIDATION.md). Explain the release-time content identity, original ignored-row preservation, complete image/method matrix, source/config/checkpoint bindings and the distinction between exact historical-weight identity and actual completed prospective training. Historical weight identity does not promise identical predictions on every runtime or prove independent personal mastery.

The checked scope is 62 synthetic/CPU controls, all 1,610 real image/annotation pairs prepared and verified, 19,320 retained rows validated/rescored with twelve exact full-summary matches, and one new sorted image/four settings on CPU. The same pinned scorer is reused. No clean installation, new full inference sweep or actual prospective training was performed for this component.

The test-dev renderer reconstructs its 12 AP rows, six contrast rows, 12 cost rows and plot from bundled aggregates or six hash-bound supplied summaries. It has 17 passing controls and performs no AP recomputation or detector inference. Reconstructed timing summaries are not fresh latency measurements. Keep these software checks separate from scientific generalisation, publication novelty and robotics deployment claims.


## Version 0.6 native-policy preparation

Read [the native result](NATIVE_TESTDEV_RESULTS.md) and its [post hoc protocol](protocols/testdev_native_sensitivity.json). Explain why holding detections fixed while changing scoring can change AP and the close slicing/960 ordering. Native scoring uses category-0 union-mask filtering, global caps before selecting a class, class-image weighting and continuous precision-envelope AP. The native-minus-converted score difference is not a detector gain.

The supported finding is that full 1280 remains highest for all three checkpoints, while slicing ranks above 960 under native scoring and below it under primary converted scoring. Give the per-checkpoint margins rather than hiding the 0.035791-point close case in an average. No statistical significance or universal ordering is established.

Distinguish all twelve full-split Python-port results from the 36 annotation-selected publisher-function cases under Octave. The third fixture is a declared fallback because score-zero target rows are absent. Numerical tolerance is not a confidence interval; bounded reference agreement is not full-split or every-runtime equivalence. This update supplies aggregates and figures, not a new runnable native-scoring adapter.
