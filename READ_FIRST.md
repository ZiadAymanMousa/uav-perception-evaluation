# Preparation and defence

Before presenting this study, explain shared weights within each seed, different trained checkpoints across seeds, and why 548 exposed development images do not become independent final-test data after repetition. Distinguish three training seeds, three seed 0 inference sweeps, six scoring conventions and two-seed current-matcher controls.

Work through the protocol table in RESULTS.md: define the percent/percentage-point units, cap-only rematching, legacy macro versus source weighting, published AP boundary substitution and current matcher capture. Explain why a score change with identical detections is not a detector gain, why the fixed-match prefix is a counterfactual, and why the seed 0 exact-source Octave check does not certify seeds 1/2 or exhaustive MATLAB equivalence. Read the upstream references before claiming novelty.

Reconcile seed 0 fixed-threshold size recall with aggregate AP and converted precision. Explain uncalibrated confidence, all-target matching before grouping, assignment-dependent ties, cap/NMS visibility limits and why the result has no size AP, selector utility or flight-safety conclusion.

Run the synthetic suite and aggregate-rendering script; inspect source, exported units and reproduction limitations. Raw data, labels, detections, weights and executable reference controls are absent. Fresh install/retraining remains unverified. The code and aggregate package is a bounded software study, not an accepted paper or production certification.


## Portable and retrospective selection preparation

Read [portable preparation](portable_execution/READ_FIRST.md), [execution scope](PORTABLE_VALIDATION.md) and [label-budget preparation](label_budget/READ_FIRST.md). Historical scripts and new prospective adapters are different entrypoints. Never call the bounded smoke run a new detector result or the retrospective diagnostic a population guarantee.


## Version 0.4 confirmation preparation

Read [TESTDEV_RESULTS.md](TESTDEV_RESULTS.md), the [protocol presentation](protocols/testdev_confirmation.json) and the [technical report](reports/Technical%20Report.pdf). Explain why full 1280 remains highest under the primary test-dev scorer while absolute AP falls; give the 3.712–4.421-point range only for full 1280. A same-checkpoint cross-split difference is not a paired-image observation or proof of a causal mechanism.

Keep scopes separate: 548 exposed development images/six conventions; 1,610 official test-dev images/one primary metric; development-only size, matcher, reference and label-budget checks. Public labels held out from this study do not prove blind/OOD or independent-scene performance. Explain all twelve AP rows, six contrasts and the timing-drift caveat. The later sanitised protocol presents an earlier internal plan without claiming public preregistration.

The completed confirmation supports the simple full-1280 comparator within this declared task. A learned selector would require a meaningful decision failure, declared cost and fair prior-work/control analysis. The test-dev material provides aggregate and protocol evidence; the execution adapter and raw inputs are outside the distributed reproduction scope.
