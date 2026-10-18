# Preparation and defence

Before presenting this study, explain shared weights within each seed, different trained checkpoints across seeds, and why 548 exposed development images do not become independent final-test data after repetition. Distinguish three training seeds, three seed 0 inference sweeps, six scoring conventions and two-seed current-matcher controls.

Work through the protocol table in RESULTS.md: define the percent/percentage-point units, cap-only rematching, legacy macro versus source weighting, published AP boundary substitution and current matcher capture. Explain why a score change with identical detections is not a detector gain, why the fixed-match prefix is a counterfactual, and why the seed 0 exact-source Octave check does not certify seeds 1/2 or exhaustive MATLAB equivalence. Read the upstream references before claiming novelty.

Reconcile seed 0 fixed-threshold size recall with aggregate AP and converted precision. Explain uncalibrated confidence, all-target matching before grouping, assignment-dependent ties, cap/NMS visibility limits and why the result has no size AP, selector utility or flight-safety conclusion.

Run the synthetic suite and aggregate-rendering script; inspect source, exported units and reproduction limitations. Raw data, labels, detections, weights and executable reference controls are absent. Fresh install/retraining remains unverified. The code and aggregate package is a bounded software study, not an accepted paper or production certification.


## Version0.3 preparation

Read [portable preparation](portable_execution/READ_FIRST.md), [execution scope](PORTABLE_VALIDATION.md) and [label-budget preparation](label_budget/READ_FIRST.md). Historical scripts and new prospective adapters are different entrypoints. Never call the bounded smoke run a new detector result or the retrospective diagnostic a population guarantee.
