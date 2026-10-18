# Result definitions and numerical exports

By Ziad Ismail

[aggregate_results.json](results/aggregate_results.json) contains the three-seed/four-setting/six-convention score cube: 72 AP values and 18 within-model convention comparisons. [Protocol CSV](results/three_seed_protocol_scores.csv) gives the same values in long form. Score units are percent; differences use percentage points (pp). All contrasts below mean **fixed slices plus global minus full 960**.

| Convention | Seed 0 difference (pp) | Seed 1 difference (pp) | Seed 2 difference (pp) |
|---|---:|---:|---:|
| Pinned converted cap1000 | -0.354219 | -0.654369 | -0.370162 |
| Pinned converted cap500 | -0.314910 | -0.627362 | -0.364737 |
| Legacy class macro cap500 | +0.096181 | -0.182065 | +0.147521 |
| Legacy source aggregate cap500 | +0.260410 | +0.028144 | +0.242340 |
| Published AP boundary cap1000 | +0.206227 | -0.045880 | +0.241516 |
| Published AP boundary cap500 | +0.117834 | -0.148905 | +0.155047 |

Pinned converted AP uses the unchanged 8.3.253 matcher/integration on the ten-class curated task. Cap 500 truncates saved score-sorted outputs and recomputes matching; it cannot recover outputs discarded by per-view NMS/caps. Legacy class macro and source aggregate use original ignore treatment, different matching/integration and a 500-output cap; their difference is aggregation weighting. They are not generic COCO AP or official challenge scores. The published-boundary-only control substitutes the exact reviewed AP function, preserving predictions/matcher; it is not a complete framework upgrade. Do not interpret the table as a complete causal decomposition.

Retained-output TP flags under cap reduction are unchanged for seed 0; seed 1 has one changed retained detection, and seed 2 has two in full 960 and one in slicing. These counts are not a new matching algorithm. An exact IoU/sort-order case explains the observed seed-1 exception in the pinned matcher. Other seed-2 cases are counted, not individually explained here.

## Captured current-matcher control: two seeds

[current_matcher_results.json](results/current_matcher_results.json) and [its CSV](results/current_matcher_scores.csv) contain 96 aggregate score rows: 2 seeds x 4 methods x 2 matchers x 3 cap modes x 2 AP integrations. The cap modes are all saved (<=1,000), first 500 retaining old matches, and first 500 rematched. The fixed-match mode is a diagnostic counterfactual, not a new detector output. Comparisons share detections and cannot be counted as independent experiments.

The control uses captured upstream matcher commit `abc2066dd6bd87f6806f73a170cf8f764ca2ebbe`, validator Git blob `9ae6fded948119f723186ca4551d73aac795cc17`, and the published AP source Git blob `2c790f77b0ee6b177332ac092a3cc70bd54d1352`. Source hashes and score-input identities appear in [provenance](results/provenance.json). The immutable capture is the definition of current in this dated control. The installed 8.3.253 package remains unchanged. No current-matcher result exists for seed 2 in this candidate.

Across all checked current-matcher cases for seeds 0 and 1, removing a lower-confidence suffix changes no retained TP flags. This concerns these saved inputs and reviewed matcher; it does not prove every floating-point/tie/runtime case. Full 1280 leads in all 24 matcher/cap-mode/integration ranking cells. Small AP changes reflect scoring semantics, not improved model predictions. Known upstream matching and integration work is cited in the README; this is a reproduction/control, not original bug discovery.

## Timing and original-size evidence

[development_operating_points.csv](results/development_operating_points.csv) contains each seed's **first-sweep** median/P95 time and pixel/pass accounting. Seed-0 three-sweep timing remains separately identified inside aggregate_results.json. Three inference sweeps are not three training seeds. One model and 548 shared development images per sweep do not support treating all image/method observations as independent training experiments. Report no stable speedup or timing-cause conclusion.

The [size summary](results/size_recall_summary.json), [all size points](results/size_recall_all_points.csv) and [size plot](results/size_recall_precision.png) remain **seed 0 only**, with the pinned matcher. Bins are (0,256), [256,1024), [1024,9216), [9216,infinity) original-pixel area, with 11,950/14,625/11,116/1,068 targets. Match against the full target population before grouping. Primary confidence/IoU are 0.25/0.50; secondary all-saved/0.50 and 0.25/0.75 are retained. Unmatched predictions have no GT area, and converted-task false positives may include visible native-ignored objects. This is recall/precision evidence, not size AP, calibrated confidence or causal miss attribution.

Full seed-0 exact-source Octave agreement covers four settings/all 548 inputs; it does not extend automatically to seeds 1/2 or every MATLAB version. The executable reference branches and original inputs are absent. Aggregates cannot regenerate benchmark scores alone. No new detector, selector, publication acceptance, final-test/generalization or autonomous-flight result is established.


## Retrospective label-budget result

The [complete diagnostic](label_budget/RESULTS.md) and [18 model/budget aggregate rows](results/label_budget/label_budget_results.csv) are a separate finite-pool selection analysis using pinned converted AP. They do not extend the native/current-matcher or seed0 size-analysis scopes. [Design and limitations](label_budget/READ_FIRST.md).
