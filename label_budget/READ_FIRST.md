# Retrospective label-budget diagnostic

By Ziad Ismail. This asks how pooled AP on a uniform image subset chooses among four fixed inference settings. The primary checkpoint is seed0; seed1/2 repeat the same frozen design. Read PROTOCOL.json and RESULTS.md. The same 100 nested masks use 16/32/64/128/256/548 images from the 548-image exposed development pool.

The core computes pooled AP from revealed per-image matching statistics. It abstains if any of ten GT classes is absent, preserves canonical image order for score ties, and uses the declared exact tie rule. The full-pool oracle evaluates the selected choice afterwards. Regret is oracle AP minus selected-setting AP, in percentage points. Fixed 1280 is informed by prior full-pool development results, not an independently discovered label-free solution.

The 100 draws describe query variability, not independent training runs or new scenes. Full 548 is one unique subset per checkpoint. GT box counts are not annotation time. There is no population guarantee, practical epsilon threshold, active acquisition method or new estimator. The evidence does not establish a need for a learned selector.

Run synthetic decision/pooled-AP controls from this directory using the pinned observed environment:

```console
python -B -m unittest test_selector_controls -v
```

The package provides the selection core, synthetic controls, exact masks and aggregate full-grid results. Source hashes in the protocol identify actual historical execution bytes; packaged copies add licence comments with equivalent ASTs. Private saved detections, ground truth sufficient-statistic caches and the historical-input full-grid orchestration are not bundled. These controls cannot recompute the real full-grid AP results from this release alone. No dataset or weights are distributed. This is application/technical-report evidence, not peer review.

Own source is AGPL-3.0-only. Ultralytics 8.3.253 supplies AP/matching in the installed environment; upstream rights and notices remain. No framework source is vendored here. See ../THIRD_PARTY_NOTICES.md.
