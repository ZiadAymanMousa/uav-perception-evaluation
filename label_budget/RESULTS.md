# Proposed portfolio addition

Using saved predictions from three trained checkpoints, a retrospective diagnostic compared uniform image-label budgets across 100 fixed query permutations per checkpoint. The full-pool AP winner was recovered in 48-66 of 100 draws at 16 images, 69-85 at 64, and 93-100 at 256. No missing-class abstentions occurred.

Rare errors remain visible: for checkpoint seed 2, one 16-image draw selected full 640 and lost 4.455 AP percentage points on the full pool, despite that group's mean regret 0.377 and p90 0.827. Mean or p90 alone would hide this tail.

These are conditional observations on 548 previously exposed development images under one frozen converted scoring convention. They establish neither label requirements nor generalisation, annotation savings or a new method. Full 1280 is development-informed; its zero additional-query regret is not independent label-free discovery. Full 548 is one unique subset per checkpoint, not 100 independent experiments. The next step is to confirm the frozen choices on test-dev, which this study has not used.
