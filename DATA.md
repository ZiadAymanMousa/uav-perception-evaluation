# Data and model inputs

The study uses 6,471 official VisDrone2019-DET training images and 548 validation images. Validation is exposed development data, including checkpoint selection. A later frozen confirmation uses all 1,610 official test-dev images with existing checkpoints and unchanged settings; it does not change training or development-selection provenance. Source: [official dataset repository](https://github.com/VisDrone/VisDrone-Dataset).

No general dataset redistribution grant was established in this review. This package contains no images, crops, annotations, converted labels, per-GT records or raw predictions. Obtain inputs from the official source and follow its applicable conditions. A public download link is not a redistribution licence.

Training ZIP SHA256: 86a77eba93137bfc16e4993860de9245b0675c0dba0d3ab98fb458699e256f84. Validation ZIP SHA256: abeea063037e5d20398837deb11084e652402a34ddf4f207bdf541a6f2a35ef9. The curated task includes 38,759 validation targets from native scored categories 1-10 mapped to 0-9. Excluded/ignored categories follow the separately declared converted policy; native ignore semantics are not imported.

Original pretrained checkpoint SHA256: 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1. Selected seed 0 checkpoint SHA256: 4d6eda28e25673f6175fb9ca32ad329d2469965b263cb4068492c25a69fa98f3. Neither file is distributed. Model licensing and dataset rights are separate; do not assume trained weights are unencumbered.


Version 0.2 adds completed checkpoint identities without distributing weights: fresh seed 1 SHA256 `99a6f2263087473a48147437f2f57c8f9c0c934f89744d77e99061ba02c88c27`; seed 2 SHA256 `15636cb841282eac605e8b2b97c469eed86b932717098abea2c6f8d4acddde3b`. All three share the original train/validation data. Size-specific exports still concern only seed 0.


## Test-dev confirmation data

The [official test-dev archive](https://drive.google.com/file/d/1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V/view), linked by the publisher, is 311,045,829 bytes with SHA256 `78b0c5078a14ee43d0b803a354e76016d7260d1704cfd1c2dc821858d839e261`. All 1,610 image/annotation pairs were audited and converted: 77,547 native rows, 75,102 scored targets and 2,445 exclusions; no new malformed rows, clips or duplicate removals. [Class counts and identity hashes](results/testdev/data_integrity_summary.json). The primary metric still omits native ignore masking.

Labels are public but were held out from this study's development decisions. No exact-byte cross-split duplicate image group was found. Near duplicates, shared scenes and pretraining exposure remain unresolved; no verified scene-map metadata establishes independent groups. This is not a blind challenge, OOD benchmark or external validation. No test-dev raw data, converted labels, sidecars, predictions or weights are redistributed. Hashes identify retained evidence without granting rights to it.
