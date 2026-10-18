# Notices and scope

Own source is AGPL-3.0-only, copyright 2026 Ziad Ismail. The audit, row conversion, curation and semantic identity helpers retain the reviewed implementation. Training/inference/review adapters derive from the separately preserved own-source software review candidate 0.2.0. Their changes are versioned here; historical experiment bytes are not replaced.

Ultralytics YOLO11 and its installed 8.3.253 framework provide training, prediction, matching and AP functions under their applicable upstream terms. This package does not vendor the framework, its starting/trained weights or its runtime. See https://github.com/ultralytics/ultralytics/tree/v8.3.253 . NumPy, Pillow, PyYAML, PyTorch, torchvision, OpenCV, psutil and Matplotlib are external dependencies with their respective upstream licences.

VisDrone authors own the original dataset and annotations. Obtain them independently and comply with the publisher's terms: https://github.com/VisDrone/VisDrone-Dataset . No dataset, converted image/label, raw detection or checkpoint is supplied. No official challenge or COCO equivalence is claimed.

No captured publisher/native evaluator or isolated current-matcher implementation is included. Historical aggregate evidence elsewhere in a release does not imply those branches are executable here.
