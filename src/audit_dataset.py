# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Ziad Ismail
"""Audit VisDrone DET files without silently converting away ignore annotations.

Infrastructure utility, not a research result or official evaluator.
Only byte-identical image leakage is detected; scene independence is not inferred.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_annotation(row: list[str]) -> dict[str, int]:
    """Parse the eight-field ground-truth format, keeping all categories."""
    fields = [v.strip() for v in row]
    # Official files may contain a final comma; do not accept arbitrary extra data.
    if len(fields) == 9 and fields[-1] == "":
        fields.pop()
    if len(fields) != 8:
        raise ValueError(f"expected 8 fields, got {len(fields)}")
    try:
        values = [int(v) for v in fields]
    except ValueError as exc:
        raise ValueError("ground-truth fields must be integers") from exc
    names = ("left", "top", "width", "height", "score", "category", "truncation", "occlusion")
    a = dict(zip(names, values))
    if a["width"] <= 0 or a["height"] <= 0:
        raise ValueError("nonpositive box dimensions")
    if a["score"] not in (0, 1):
        raise ValueError("ground-truth score must be 0 or 1")
    if not 0 <= a["category"] <= 11:
        raise ValueError("category must be in 0..11")
    if a["truncation"] not in (0, 1):
        raise ValueError("truncation must be 0 or 1")
    if a["occlusion"] not in (0, 1, 2):
        raise ValueError("occlusion must be 0, 1 or 2")
    return a


def audit(splits: dict[str, Path]) -> dict:
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "data audit only; not detection evaluation",
        "limitations": ["exact-byte duplicates only", "no scene-independence inference",
                        "no AP calculation or official ignore-region evaluation"],
        "splits": {}, "issues": [], "cross_split_duplicates": [],
    }
    image_hashes = defaultdict(list)

    def issue(level: str, split: str, path: str, detail: str) -> None:
        report["issues"].append(dict(level=level, split=split, path=path, detail=detail))

    for label, root in sorted(splits.items()):
        root = root.resolve()
        image_dir, annotation_dir = root / "images", root / "annotations"
        stats = Counter()
        classes = Counter()
        manifest = []
        report["splits"][label] = dict(root=str(root), counts={}, class_counts={}, manifest=manifest)
        if not image_dir.is_dir() or not annotation_dir.is_dir():
            issue("error", label, str(root), "expected images and annotations directories")
            continue
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        annotations = {p.stem: p for p in annotation_dir.glob("*.txt")}
        stems = Counter(p.stem for p in images)
        if not images:
            issue("error", label, "images", "no supported images found")
        for stem, count in stems.items():
            if count > 1:
                issue("error", label, stem, "multiple image files share an annotation stem")
        for stem, ann in annotations.items():
            if stem not in stems:
                issue("error", label, ann.name, "annotation has no matching image")
        for path in images:
            stats["images"] += 1
            rel = str(path.relative_to(root)).replace("\\", "/")
            record = {"image": rel}
            manifest.append(record)
            try:
                record["image_sha256"] = digest(path)
                image_hashes[record["image_sha256"]].append({"split": label, "image": rel})
                with Image.open(path) as im:
                    width, height = im.size
                    im.verify()
                record["width"], record["height"] = width, height
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                issue("error", label, rel, f"image unreadable: {type(exc).__name__}")
                continue
            ann = annotations.get(path.stem)
            if ann is None:
                issue("error", label, rel, "image has no matching annotation")
                continue
            try:
                record["annotation_sha256"] = digest(ann)
                with ann.open(encoding="utf-8-sig", newline="") as f:
                    for line_number, row in enumerate(csv.reader(f), 1):
                        if not row or all(not field.strip() for field in row):
                            continue
                        try:
                            a = parse_annotation(row)
                        except ValueError as exc:
                            issue("error", label, f"{ann.name}:{line_number}", str(exc))
                            stats["malformed_rows"] += 1
                            continue
                        stats["annotation_rows"] += 1
                        classes[str(a["category"])] += 1
                        if a["category"] == 0:
                            stats["ignore_regions"] += 1
                        elif a["category"] == 11:
                            stats["other_category_rows"] += 1
                        elif a["score"] == 0:
                            stats["ignored_scored_objects"] += 1
                        else:
                            stats["scored_target_objects"] += 1
                            if a["width"] * a["height"] < 32 ** 2:
                                stats["target_area_below_1024_pixels"] += 1
                        if (a["left"] < 0 or a["top"] < 0 or
                            a["left"] + a["width"] > width or a["top"] + a["height"] > height):
                            issue("warning", label, f"{ann.name}:{line_number}", "box extends outside image; inspect, do not silently clip")
            except (OSError, UnicodeError) as exc:
                issue("error", label, ann.name, f"annotation unreadable: {type(exc).__name__}")
        report["splits"][label]["counts"] = dict(sorted(stats.items()))
        report["splits"][label]["class_counts"] = dict(sorted(classes.items(), key=lambda x: int(x[0])))

    for sha, entries in sorted(image_hashes.items()):
        if len({entry["split"] for entry in entries}) > 1:
            report["cross_split_duplicates"].append({"sha256": sha, "entries": entries})
            issue("error", "multiple", sha, "byte-identical image present in multiple splits")
    # Hash only relative content records; timestamps and machine-specific roots do not affect it.
    manifests = {name: item["manifest"] for name, item in report["splits"].items()}
    report["manifest_sha256"] = hashlib.sha256(json.dumps(manifests, sort_keys=True).encode()).hexdigest()
    report["error_count"] = sum(i["level"] == "error" for i in report["issues"])
    report["warning_count"] = sum(i["level"] == "warning" for i in report["issues"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", required=True, metavar="NAME=ROOT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits = {}
    for value in args.split:
        if "=" not in value:
            parser.error("each --split must be NAME=ROOT")
        name, root = value.split("=", 1)
        if not name or not root or name in splits:
            parser.error("split names/roots must be nonempty and split names unique")
        splits[name] = Path(root)
    report = audit(splits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "errors": report["error_count"],
                      "warnings": report["warning_count"], "manifest_sha256": report["manifest_sha256"]}))
    return 1 if report["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
