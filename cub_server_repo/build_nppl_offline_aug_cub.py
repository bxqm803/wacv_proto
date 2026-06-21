#!/usr/bin/env python3
"""Build NPPL/TesNet-style offline augmented CUB training images.

This script mirrors the augmentation recipe referenced by NPPL's official repo:
  1) rotation: rotate [-15, +15] degrees, then horizontal flip with p=0.5
  2) skew: magnitude=0.2, then horizontal flip with p=0.5
  3) shear: [-10, +10] degrees, then horizontal flip with p=0.5
  4) random distortion: grid 10x10, magnitude=5, then horizontal flip with p=0.5

For each class directory, each pipeline is processed 10 times. Therefore, the
augmented directory contains approximately 40 generated samples per original
training crop, matching the referenced TesNet script used by NPPL.

It first creates CUB GT-bird crops under:
  <output-root>/train_cropped/<class>/<image>.jpg
and writes augmented images to:
  <output-root>/train_cropped_augmented/<class>/

The target augmented directory intentionally does NOT copy originals by default;
this matches NPPL's train.py, which trains directly from train_cropped_augmented.

IMPORTANT FOR PART-BBOX TRAINING
--------------------------------
This script reproduces NPPL's image-only augmentation. The random-distortion
operation is non-linear, so existing GroundingDINO part boxes from the original
image are NOT geometrically aligned to these generated images. Do not plug this
folder into a part-bbox-supervised loader until a matching augmented sample/
part-box manifest is built or part supervision is disabled for these samples.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, asdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import Augmentor
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: Augmentor. Install it in your environment with:\n"
        "  pip install Augmentor\n"
    ) from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class AugConfig:
    repetitions_per_operation: int = 10
    rotate_left: int = 15
    rotate_right: int = 15
    skew_magnitude: float = 0.2
    shear_left: int = 10
    shear_right: int = 10
    distortion_grid_width: int = 10
    distortion_grid_height: int = 10
    distortion_magnitude: int = 5
    hflip_probability: float = 0.5


@dataclass(frozen=True)
class ClassTask:
    class_dir: str
    output_dir: str
    cfg_dict: Dict[str, object]
    overwrite_incomplete: bool
    seed: int


def read_kv_text(path: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, v = line.split(maxsplit=1)
            out[int(k)] = v
    return out


def read_kv_int(path: Path) -> Dict[int, int]:
    out: Dict[int, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, v = line.split()
            out[int(k)] = int(v)
    return out


def read_bboxes(path: Path) -> Dict[int, Tuple[float, float, float, float]]:
    out: Dict[int, Tuple[float, float, float, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            vals = line.strip().split()
            if vals:
                out[int(vals[0])] = tuple(float(x) for x in vals[1:5])
    return out


def crop_bbox(image: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
    x, y, w, h = bbox
    width, height = image.size
    x1 = max(0, min(width - 1, int(np.floor(x))))
    y1 = max(0, min(height - 1, int(np.floor(y))))
    x2 = max(x1 + 1, min(width, int(np.ceil(x + w))))
    y2 = max(y1 + 1, min(height, int(np.ceil(y + h))))
    return image.crop((x1, y1, x2, y2))


def image_files(directory: Path) -> List[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def build_train_crops(cub_root: Path, crop_root: Path, jpeg_quality: int, skip_existing: bool) -> List[Dict[str, object]]:
    paths = read_kv_text(cub_root / "images.txt")
    split_map = read_kv_int(cub_root / "train_test_split.txt")
    bboxes = read_bboxes(cub_root / "bounding_boxes.txt")

    train_ids = sorted(i for i in paths if split_map[i] == 1)
    records: List[Dict[str, object]] = []

    print(f"[Crop] writing {len(train_ids)} GT bird crops to: {crop_root}")
    for image_id in tqdm(train_ids, desc="Build train_cropped", ncols=120):
        relpath = Path(paths[image_id])
        out_path = crop_root / relpath
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not (skip_existing and out_path.is_file()):
            with Image.open(cub_root / "images" / relpath) as img:
                crop = crop_bbox(img.convert("RGB"), bboxes[image_id])
                crop.save(out_path, quality=jpeg_quality, optimize=False)

        records.append({
            "image_id": image_id,
            "source_relpath": relpath.as_posix(),
            "crop_relpath": relpath.as_posix(),
            "class_dir": relpath.parent.as_posix(),
            "label": int(relpath.parent.name.split(".", 1)[0]) - 1 if "." in relpath.parent.name else None,
            "bbox_xywh_raw": [float(x) for x in bboxes[image_id]],
        })

    return records


def _run_pipeline(source_dir: Path, target_dir: Path, operation: str, cfg: AugConfig) -> None:
    p = Augmentor.Pipeline(source_directory=str(source_dir), output_directory=str(target_dir))

    if operation == "rotate":
        p.rotate(
            probability=1,
            max_left_rotation=cfg.rotate_left,
            max_right_rotation=cfg.rotate_right,
        )
    elif operation == "skew":
        p.skew(probability=1, magnitude=cfg.skew_magnitude)
    elif operation == "shear":
        p.shear(
            probability=1,
            max_shear_left=cfg.shear_left,
            max_shear_right=cfg.shear_right,
        )
    elif operation == "random_distortion":
        p.random_distortion(
            probability=1.0,
            grid_width=cfg.distortion_grid_width,
            grid_height=cfg.distortion_grid_height,
            magnitude=cfg.distortion_magnitude,
        )
    else:
        raise ValueError(operation)

    p.flip_left_right(probability=cfg.hflip_probability)

    # This is intentionally the same outer loop as TesNet's img_aug.py.
    for _ in range(cfg.repetitions_per_operation):
        p.process()


def augment_one_class(task: ClassTask) -> Tuple[str, str, int, int]:
    source_dir = Path(task.class_dir)
    target_dir = Path(task.output_dir)
    cfg = AugConfig(**task.cfg_dict)

    random.seed(task.seed)
    np.random.seed(task.seed % (2**32 - 1))

    source_count = count_images(source_dir)
    if source_count == 0:
        return source_dir.name, "empty", 0, 0

    expected = source_count * 4 * cfg.repetitions_per_operation
    marker = target_dir / ".nppl_offline_aug_complete.json"

    if marker.is_file():
        try:
            meta = json.loads(marker.read_text(encoding="utf-8"))
            output_count = count_images(target_dir)
            if (
                int(meta.get("source_count", -1)) == source_count
                and int(meta.get("expected_count", -1)) == expected
                and output_count >= expected
            ):
                return source_dir.name, "skipped_complete", source_count, output_count
        except Exception:
            pass

    if target_dir.exists() and task.overwrite_incomplete:
        for p in target_dir.iterdir():
            if p.name == marker.name:
                continue
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Exact operation order and settings referenced by NPPL -> TesNet.
    for operation in ("rotate", "skew", "shear", "random_distortion"):
        _run_pipeline(source_dir, target_dir, operation, cfg)

    output_count = count_images(target_dir)
    marker.write_text(
        json.dumps(
            {
                "source_class_dir": str(source_dir),
                "source_count": source_count,
                "expected_count": expected,
                "output_count": output_count,
                "operations": ["rotate", "skew", "shear", "random_distortion"],
                "config": asdict(cfg),
                "note": "Target contains augmented samples only; originals remain in train_cropped.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    status = "done" if output_count >= expected else "count_mismatch"
    return source_dir.name, status, source_count, output_count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Create NPPL/TesNet-style offline CUB augmentation with saved image files."
    )
    p.add_argument("--cub-root", default="./data/CUB_200_2011")
    p.add_argument(
        "--output-root",
        default="./data/cub200_cropped",
        help="Creates train_cropped and train_cropped_augmented below this directory.",
    )
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--workers", type=int, default=1,
                   help="Class-level CPU workers. Use 1 to mirror the original sequential script.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-existing-crops", action="store_true")
    p.add_argument("--keep-incomplete-class", action="store_true",
                   help="Do not clear a class folder lacking a completion marker before rebuilding it.")
    p.add_argument("--crops-only", action="store_true")
    p.add_argument("--class-limit", type=int, default=0,
                   help="For a quick test, only process the first N class folders; 0 means all.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cub_root = Path(args.cub_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    crop_root = output_root / "train_cropped"
    aug_root = output_root / "train_cropped_augmented"

    required = [
        cub_root / "images",
        cub_root / "images.txt",
        cub_root / "train_test_split.txt",
        cub_root / "bounding_boxes.txt",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing CUB inputs:\n  - " + "\n  - ".join(missing))

    output_root.mkdir(parents=True, exist_ok=True)
    crop_records = build_train_crops(
        cub_root,
        crop_root,
        jpeg_quality=args.jpeg_quality,
        skip_existing=args.skip_existing_crops,
    )

    with (output_root / "train_cropped_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in crop_records:
            f.write(json.dumps(row) + "\n")

    if args.crops_only:
        print(f"[Done] Crops only: {crop_root}")
        return

    class_dirs = sorted(p for p in crop_root.iterdir() if p.is_dir())
    if args.class_limit > 0:
        class_dirs = class_dirs[:args.class_limit]

    cfg = AugConfig()
    print("[NPPL/TesNet exact offline recipe]")
    print(json.dumps(asdict(cfg), indent=2))
    print(f"[Aug] source={crop_root}")
    print(f"[Aug] target={aug_root}")
    print(f"[Aug] classes={len(class_dirs)} | expected images ≈ {len(crop_records) * 40:,}")

    tasks = [
        ClassTask(
            class_dir=str(d),
            output_dir=str(aug_root / d.name),
            cfg_dict=asdict(cfg),
            overwrite_incomplete=not args.keep_incomplete_class,
            seed=args.seed + i * 1009,
        )
        for i, d in enumerate(class_dirs)
    ]

    results: List[Tuple[str, str, int, int]] = []
    if args.workers <= 1:
        for task in tqdm(tasks, desc="Offline augment classes", ncols=120):
            results.append(augment_one_class(task))
    else:
        with Pool(processes=args.workers) as pool:
            for result in tqdm(
                pool.imap_unordered(augment_one_class, tasks),
                total=len(tasks),
                desc="Offline augment classes",
                ncols=120,
            ):
                results.append(result)

    summary = {
        "cub_root": str(cub_root),
        "train_cropped": str(crop_root),
        "train_cropped_augmented": str(aug_root),
        "num_original_train_images": len(crop_records),
        "num_classes": len(class_dirs),
        "expected_augmented_images": len(crop_records) * 40,
        "actual_augmented_images": sum(r[3] for r in results),
        "class_results": [
            {"class_dir": c, "status": s, "source_count": n_src, "output_count": n_out}
            for c, s, n_src, n_out in sorted(results)
        ],
        "augmentation_recipe": asdict(cfg),
        "matches_nppl_referenced_tesnet_script": True,
        "important": (
            "Augmented images do not have transformed GroundingDINO part boxes. "
            "Do not reuse original part boxes without a companion augmented-box pipeline."
        ),
    }
    (output_root / "nppl_offline_augmentation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n[Done]")
    print(f"Original crops: {crop_root}")
    print(f"Augmented images: {aug_root}")
    print(f"Expected approx: {summary['expected_augmented_images']:,}")
    print(f"Actual counted:  {summary['actual_augmented_images']:,}")
    print(f"Summary: {output_root / 'nppl_offline_augmentation_summary.json'}")


if __name__ == "__main__":
    main()
