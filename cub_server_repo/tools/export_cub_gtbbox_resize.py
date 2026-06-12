#!/usr/bin/env python3
"""Export CUB-200-2011 GT-bbox crops warped directly to a fixed square size.

Geometry:
    raw CUB image
    -> crop with bounding_boxes.txt (x, y, width, height)
    -> direct bicubic resize to SIZE x SIZE
    -> save under OUTPUT_ROOT/images/<original relative path>

The original relative paths and official train/test split are preserved so the
saved images can be consumed by both DINOv2 and GroundingDINO builders.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def read_kv_text(path: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                key, value = line.split(maxsplit=1)
                out[int(key)] = value
    return out


def read_kv_int(path: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                key, value = line.split()
                out[int(key)] = int(value)
    return out


def read_bboxes(path: str) -> Dict[int, Tuple[float, float, float, float]]:
    out: Dict[int, Tuple[float, float, float, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            if values:
                out[int(values[0])] = tuple(float(x) for x in values[1:5])
    return out


@dataclass(frozen=True)
class Sample:
    image_id: int
    relpath: str
    is_train: bool
    bbox: Tuple[float, float, float, float]


def build_samples(cub_root: str) -> List[Sample]:
    paths = read_kv_text(os.path.join(cub_root, "images.txt"))
    split_map = read_kv_int(os.path.join(cub_root, "train_test_split.txt"))
    boxes = read_bboxes(os.path.join(cub_root, "bounding_boxes.txt"))
    return [
        Sample(
            image_id=image_id,
            relpath=paths[image_id],
            is_train=split_map[image_id] == 1,
            bbox=boxes[image_id],
        )
        for image_id in sorted(paths)
    ]


def crop_bbox(image: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
    x, y, width, height = bbox
    image_width, image_height = image.size
    x1 = max(0, min(image_width - 1, int(math.floor(x))))
    y1 = max(0, min(image_height - 1, int(math.floor(y))))
    x2 = max(x1 + 1, min(image_width, int(math.ceil(x + width))))
    y2 = max(y1 + 1, min(image_height, int(math.ceil(y + height))))
    return image.crop((x1, y1, x2, y2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop CUB images with official GT bboxes and directly resize them to a square."
    )
    parser.add_argument("--cub-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cub_root = os.path.abspath(os.path.expanduser(args.cub_root))
    output_root = os.path.abspath(os.path.expanduser(args.output_root))
    output_images = os.path.join(output_root, "images")

    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if not os.path.isdir(os.path.join(cub_root, "images")):
        raise FileNotFoundError(os.path.join(cub_root, "images"))

    samples = build_samples(cub_root)
    if args.split == "train":
        samples = [sample for sample in samples if sample.is_train]
    elif args.split == "test":
        samples = [sample for sample in samples if not sample.is_train]

    os.makedirs(output_images, exist_ok=True)
    train_relpaths: List[str] = []
    test_relpaths: List[str] = []
    written = 0
    skipped = 0

    for sample in tqdm(samples, desc=f"Export GT-bbox warp{args.image_size}", ncols=120):
        source_path = os.path.join(cub_root, "images", sample.relpath)
        target_path = os.path.join(output_images, sample.relpath)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        if sample.is_train:
            train_relpaths.append(sample.relpath)
        else:
            test_relpaths.append(sample.relpath)

        if os.path.isfile(target_path) and not args.overwrite:
            skipped += 1
            continue

        with Image.open(source_path) as image:
            image = image.convert("RGB")
            image = crop_bbox(image, sample.bbox)
            image = image.resize(
                (args.image_size, args.image_size),
                resample=Image.Resampling.BICUBIC,
            )
            # CUB files are JPEG. Preserve the original relative filename while
            # writing a deterministic, high-quality RGB JPEG.
            image.save(target_path, format="JPEG", quality=args.quality, subsampling=0)
        written += 1

    metadata = {
        "source_cub_root": cub_root,
        "output_root": output_root,
        "image_size": args.image_size,
        "geometry": "CUB ground-truth bbox crop, then direct square resize; no padding; no center crop",
        "resample": "PIL.Image.Resampling.BICUBIC",
        "jpeg_quality": args.quality,
        "num_selected": len(samples),
        "num_written": written,
        "num_skipped": skipped,
    }
    with open(os.path.join(output_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_root, "train_relpaths.json"), "w", encoding="utf-8") as f:
        json.dump(train_relpaths, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_root, "test_relpaths.json"), "w", encoding="utf-8") as f:
        json.dump(test_relpaths, f, ensure_ascii=False, indent=2)

    print(f"[Done] output={output_root}")
    print(f"[Images] written={written} skipped={skipped} selected={len(samples)}")
    print(f"[Geometry] GT bbox crop -> direct resize {args.image_size}x{args.image_size}")


if __name__ == "__main__":
    main()
