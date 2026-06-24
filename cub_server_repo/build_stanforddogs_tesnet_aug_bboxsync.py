#!/usr/bin/env python3
"""Offline Stanford Dogs TesNet-style augmentation with synchronized part boxes.

Policy reproduced from TesNet's preprocess_data/img_aug.py:
  - 10 random rotations in [-15, 15] degrees, each followed by HFlip(p=0.5)
  - 10 random skews with magnitude 0.2, each followed by HFlip(p=0.5)
  - 10 random x-shears in [-10, 10] degrees, each followed by HFlip(p=0.5)
  - 10 random 10x10 grid distortions with magnitude 5, each followed by HFlip(p=0.5)

Geometry rule:
  - original / rotation / shear: part boxes are transformed by the exact same
    affine matrix as the image.
  - skew / random_distortion: boxes are intentionally set to invalid (-1),
    including after the optional horizontal flip, as requested.

All images start from:
  original image -> official Stanford Dogs union dog bbox -> direct resize(224, 224)
and all saved part boxes are therefore in 224x224 crop coordinates.

Outputs:
  <output-root>/
    train_cropped/<class>/<stem>.jpg
    train_cropped_augmented/<class>/<stem>_<variant>.jpg
    box_shards/shard_00000.pt
    ...
    train_tesnet_aug_manifest.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.io import loadmat
from tqdm import tqdm


DEFAULT_PARTS = ("head", "ear", "muzzle", "body", "leg", "tail")


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def safe_torch_load(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(payload: Dict[str, Any], path: str) -> None:
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def pil_resample_bicubic() -> int:
    return getattr(Image.Resampling, "BICUBIC", Image.BICUBIC)


def pil_affine_mode() -> int:
    return getattr(Image.Transform, "AFFINE", Image.AFFINE)


def pil_perspective_mode() -> int:
    return getattr(Image.Transform, "PERSPECTIVE", Image.PERSPECTIVE)


def pil_mesh_mode() -> int:
    return getattr(Image.Transform, "MESH", Image.MESH)


# -----------------------------------------------------------------------------
# Stanford Dogs metadata
# -----------------------------------------------------------------------------
def matlab_string(value: Any) -> str:
    current = value
    while isinstance(current, np.ndarray):
        if current.size == 0:
            raise ValueError("Encountered empty MATLAB string.")
        current = current.item() if current.size == 1 else current.flat[0]

    if isinstance(current, bytes):
        return current.decode("utf-8")
    return str(current)


def parse_split_mat(path: str) -> List[Tuple[str, int]]:
    data = loadmat(path)
    if "annotation_list" not in data or "labels" not in data:
        raise KeyError(f"{path} must include annotation_list and labels.")

    annotations = data["annotation_list"].squeeze()
    labels = np.asarray(data["labels"]).squeeze()
    if len(annotations) != len(labels):
        raise RuntimeError(f"Split mismatch in {path}: {len(annotations)} vs {len(labels)}.")

    return [
        (matlab_string(annotation).replace("\\", "/"), int(np.asarray(label).item()) - 1)
        for annotation, label in zip(annotations, labels)
    ]


def parse_annotation_union_box(path: str) -> Tuple[int, int, int, int]:
    root = ET.parse(path).getroot()
    boxes: List[Tuple[int, int, int, int]] = []

    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = int(float(bndbox.findtext("xmin", "0")))
        ymin = int(float(bndbox.findtext("ymin", "0")))
        xmax = int(float(bndbox.findtext("xmax", "0")))
        ymax = int(float(bndbox.findtext("ymax", "0")))

        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))

    if not boxes:
        raise RuntimeError(f"No valid dog bbox in {path}")

    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def crop_xyxy(image: Image.Image, box: Sequence[int]) -> Image.Image:
    x1, y1, x2, y2 = [int(v) for v in box]
    image_w, image_h = image.size

    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(x1 + 1, min(image_w, x2))
    y2 = max(y1 + 1, min(image_h, y2))

    return image.crop((x1, y1, x2, y2))


# -----------------------------------------------------------------------------
# Box-cache loading
# -----------------------------------------------------------------------------
def valid_xyxy(raw: Any) -> np.ndarray:
    if raw is None:
        return np.empty((0, 4), dtype=np.float32)

    if isinstance(raw, torch.Tensor):
        array = raw.detach().cpu().float().numpy()
    else:
        array = np.asarray(raw, dtype=np.float32)

    if array.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    if array.ndim == 1 and array.size == 4:
        array = array.reshape(1, 4)
    elif array.ndim >= 2 and array.shape[-1] == 4:
        array = array.reshape(-1, 4)
    else:
        return np.empty((0, 4), dtype=np.float32)

    keep = (
        np.isfinite(array).all(axis=1)
        & (array[:, 0] >= 0)
        & (array[:, 1] >= 0)
        & (array[:, 2] > array[:, 0])
        & (array[:, 3] > array[:, 1])
    )
    return array[keep].astype(np.float32, copy=False)


def load_source_boxes(
    cache_path: str,
    samples: Sequence[Dict[str, Any]],
    parts: Sequence[str],
    image_size: int,
    allow_incomplete: bool,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    payload = safe_torch_load(cache_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), dict):
        raise TypeError(f"Part cache must contain a records dict: {cache_path}")

    meta = payload.get("meta", {})
    records = payload["records"]

    expected = int(meta.get("expected_images", 0))
    processed = int(meta.get("processed_images", len(records)))
    if not allow_incomplete and (expected <= 0 or processed < expected):
        raise RuntimeError(
            f"GDINO part cache is incomplete: processed={processed}, expected={expected}. "
            "Finish GDINO first, or pass --allow-incomplete-boxes."
        )

    declared_parts = {
        str(item.get("name", "")).lower()
        for item in meta.get("parts", [])
        if isinstance(item, dict)
    }
    missing_parts = [part for part in parts if part not in declared_parts]
    if missing_parts:
        raise KeyError(f"GDINO cache misses requested parts: {missing_parts}.")

    source_resize = int(meta.get("resized_coordinate_size", image_size))
    scale = float(image_size) / max(1.0, float(source_resize))

    all_per_sample: List[List[np.ndarray]] = []
    max_boxes = 1
    missing_records = 0

    for sample in samples:
        record = records.get(sample["rel_annotation"])
        image_boxes: List[np.ndarray] = []

        if not isinstance(record, dict):
            missing_records += 1
            image_boxes = [np.empty((0, 4), dtype=np.float32) for _ in parts]
        else:
            part_records = record.get("parts", {})
            for part in parts:
                entry = part_records.get(part, {}) if isinstance(part_records, dict) else {}
                boxes = valid_xyxy(entry.get("boxes_xyxy_resize"))
                if scale != 1.0 and boxes.size:
                    boxes = boxes * scale
                image_boxes.append(boxes)
                max_boxes = max(max_boxes, int(boxes.shape[0]))

        all_per_sample.append(image_boxes)

    boxes_tensor = np.full(
        (len(samples), len(parts), max_boxes, 4),
        -1.0,
        dtype=np.float32,
    )

    for sample_index, image_boxes in enumerate(all_per_sample):
        for part_index, boxes in enumerate(image_boxes):
            if boxes.size == 0:
                continue
            count = min(max_boxes, boxes.shape[0])
            boxes_tensor[sample_index, part_index, :count] = boxes[:count]

    summary = {
        "expected_images": expected,
        "processed_images": processed,
        "missing_records": missing_records,
        "source_resize": source_resize,
        "max_boxes": max_boxes,
    }
    return boxes_tensor, max_boxes, summary


# -----------------------------------------------------------------------------
# Affine geometry, synchronized with part boxes
# -----------------------------------------------------------------------------
def translation(tx: float, ty: float) -> np.ndarray:
    return np.array(
        [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_matrix(angle_deg: float, width: int, height: int) -> np.ndarray:
    angle_rad = math.radians(float(angle_deg))
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0

    rotate = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return translation(cx, cy) @ rotate @ translation(-cx, -cy)


def x_shear_matrix(shear_deg: float, width: int, height: int) -> np.ndarray:
    tangent = math.tan(math.radians(float(shear_deg)))
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0

    shear = np.array(
        [[1.0, tangent, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return translation(cx, cy) @ shear @ translation(-cx, -cy)


def hflip_matrix(width: int) -> np.ndarray:
    return np.array(
        [[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def apply_affine_image(
    image: Image.Image,
    source_to_output: np.ndarray,
    fill_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    inverse = np.linalg.inv(source_to_output)
    coefficients = (
        float(inverse[0, 0]),
        float(inverse[0, 1]),
        float(inverse[0, 2]),
        float(inverse[1, 0]),
        float(inverse[1, 1]),
        float(inverse[1, 2]),
    )
    return image.transform(
        image.size,
        pil_affine_mode(),
        coefficients,
        resample=pil_resample_bicubic(),
        fillcolor=fill_color,
    )


def apply_affine_boxes(
    boxes: np.ndarray,
    source_to_output: np.ndarray,
    image_size: int,
) -> np.ndarray:
    """Transform axis-aligned xyxy boxes by mapping all four corners."""
    output = np.full_like(boxes, -1.0, dtype=np.float32)
    if boxes.size == 0:
        return output

    valid = (
        (boxes[..., 0] >= 0)
        & (boxes[..., 1] >= 0)
        & (boxes[..., 2] > boxes[..., 0])
        & (boxes[..., 3] > boxes[..., 1])
    )

    for part_index, box_index in zip(*np.where(valid)):
        x1, y1, x2, y2 = boxes[part_index, box_index].astype(np.float64)
        corners = np.array(
            [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
            dtype=np.float64,
        ).T

        transformed = source_to_output @ corners
        tx = transformed[0] / np.maximum(transformed[2], 1e-8)
        ty = transformed[1] / np.maximum(transformed[2], 1e-8)

        nx1 = float(np.clip(tx.min(), 0.0, image_size))
        ny1 = float(np.clip(ty.min(), 0.0, image_size))
        nx2 = float(np.clip(tx.max(), 0.0, image_size))
        ny2 = float(np.clip(ty.max(), 0.0, image_size))

        if nx2 - nx1 >= 1.0 and ny2 - ny1 >= 1.0:
            output[part_index, box_index] = [nx1, ny1, nx2, ny2]

    return output


# -----------------------------------------------------------------------------
# Nonlinear TesNet-style transforms: images only, never part boxes
# -----------------------------------------------------------------------------
def perspective_coefficients(
    output_points: np.ndarray,
    source_points: np.ndarray,
) -> Tuple[float, ...]:
    """Solve coefficients mapping output xy coordinates back to source xy."""
    matrix: List[List[float]] = []
    vector: List[float] = []

    for (x, y), (u, v) in zip(output_points, source_points):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        vector.append(u)
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        vector.append(v)

    coefficients = np.linalg.solve(
        np.asarray(matrix, dtype=np.float64),
        np.asarray(vector, dtype=np.float64),
    )
    return tuple(float(value) for value in coefficients)


def random_skew_image(
    image: Image.Image,
    rng: random.Random,
    magnitude: float = 0.20,
) -> Image.Image:
    """Perspective-style skew, corresponding to TesNet's Augmentor skew family."""
    width, height = image.size
    amount = magnitude * min(width, height) * rng.uniform(0.5, 1.0)
    source = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float64,
    )
    destination = source.copy()

    mode = rng.randrange(4)
    if mode == 0:
        destination[[0, 1], 0] += amount
    elif mode == 1:
        destination[[2, 3], 0] -= amount
    elif mode == 2:
        destination[[0, 3], 1] += amount
    else:
        destination[[1, 2], 1] -= amount

    destination[:, 0] = np.clip(destination[:, 0], 0.0, width - 1.0)
    destination[:, 1] = np.clip(destination[:, 1], 0.0, height - 1.0)

    coefficients = perspective_coefficients(destination, source)
    return image.transform(
        image.size,
        pil_perspective_mode(),
        coefficients,
        resample=pil_resample_bicubic(),
        fillcolor=(0, 0, 0),
    )


def random_grid_distortion_image(
    image: Image.Image,
    rng: random.Random,
    grid_width: int = 10,
    grid_height: int = 10,
    magnitude: float = 5.0,
) -> Image.Image:
    """Piecewise mesh warp matching TesNet's 10x10 random-distortion family."""
    width, height = image.size
    xs = np.linspace(0.0, float(width), grid_width + 1)
    ys = np.linspace(0.0, float(height), grid_height + 1)

    source_x = np.tile(xs[None, :], (grid_height + 1, 1))
    source_y = np.tile(ys[:, None], (1, grid_width + 1))

    for gy in range(1, grid_height):
        for gx in range(1, grid_width):
            source_x[gy, gx] += rng.uniform(-magnitude, magnitude)
            source_y[gy, gx] += rng.uniform(-magnitude, magnitude)

    source_x = np.clip(source_x, 0.0, float(width))
    source_y = np.clip(source_y, 0.0, float(height))

    mesh = []
    for gy in range(grid_height):
        for gx in range(grid_width):
            x0 = int(round(xs[gx]))
            y0 = int(round(ys[gy]))
            x1 = int(round(xs[gx + 1]))
            y1 = int(round(ys[gy + 1]))
            if x1 <= x0 or y1 <= y0:
                continue

            # PIL MESH quad ordering: upper-left, lower-left, lower-right, upper-right.
            quad = (
                float(source_x[gy, gx]),
                float(source_y[gy, gx]),
                float(source_x[gy + 1, gx]),
                float(source_y[gy + 1, gx]),
                float(source_x[gy + 1, gx + 1]),
                float(source_y[gy + 1, gx + 1]),
                float(source_x[gy, gx + 1]),
                float(source_y[gy, gx + 1]),
            )
            mesh.append(((x0, y0, x1, y1), quad))

    return image.transform(
        image.size,
        pil_mesh_mode(),
        mesh,
        resample=pil_resample_bicubic(),
    )


# -----------------------------------------------------------------------------
# Dataset output paths / variants
# -----------------------------------------------------------------------------
def build_variants(variants_per_type: int) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = [
        {"name": "original", "family": "original", "index": -1, "has_boxes": True}
    ]
    for family, has_boxes in (
        ("rotate", True),
        ("skew", False),
        ("shear", True),
        ("distortion", False),
    ):
        for index in range(variants_per_type):
            variants.append(
                {
                    "name": f"{family}_{index:02d}",
                    "family": family,
                    "index": index,
                    "has_boxes": has_boxes,
                }
            )
    return variants


def image_relative_path(rel_annotation: str, variant: Dict[str, Any]) -> str:
    parent, stem = os.path.split(rel_annotation)
    if variant["family"] == "original":
        return os.path.join("train_cropped", parent, stem + ".jpg")
    return os.path.join(
        "train_cropped_augmented",
        parent,
        f"{stem}_{variant['name']}.jpg",
    )


def save_jpeg(image: Image.Image, path: str, quality: int) -> None:
    ensure_parent(path)
    image.convert("RGB").save(
        path,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=False,
    )


def process_source(task: Dict[str, Any]) -> Tuple[int, np.ndarray]:
    """Generate all saved variants for one source image; called in worker processes."""
    source_index = int(task["source_index"])
    image_path = task["image_path"]
    annotation_path = task["annotation_path"]
    rel_annotation = task["rel_annotation"]
    source_boxes = np.asarray(task["source_boxes"], dtype=np.float32)
    output_root = task["output_root"]
    image_size = int(task["image_size"])
    variants = task["variants"]
    seed = int(task["seed"])
    jpeg_quality = int(task["jpeg_quality"])

    rng = random.Random(seed + source_index * 1009)

    with Image.open(image_path) as pil_image:
        image = pil_image.convert("RGB")

    dog_crop = crop_xyxy(image, parse_annotation_union_box(annotation_path))
    base_image = dog_crop.resize((image_size, image_size), resample=pil_resample_bicubic())

    num_variants = len(variants)
    transformed_boxes = np.full(
        (num_variants, source_boxes.shape[0], source_boxes.shape[1], 4),
        -1.0,
        dtype=np.float32,
    )

    for variant_index, variant in enumerate(variants):
        family = variant["family"]
        output_path = os.path.join(output_root, image_relative_path(rel_annotation, variant))

        if family == "original":
            output_image = base_image
            transformed_boxes[variant_index] = source_boxes

        elif family == "rotate":
            angle = rng.uniform(-15.0, 15.0)
            matrix = rotation_matrix(angle, image_size, image_size)
            if rng.random() < 0.5:
                matrix = hflip_matrix(image_size) @ matrix
            output_image = apply_affine_image(base_image, matrix)
            transformed_boxes[variant_index] = apply_affine_boxes(
                source_boxes, matrix, image_size
            )

        elif family == "shear":
            shear_angle = rng.uniform(-10.0, 10.0)
            matrix = x_shear_matrix(shear_angle, image_size, image_size)
            if rng.random() < 0.5:
                matrix = hflip_matrix(image_size) @ matrix
            output_image = apply_affine_image(base_image, matrix)
            transformed_boxes[variant_index] = apply_affine_boxes(
                source_boxes, matrix, image_size
            )

        elif family == "skew":
            output_image = random_skew_image(base_image, rng, magnitude=0.20)
            if rng.random() < 0.5:
                output_image = ImageOps.mirror(output_image)
            # Deliberately keep all boxes invalid for non-linear skew.

        elif family == "distortion":
            output_image = random_grid_distortion_image(
                base_image,
                rng,
                grid_width=10,
                grid_height=10,
                magnitude=5.0,
            )
            if rng.random() < 0.5:
                output_image = ImageOps.mirror(output_image)
            # Deliberately keep all boxes invalid for non-linear distortion.

        else:
            raise RuntimeError(f"Unknown augmentation family: {family}")

        save_jpeg(output_image, output_path, jpeg_quality)

    return source_index, transformed_boxes


# -----------------------------------------------------------------------------
# Build / resume
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Build offline TesNet-style Stanford Dogs augmentation with synchronized boxes"
    )
    parser.add_argument("--dogs-root", required=True)
    parser.add_argument("--gdino-box-file", required=True)
    parser.add_argument("--output-root", required=True)

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--variants-per-type", type=int, default=10)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-sources",
        type=int,
        default=0,
        help="For smoke tests only; 0 means all official train images.",
    )
    parser.add_argument("--allow-incomplete-boxes", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.dogs_root = str(Path(args.dogs_root).expanduser().resolve())
    args.gdino_box_file = str(Path(args.gdino_box_file).expanduser().resolve())
    args.output_root = str(Path(args.output_root).expanduser().resolve())

    required = [
        Path(args.dogs_root) / "Images",
        Path(args.dogs_root) / "Annotation",
        Path(args.dogs_root) / "train_list.mat",
        Path(args.gdino_box_file),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n  - " + "\n  - ".join(missing))

    if args.image_size < 16:
        raise ValueError("--image-size must be >= 16.")
    if args.variants_per_type < 1:
        raise ValueError("--variants-per-type must be >= 1.")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")
    if args.shard_size < 1:
        raise ValueError("--shard-size must be >= 1.")
    if not (1 <= args.jpeg_quality <= 100):
        raise ValueError("--jpeg-quality must be in [1, 100].")


def shard_path(output_root: str, shard_index: int) -> str:
    return os.path.join(output_root, "box_shards", f"shard_{shard_index:05d}.pt")


def shard_is_complete(
    path: str,
    source_start: int,
    source_end: int,
    num_variants: int,
    num_parts: int,
    max_boxes: int,
) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        payload = safe_torch_load(path)
        boxes = payload.get("boxes_xyxy_resize")
        return (
            isinstance(payload, dict)
            and int(payload.get("source_start", -1)) == source_start
            and int(payload.get("source_end", -1)) == source_end
            and isinstance(boxes, torch.Tensor)
            and tuple(boxes.shape)
            == (source_end - source_start, num_variants, num_parts, max_boxes, 4)
        )
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    validate_args(args)

    parts = list(DEFAULT_PARTS)
    variants = build_variants(args.variants_per_type)
    num_variants = len(variants)

    all_items = parse_split_mat(os.path.join(args.dogs_root, "train_list.mat"))
    if args.max_sources > 0:
        all_items = all_items[: args.max_sources]

    samples = [
        {
            "rel_annotation": rel_annotation,
            "label": int(label),
            "image_path": os.path.join(args.dogs_root, "Images", rel_annotation + ".jpg"),
            "annotation_path": os.path.join(args.dogs_root, "Annotation", rel_annotation),
        }
        for rel_annotation, label in all_items
    ]

    source_boxes, max_boxes, cache_summary = load_source_boxes(
        cache_path=args.gdino_box_file,
        samples=samples,
        parts=parts,
        image_size=args.image_size,
        allow_incomplete=args.allow_incomplete_boxes,
    )

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(os.path.join(args.output_root, "box_shards"), exist_ok=True)

    num_shards = math.ceil(len(samples) / args.shard_size)
    print(
        f"[Build] sources={len(samples):,}; variants/source={num_variants}; "
        f"saved images={len(samples) * num_variants:,}; "
        f"parts={parts}; max_boxes={max_boxes}; shards={num_shards}"
    )
    print(
        "[Policy] 10 rotation + 10 skew + 10 shear + 10 distortion "
        "(or --variants-per-type); HFlip(p=0.5) in every family."
    )
    print(
        "[Boxes] original/rotation/shear synchronized; "
        "skew/distortion deliberately invalidated."
    )
    print(f"[GDINO] {cache_summary}")

    saved_shards: List[str] = []

    for shard_index in range(num_shards):
        source_start = shard_index * args.shard_size
        source_end = min(len(samples), source_start + args.shard_size)
        current_shard = shard_path(args.output_root, shard_index)
        shard_relpath = os.path.relpath(current_shard, args.output_root)

        if args.resume and shard_is_complete(
            current_shard,
            source_start,
            source_end,
            num_variants,
            len(parts),
            max_boxes,
        ):
            saved_shards.append(shard_relpath)
            print(
                f"[Resume] shard {shard_index + 1}/{num_shards}: "
                f"sources {source_start}:{source_end} already complete."
            )
            continue

        shard_boxes = np.full(
            (
                source_end - source_start,
                num_variants,
                len(parts),
                max_boxes,
                4,
            ),
            -1.0,
            dtype=np.float32,
        )

        tasks = [
            {
                "source_index": global_index,
                "image_path": samples[global_index]["image_path"],
                "annotation_path": samples[global_index]["annotation_path"],
                "rel_annotation": samples[global_index]["rel_annotation"],
                "source_boxes": source_boxes[global_index],
                "output_root": args.output_root,
                "image_size": args.image_size,
                "variants": variants,
                "seed": args.seed,
                "jpeg_quality": args.jpeg_quality,
            }
            for global_index in range(source_start, source_end)
        ]

        progress = tqdm(
            total=len(tasks),
            desc=f"Aug shard {shard_index + 1}/{num_shards}",
            dynamic_ncols=True,
        )

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_source, task) for task in tasks]
            for future in as_completed(futures):
                source_index, output_boxes = future.result()
                shard_boxes[source_index - source_start] = output_boxes
                progress.update(1)
        progress.close()

        shard_payload = {
            "source_start": source_start,
            "source_end": source_end,
            "source_rel_annotations": [
                samples[index]["rel_annotation"] for index in range(source_start, source_end)
            ],
            "labels": torch.tensor(
                [samples[index]["label"] for index in range(source_start, source_end)],
                dtype=torch.long,
            ),
            "boxes_xyxy_resize": torch.from_numpy(shard_boxes),
        }
        atomic_torch_save(shard_payload, current_shard)
        saved_shards.append(shard_relpath)
        print(
            f"[Saved] shard {shard_index + 1}/{num_shards}: "
            f"{current_shard}"
        )

    manifest = {
        "meta": {
            "dataset": "Stanford Dogs",
            "split": "train",
            "source_geometry": "official dog bbox crop -> direct resize(image_size,image_size)",
            "image_size": int(args.image_size),
            "parts": parts,
            "max_boxes": int(max_boxes),
            "variants_per_type": int(args.variants_per_type),
            "num_variants_per_source": int(num_variants),
            "policy": {
                "rotate": {"count": args.variants_per_type, "degrees": [-15.0, 15.0], "hflip_prob": 0.5, "boxes": "affine-synchronized"},
                "skew": {"count": args.variants_per_type, "magnitude": 0.20, "hflip_prob": 0.5, "boxes": "invalid (-1), nonlinear family"},
                "shear": {"count": args.variants_per_type, "degrees": [-10.0, 10.0], "hflip_prob": 0.5, "boxes": "affine-synchronized"},
                "distortion": {"count": args.variants_per_type, "grid": [10, 10], "magnitude": 5.0, "hflip_prob": 0.5, "boxes": "invalid (-1), nonlinear family"},
            },
            "source_box_cache": args.gdino_box_file,
            "cache_summary": cache_summary,
        },
        "variants": variants,
        "source_rel_annotations": [sample["rel_annotation"] for sample in samples],
        "source_labels": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
        "box_shards": saved_shards,
    }
    manifest_path = os.path.join(args.output_root, "train_tesnet_aug_manifest.pt")
    atomic_torch_save(manifest, manifest_path)

    summary = {
        "num_sources": len(samples),
        "num_variants_per_source": num_variants,
        "total_saved_images": len(samples) * num_variants,
        "parts": parts,
        "max_boxes": max_boxes,
        "manifest": manifest_path,
        "shards": saved_shards,
    }
    with open(os.path.join(args.output_root, "build_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Done] manifest: {manifest_path}")
    print(
        "[Next] The current training script still uses raw train images only. "
        "Use the offline-augmentation-aware training script after this build completes."
    )


if __name__ == "__main__":
    main()
