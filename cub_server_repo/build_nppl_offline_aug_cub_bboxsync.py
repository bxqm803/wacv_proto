#!/usr/bin/env python3
"""Build NPPL/TesNet-style offline CUB augmentation with synchronized part boxes.

Why this replacement exists
----------------------------
The earlier Augmentor-based version can fail inside Augmentor.random_distortion
on narrow crops ("Coordinate 'lower' is less than 'upper'"). It also cannot
record the sampled geometry, so its old GroundingDINO boxes cannot be aligned
with generated images.

This replacement keeps the same NPPL/TesNet operation family and count:
  - rotate in [-15, +15] degrees, 10 variants/source
  - skew / projective warp with magnitude 0.2, 10 variants/source
  - shear in [-10, +10] degrees, 10 variants/source
  - grid random-distortion 10x10, magnitude 5, 10 variants/source
  - every variant is horizontally flipped with probability 0.5

Differences from Augmentor
--------------------------
It is not bit-for-bit identical to Augmentor's random sampler. Geometry is
implemented explicitly with OpenCV so that rotation, skew/projective warp,
shear, and flip can transform GroundingDINO boxes exactly through the same
homography. The non-linear grid-distortion samples are still generated, but
receive invalid boxes and are marked bbox_supervised=False.

Outputs
-------
<output-root>/train_cropped/
<output-root>/train_cropped_augmented/<class>/...
<output-root>/aug_chunks/<class>.pt
<output-root>/train_augmented_manifest.jsonl
<output-root>/train_part_boxes_augmented_nppl.pt

The final .pt uses the old tensor cache key `part_boxes_xyxy_pix`, so the
existing training loader can read its boxes once an augmented-image Dataset is
added. The cache uses augmented-image relpaths, not original CUB relpaths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PARTS = ["beak", "head", "tail", "body", "feet", "wing"]
MAX_BOXES = 2  # original visible GDINO builder keeps at most two boxes (wing/feet)


@dataclass(frozen=True)
class AugConfig:
    repetitions_per_operation: int = 10
    rotate_left: float = 15.0
    rotate_right: float = 15.0
    skew_magnitude: float = 0.20
    shear_left: float = 10.0
    shear_right: float = 10.0
    distortion_grid_width: int = 10
    distortion_grid_height: int = 10
    distortion_magnitude: float = 5.0
    hflip_probability: float = 0.5


@dataclass(frozen=True)
class ClassTask:
    class_name: str
    crop_root: str
    aug_root: str
    chunk_path: str
    records: List[Dict[str, Any]]
    cfg_dict: Dict[str, Any]
    image_size: int
    jpeg_quality: int
    seed: int
    overwrite_incomplete: bool
    operations: Tuple[str, ...]


# -----------------------------------------------------------------------------
# CUB metadata
# -----------------------------------------------------------------------------
def read_kv_text(path: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                k, v = line.split(maxsplit=1)
                out[int(k)] = v
    return out


def read_kv_int(path: Path) -> Dict[int, int]:
    out: Dict[int, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
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
    x1 = max(0, min(width - 1, int(math.floor(x))))
    y1 = max(0, min(height - 1, int(math.floor(y))))
    x2 = max(x1 + 1, min(width, int(math.ceil(x + w))))
    y2 = max(y1 + 1, min(height, int(math.ceil(y + h))))
    return image.crop((x1, y1, x2, y2))


def pil_resize_square(image: Image.Image, size: int) -> Image.Image:
    mode = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    return image.resize((size, size), mode)


def build_train_crops(
    cub_root: Path,
    crop_root: Path,
    jpeg_quality: int,
    skip_existing: bool,
) -> List[Dict[str, Any]]:
    paths = read_kv_text(cub_root / "images.txt")
    split_map = read_kv_int(cub_root / "train_test_split.txt")
    bboxes = read_bboxes(cub_root / "bounding_boxes.txt")

    records: List[Dict[str, Any]] = []
    train_ids = sorted(i for i in paths if split_map[i] == 1)
    print(f"[Crop] writing {len(train_ids)} GT bird crops to: {crop_root}")

    for image_id in tqdm(train_ids, desc="Build train_cropped", ncols=120):
        relpath = Path(paths[image_id])
        out_path = crop_root / relpath
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not (skip_existing and out_path.is_file()):
            with Image.open(cub_root / "images" / relpath) as img:
                crop = crop_bbox(img.convert("RGB"), bboxes[image_id])
                crop.save(out_path, quality=jpeg_quality, optimize=False)
        records.append(
            {
                "image_id": int(image_id),
                "source_relpath": relpath.as_posix(),
                "crop_relpath": relpath.as_posix(),
                "class_name": relpath.parent.name,
                "label": int(relpath.parent.name.split(".", 1)[0]) - 1,
                "bbox_xywh_raw": [float(v) for v in bboxes[image_id]],
            }
        )
    return records


# -----------------------------------------------------------------------------
# Source GDINO cache -> fixed [P,M,4] boxes at augmentation image_size.
# -----------------------------------------------------------------------------
def _as_boxes(raw: Any) -> np.ndarray:
    if raw is None:
        return np.zeros((0, 4), dtype=np.float32)
    if isinstance(raw, torch.Tensor):
        arr = raw.detach().cpu().float().numpy()
    else:
        arr = np.asarray(raw, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if arr.ndim == 1 and arr.size == 4:
        arr = arr.reshape(1, 4)
    elif arr.ndim >= 2 and arr.shape[-1] == 4:
        arr = arr.reshape(-1, 4)
    else:
        return np.zeros((0, 4), dtype=np.float32)
    keep = (arr[:, 0] >= 0) & (arr[:, 2] > arr[:, 0]) & (arr[:, 3] > arr[:, 1])
    return arr[keep].astype(np.float32)


def _as_scores(raw: Any, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    if raw is None:
        return np.ones((n,), dtype=np.float32)
    if isinstance(raw, torch.Tensor):
        arr = raw.detach().cpu().float().numpy().reshape(-1)
    else:
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.size < n:
        arr = np.pad(arr, (0, n - arr.size), constant_values=1.0)
    return arr[:n].astype(np.float32)


def _infer_cache_parts(cache: Dict[str, Any]) -> List[str]:
    if "parts" in cache:
        return [str(x).lower() for x in cache["parts"]]
    if isinstance(cache.get("prompts"), dict):
        return [str(x).lower() for x in cache["prompts"].keys()]
    return list(PARTS)


def _part_index(cache_parts: Sequence[str], target: str) -> int | None:
    target = target.lower()
    for i, name in enumerate(cache_parts):
        n = str(name).lower()
        if n == target:
            return i
        if target == "wing" and "wing" in n:
            return i
        if target == "beak" and ("beak" in n or "bill" in n):
            return i
        if target == "feet" and ("foot" in n or "feet" in n or "leg" in n):
            return i
    return None


def _selected_entry(cache: Dict[str, Any], row: int, part_idx: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Read selected boxes/scores/visibility for one source cache row and part."""
    visible = True
    if "visible_parts" in cache:
        try:
            visible = bool(cache["visible_parts"][row][part_idx])
        except Exception:
            visible = True

    if "selected_boxes_xyxy" in cache:
        selected = cache["selected_boxes_xyxy"]
        try:
            raw_boxes = selected[row][part_idx]
        except Exception:
            raw_boxes = None
        raw_scores = None
        if "selected_scores" in cache:
            try:
                raw_scores = cache["selected_scores"][row][part_idx]
            except Exception:
                raw_scores = None
        boxes = _as_boxes(raw_boxes)
        return boxes, _as_scores(raw_scores, len(boxes)), visible

    if "part_boxes_xyxy_pix" in cache:
        pb = cache["part_boxes_xyxy_pix"]
        if isinstance(pb, torch.Tensor):
            raw_boxes = pb[row, part_idx]
        else:
            raw_boxes = pb[row][part_idx]
        boxes = _as_boxes(raw_boxes)
        return boxes, np.ones((len(boxes),), dtype=np.float32), visible

    raise KeyError("Source GDINO cache needs selected_boxes_xyxy or part_boxes_xyxy_pix")


def attach_source_gdino(
    records: List[Dict[str, Any]],
    cache_path: Path,
    output_image_size: int,
) -> None:
    cache = torch.load(cache_path, map_location="cpu")
    if "relpaths" not in cache:
        raise KeyError(f"Missing relpaths in GDINO cache: {cache_path}")
    rel_to_row = {str(v): i for i, v in enumerate(cache["relpaths"])}
    cache_parts = _infer_cache_parts(cache)
    source_size = int(cache.get("image_size", cache.get("img_size", output_image_size)))
    scale = float(output_image_size) / max(1.0, float(source_size))

    part_map = [_part_index(cache_parts, p) for p in PARTS]
    missing = 0
    for rec in records:
        boxes = np.full((len(PARTS), MAX_BOXES, 4), -1.0, dtype=np.float32)
        scores = np.full((len(PARTS), MAX_BOXES), -1.0, dtype=np.float32)
        visible = np.zeros((len(PARTS),), dtype=np.bool_)
        row = rel_to_row.get(str(rec["source_relpath"]))
        if row is None:
            missing += 1
        else:
            for p, src_idx in enumerate(part_map):
                if src_idx is None:
                    continue
                src_boxes, src_scores, src_visible = _selected_entry(cache, row, src_idx)
                src_boxes = src_boxes * scale
                m = min(MAX_BOXES, len(src_boxes))
                if m > 0:
                    boxes[p, :m] = src_boxes[:m]
                    scores[p, :m] = src_scores[:m]
                visible[p] = bool(src_visible and m > 0)
        rec["source_boxes"] = boxes
        rec["source_scores"] = scores
        rec["source_visible"] = visible

    print(
        f"[GDINO] source={cache_path.name} source_size={source_size} target_size={output_image_size} "
        f"scale={scale:.4f} attached={len(records)-missing}/{len(records)} missing={missing}"
    )


# -----------------------------------------------------------------------------
# Geometric transforms and synchronized boxes.
# -----------------------------------------------------------------------------
def _seed_for(seed: int, image_id: int, op: str, rep: int) -> int:
    raw = f"{seed}|{image_id}|{op}|{rep}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little") & 0x7FFFFFFF


def _warp_homography(image: np.ndarray, hmat: np.ndarray) -> np.ndarray:
    side = int(image.shape[0])
    return cv2.warpPerspective(
        image,
        hmat.astype(np.float32),
        (side, side),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _rotation_hmat(side: int, angle_deg: float) -> np.ndarray:
    c = 0.5 * float(side - 1)
    aff = cv2.getRotationMatrix2D((c, c), float(angle_deg), 1.0)
    out = np.eye(3, dtype=np.float32)
    out[:2] = aff.astype(np.float32)
    return out


def _shear_hmat(side: int, rng: random.Random, cfg: AugConfig) -> Tuple[np.ndarray, Dict[str, float]]:
    angle = rng.uniform(-cfg.shear_left, cfg.shear_right)
    shear = math.tan(math.radians(angle))
    c = 0.5 * float(side)
    hmat = np.eye(3, dtype=np.float32)
    if rng.random() < 0.5:
        hmat[0, 1] = shear
        hmat[0, 2] = -shear * c
        axis = "x"
    else:
        hmat[1, 0] = shear
        hmat[1, 2] = -shear * c
        axis = "y"
    return hmat, {"shear_deg": float(angle), "shear_axis": axis}


def _valid_quad(quad: np.ndarray, side: int) -> bool:
    q = quad.astype(np.float32)
    if not cv2.isContourConvex(q.reshape(-1, 1, 2)):
        return False
    area = abs(float(cv2.contourArea(q.reshape(-1, 1, 2))))
    return area >= 0.35 * float(side * side)


def _skew_hmat(side: int, rng: random.Random, cfg: AugConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    src = np.array(
        [[0.0, 0.0], [float(side - 1), 0.0], [float(side - 1), float(side - 1)], [0.0, float(side - 1)]],
        dtype=np.float32,
    )
    max_jitter = float(cfg.skew_magnitude) * float(side)
    for _ in range(50):
        jitter = np.array(
            [[rng.uniform(-max_jitter, max_jitter), rng.uniform(-max_jitter, max_jitter)] for _ in range(4)],
            dtype=np.float32,
        )
        dst = src + jitter
        if _valid_quad(dst, side):
            hmat = cv2.getPerspectiveTransform(src, dst).astype(np.float32)
            return hmat, {"skew_dst_corners": dst.round(4).tolist()}
    return np.eye(3, dtype=np.float32), {"skew_fallback_identity": True}


def _random_distortion(image: np.ndarray, rng: np.random.Generator, cfg: AugConfig) -> np.ndarray:
    """Robust grid random distortion. Deliberately has no usable box mapping."""
    h, w = image.shape[:2]
    gw = max(2, int(cfg.distortion_grid_width))
    gh = max(2, int(cfg.distortion_grid_height))
    mag = float(cfg.distortion_magnitude)

    dx_small = rng.uniform(-mag, mag, size=(gh + 1, gw + 1)).astype(np.float32)
    dy_small = rng.uniform(-mag, mag, size=(gh + 1, gw + 1)).astype(np.float32)
    # Keep outer edges fixed; prevents invalid crops / empty output regions.
    dx_small[[0, -1], :] = 0.0
    dx_small[:, [0, -1]] = 0.0
    dy_small[[0, -1], :] = 0.0
    dy_small[:, [0, -1]] = 0.0

    dx = cv2.resize(dx_small, (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy_small, (w, h), interpolation=cv2.INTER_CUBIC)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(
        image,
        xx + dx,
        yy + dy,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _transform_one_box(box: np.ndarray, hmat: np.ndarray, side: int, hflip: bool) -> np.ndarray | None:
    x1, y1, x2, y2 = [float(v) for v in box]
    if not (x1 >= 0 and x2 > x1 and y2 > y1):
        return None
    pts = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pts, hmat.astype(np.float32))[0]
    nx1, ny1 = out.min(axis=0)
    nx2, ny2 = out.max(axis=0)
    if hflip:
        nx1, nx2 = float(side) - nx2, float(side) - nx1
    nx1 = float(np.clip(nx1, 0.0, float(side)))
    ny1 = float(np.clip(ny1, 0.0, float(side)))
    nx2 = float(np.clip(nx2, 0.0, float(side)))
    ny2 = float(np.clip(ny2, 0.0, float(side)))
    if nx2 <= nx1 + 1.0 or ny2 <= ny1 + 1.0:
        return None
    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


def transform_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    source_visible: np.ndarray,
    hmat: np.ndarray,
    side: int,
    hflip: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_boxes = np.full_like(boxes, -1.0, dtype=np.float32)
    out_scores = np.full_like(scores, -1.0, dtype=np.float32)
    out_visible = np.zeros_like(source_visible, dtype=np.bool_)
    for p in range(boxes.shape[0]):
        kept = 0
        for m in range(boxes.shape[1]):
            transformed = _transform_one_box(boxes[p, m], hmat, side, hflip)
            if transformed is not None:
                out_boxes[p, kept] = transformed
                out_scores[p, kept] = scores[p, m]
                kept += 1
        out_visible[p] = bool(source_visible[p] and kept > 0)
    return out_boxes, out_scores, out_visible


def make_variant(
    image: np.ndarray,
    operation: str,
    rng_py: random.Random,
    rng_np: np.random.Generator,
    cfg: AugConfig,
) -> Tuple[np.ndarray, np.ndarray | None, Dict[str, Any], bool]:
    side = int(image.shape[0])
    metadata: Dict[str, Any] = {"operation": operation}
    is_linear = operation != "random_distortion"

    if operation == "rotate":
        angle = rng_py.uniform(-cfg.rotate_left, cfg.rotate_right)
        hmat = _rotation_hmat(side, angle)
        out = _warp_homography(image, hmat)
        metadata["rotate_deg"] = float(angle)
    elif operation == "shear":
        hmat, extra = _shear_hmat(side, rng_py, cfg)
        out = _warp_homography(image, hmat)
        metadata.update(extra)
    elif operation == "skew":
        hmat, extra = _skew_hmat(side, rng_py, cfg)
        out = _warp_homography(image, hmat)
        metadata.update(extra)
    elif operation == "random_distortion":
        hmat = None
        out = _random_distortion(image, rng_np, cfg)
        metadata.update(
            {
                "grid_width": int(cfg.distortion_grid_width),
                "grid_height": int(cfg.distortion_grid_height),
                "distortion_magnitude": float(cfg.distortion_magnitude),
            }
        )
    else:
        raise ValueError(f"Unknown operation: {operation}")

    hflip = bool(rng_py.random() < cfg.hflip_probability)
    if hflip:
        out = np.ascontiguousarray(out[:, ::-1])
    metadata["hflip"] = hflip
    return out, hmat, metadata, is_linear


# -----------------------------------------------------------------------------
# Per-class worker.
# -----------------------------------------------------------------------------
def _count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _save_jpeg(rgb: np.ndarray, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path, quality=int(quality), optimize=False)


def _class_complete(target_dir: Path, chunk_path: Path, expected: int) -> bool:
    marker = target_dir / ".nppl_bboxsync_complete.json"
    if not (marker.is_file() and chunk_path.is_file()):
        return False
    try:
        m = json.loads(marker.read_text(encoding="utf-8"))
        return int(m.get("expected_count", -1)) == expected and _count_images(target_dir) >= expected
    except Exception:
        return False


def _clear_class_output(target_dir: Path, chunk_path: Path) -> None:
    if target_dir.exists():
        for p in target_dir.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    target_dir.mkdir(parents=True, exist_ok=True)
    if chunk_path.exists():
        chunk_path.unlink()


def augment_one_class(task: ClassTask) -> Tuple[str, str, int, int]:
    # Avoid OpenCV spawning its own large thread pool inside each multiprocessing worker.
    cv2.setNumThreads(0)

    cfg = AugConfig(**task.cfg_dict)
    crop_root = Path(task.crop_root)
    target_dir = Path(task.aug_root) / task.class_name
    chunk_path = Path(task.chunk_path)
    expected = len(task.records) * len(task.operations) * int(cfg.repetitions_per_operation)

    if _class_complete(target_dir, chunk_path, expected):
        return task.class_name, "skipped_complete", len(task.records), expected

    if task.overwrite_incomplete:
        _clear_class_output(target_dir, chunk_path)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
        chunk_path.parent.mkdir(parents=True, exist_ok=True)

    relpaths: List[str] = []
    source_relpaths: List[str] = []
    source_ids: List[int] = []
    labels: List[int] = []
    operations: List[str] = []
    geometries: List[str] = []
    transform_meta: List[Dict[str, Any]] = []
    bbox_supervised: List[bool] = []
    all_boxes: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_visible: List[np.ndarray] = []

    for rec in task.records:
        src_path = crop_root / rec["crop_relpath"]
        with Image.open(src_path) as im:
            image = np.asarray(pil_resize_square(im.convert("RGB"), task.image_size), dtype=np.uint8)

        stem = Path(rec["crop_relpath"]).stem
        src_boxes = np.asarray(rec["source_boxes"], dtype=np.float32)
        src_scores = np.asarray(rec["source_scores"], dtype=np.float32)
        src_visible = np.asarray(rec["source_visible"], dtype=np.bool_)

        for op in task.operations:
            for rep in range(int(cfg.repetitions_per_operation)):
                seed = _seed_for(task.seed, int(rec["image_id"]), op, rep)
                py_rng = random.Random(seed)
                np_rng = np.random.default_rng(seed)
                variant, hmat, meta, is_linear = make_variant(image, op, py_rng, np_rng, cfg)

                if is_linear:
                    boxes, scores, visible = transform_boxes(
                        src_boxes,
                        src_scores,
                        src_visible,
                        hmat,
                        task.image_size,
                        bool(meta["hflip"]),
                    )
                    supervised = bool(visible.any())
                    geometry = "homography"
                else:
                    boxes = np.full_like(src_boxes, -1.0, dtype=np.float32)
                    scores = np.full_like(src_scores, -1.0, dtype=np.float32)
                    visible = np.zeros_like(src_visible, dtype=np.bool_)
                    supervised = False
                    geometry = "nonlinear_no_bbox"

                filename = f"{stem}__nppl_{op}_r{rep:02d}.jpg"
                aug_relpath = f"{task.class_name}/{filename}"
                _save_jpeg(variant, target_dir / filename, task.jpeg_quality)

                relpaths.append(aug_relpath)
                source_relpaths.append(str(rec["source_relpath"]))
                source_ids.append(int(rec["image_id"]))
                labels.append(int(rec["label"]))
                operations.append(op)
                geometries.append(geometry)
                transform_meta.append(meta)
                bbox_supervised.append(supervised)
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_visible.append(visible)

    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk = {
        "class_name": task.class_name,
        "relpaths": relpaths,
        "source_relpaths": source_relpaths,
        "source_image_ids": torch.tensor(source_ids, dtype=torch.int64),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "operations": operations,
        "geometries": geometries,
        "transform_meta": transform_meta,
        "bbox_supervised": torch.tensor(bbox_supervised, dtype=torch.bool),
        "part_boxes_xyxy_pix": torch.from_numpy(np.stack(all_boxes, axis=0)).float(),
        "part_scores": torch.from_numpy(np.stack(all_scores, axis=0)).float(),
        "visible_parts": torch.from_numpy(np.stack(all_visible, axis=0)).bool(),
    }
    tmp = chunk_path.with_suffix(chunk_path.suffix + ".tmp")
    torch.save(chunk, tmp)
    os.replace(tmp, chunk_path)

    marker = {
        "class_name": task.class_name,
        "source_count": len(task.records),
        "expected_count": expected,
        "output_count": _count_images(target_dir),
        "operations": list(task.operations),
        "augmentation_recipe": asdict(cfg),
        "bbox_note": "rotate/skew/shear/flip boxes are synchronized; random_distortion has invalid boxes.",
    }
    (target_dir / ".nppl_bboxsync_complete.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return task.class_name, "done", len(task.records), len(relpaths)


# -----------------------------------------------------------------------------
# Final cache / manifest assembly.
# -----------------------------------------------------------------------------
def assemble_final_cache(output_root: Path, class_names: Sequence[str], image_size: int) -> Dict[str, Any]:
    chunk_dir = output_root / "aug_chunks"
    chunks: List[Dict[str, Any]] = []
    for class_name in class_names:
        p = chunk_dir / f"{class_name}.pt"
        if not p.is_file():
            raise FileNotFoundError(f"Missing completed chunk: {p}")
        chunks.append(torch.load(p, map_location="cpu"))

    relpaths: List[str] = []
    source_relpaths: List[str] = []
    operations: List[str] = []
    geometries: List[str] = []
    transform_meta: List[Dict[str, Any]] = []
    for ch in chunks:
        relpaths.extend(ch["relpaths"])
        source_relpaths.extend(ch["source_relpaths"])
        operations.extend(ch["operations"])
        geometries.extend(ch["geometries"])
        transform_meta.extend(ch["transform_meta"])

    cache = {
        "format": "nppl_offline_aug_bboxsync_v1",
        "parts": list(PARTS),
        "image_size": int(image_size),
        "relpaths": relpaths,
        "source_relpaths": source_relpaths,
        "source_image_ids": torch.cat([ch["source_image_ids"] for ch in chunks], dim=0),
        "labels": torch.cat([ch["labels"] for ch in chunks], dim=0),
        "operations": operations,
        "geometries": geometries,
        "transform_meta": transform_meta,
        "bbox_supervised": torch.cat([ch["bbox_supervised"] for ch in chunks], dim=0),
        # Existing loader supports this old tensor format [N,P,M,4].
        "part_boxes_xyxy_pix": torch.cat([ch["part_boxes_xyxy_pix"] for ch in chunks], dim=0),
        "part_scores": torch.cat([ch["part_scores"] for ch in chunks], dim=0),
        "visible_parts": torch.cat([ch["visible_parts"] for ch in chunks], dim=0),
    }

    cache_path = output_root / "train_part_boxes_augmented_nppl.pt"
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(cache, tmp)
    os.replace(tmp, cache_path)

    manifest_path = output_root / "train_augmented_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        supervised = cache["bbox_supervised"].tolist()
        labels = cache["labels"].tolist()
        ids = cache["source_image_ids"].tolist()
        visible = cache["visible_parts"].tolist()
        for i, rel in enumerate(cache["relpaths"]):
            f.write(
                json.dumps(
                    {
                        "aug_relpath": rel,
                        "source_relpath": cache["source_relpaths"][i],
                        "source_image_id": int(ids[i]),
                        "label": int(labels[i]),
                        "operation": cache["operations"][i],
                        "geometry": cache["geometries"][i],
                        "bbox_supervised": bool(supervised[i]),
                        "visible_parts": [bool(v) for v in visible[i]],
                        "transform": cache["transform_meta"][i],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {
        "cache_path": str(cache_path),
        "manifest_path": str(manifest_path),
        "count": len(cache["relpaths"]),
        "bbox_supervised_count": int(cache["bbox_supervised"].sum().item()),
        "bbox_unsupervised_count": int((~cache["bbox_supervised"]).sum().item()),
    }


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("NPPL-style offline CUB augmentation with synchronized affine GDINO boxes")
    p.add_argument("--cub-root", default="./data/CUB_200_2011")
    p.add_argument("--output-root", default="./data/cub200_cropped_nppl_bboxsync")
    p.add_argument(
        "--gdino-train-cache",
        default="./artifacts/gdino_part_boxes_visible_resize224/train_part_boxes_visible_resize224.pt",
        help="Original train GDINO cache on GT bird crop geometry.",
    )
    p.add_argument("--image-size", type=int, default=224,
                   help="Offline augmented image size and bbox coordinate system.")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--workers", type=int, default=8,
                   help="CPU processes. No GPU is used by this image-writing stage.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-existing-crops", action="store_true")
    p.add_argument("--keep-incomplete-class", action="store_true",
                   help="Do not clear partial class output. Usually leave this off for correctness.")
    p.add_argument("--crops-only", action="store_true")
    p.add_argument("--class-limit", type=int, default=0)
    p.add_argument("--repetitions-per-operation", type=int, default=10)
    p.add_argument("--operations", default="rotate,skew,shear,random_distortion",
                   help="Comma-separated subset. Default matches NPPL/TesNet operation family.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cub_root = Path(args.cub_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    cache_path = Path(args.gdino_train_cache).expanduser().resolve()
    crop_root = output_root / "train_cropped"
    aug_root = output_root / "train_cropped_augmented"
    chunk_dir = output_root / "aug_chunks"

    required = [
        cub_root / "images",
        cub_root / "images.txt",
        cub_root / "train_test_split.txt",
        cub_root / "bounding_boxes.txt",
        cache_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input:\n  - " + "\n  - ".join(missing))

    operations = tuple(x.strip() for x in args.operations.split(",") if x.strip())
    valid_ops = {"rotate", "skew", "shear", "random_distortion"}
    bad = [x for x in operations if x not in valid_ops]
    if bad or not operations:
        raise ValueError(f"Invalid --operations {bad}; allowed={sorted(valid_ops)}")

    output_root.mkdir(parents=True, exist_ok=True)
    aug_root.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    records = build_train_crops(cub_root, crop_root, args.jpeg_quality, args.skip_existing_crops)
    with (output_root / "train_cropped_manifest.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.crops_only:
        print(f"[Done] crops only: {crop_root}")
        return

    attach_source_gdino(records, cache_path, int(args.image_size))
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_class.setdefault(str(rec["class_name"]), []).append(rec)
    class_names = sorted(by_class)
    if args.class_limit > 0:
        class_names = class_names[: int(args.class_limit)]

    cfg = AugConfig(repetitions_per_operation=int(args.repetitions_per_operation))
    expected_total = sum(len(by_class[c]) for c in class_names) * len(operations) * cfg.repetitions_per_operation
    print("[Recipe] NPPL/TesNet operation family with synchronized affine/projective boxes")
    print(json.dumps(asdict(cfg), indent=2))
    print(f"[Aug] operations={operations}")
    print(f"[Aug] source crops={crop_root}")
    print(f"[Aug] output images={aug_root}")
    print(f"[Aug] expected images={expected_total:,}")
    print("[BBox] rotate/skew/shear/flip synchronized; random_distortion saved with invalid boxes.")

    tasks = [
        ClassTask(
            class_name=class_name,
            crop_root=str(crop_root),
            aug_root=str(aug_root),
            chunk_path=str(chunk_dir / f"{class_name}.pt"),
            records=by_class[class_name],
            cfg_dict=asdict(cfg),
            image_size=int(args.image_size),
            jpeg_quality=int(args.jpeg_quality),
            seed=int(args.seed),
            overwrite_incomplete=not args.keep_incomplete_class,
            operations=operations,
        )
        for class_name in class_names
    ]

    results: List[Tuple[str, str, int, int]] = []
    if args.workers <= 1:
        for task in tqdm(tasks, desc="Offline augment classes", ncols=120):
            results.append(augment_one_class(task))
    else:
        with Pool(processes=int(args.workers)) as pool:
            for res in tqdm(pool.imap_unordered(augment_one_class, tasks), total=len(tasks), desc="Offline augment classes", ncols=120):
                results.append(res)

    final = assemble_final_cache(output_root, class_names, int(args.image_size))
    summary = {
        "cub_root": str(cub_root),
        "gdino_train_cache": str(cache_path),
        "output_root": str(output_root),
        "train_cropped": str(crop_root),
        "train_cropped_augmented": str(aug_root),
        "operations": list(operations),
        "augmentation_recipe": asdict(cfg),
        "expected_augmented_images": int(expected_total),
        "actual_augmented_images": int(final["count"]),
        "bbox_supervised_count": int(final["bbox_supervised_count"]),
        "bbox_unsupervised_count": int(final["bbox_unsupervised_count"]),
        "cache_path": final["cache_path"],
        "manifest_path": final["manifest_path"],
        "class_results": [
            {"class_name": c, "status": s, "source_count": src, "output_count": out}
            for c, s, src, out in sorted(results)
        ],
        "note": (
            "This preserves NPPL/TesNet operation types and counts, but uses explicit OpenCV geometry "
            "instead of Augmentor so box transforms are known. It is not bit-for-bit identical to Augmentor."
        ),
    }
    summary_path = output_root / "nppl_offline_augmentation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[Done]")
    print(f"images:   {aug_root}")
    print(f"bbox pt:  {final['cache_path']}")
    print(f"manifest: {final['manifest_path']}")
    print(f"counts:   total={final['count']:,} supervised={final['bbox_supervised_count']:,} nonlinear_no_bbox={final['bbox_unsupervised_count']:,}")


if __name__ == "__main__":
    main()
