#!/usr/bin/env python3
"""Part-gated shared-prototype training on TesNet-style augmented Stanford Dogs.

Train geometry:
  official dog bbox crop -> direct Resize(224,224) -> offline TesNet-style augmentation
  -> DINOv2 ViT-B/14 patch tokens.

Weak semantic routing supervision is applied only when an augmented sample carries
valid part boxes.  Original, rotation and shear samples use synchronized boxes;
skew and grid-distortion samples contain only invalid (-1) boxes and therefore
contribute classification / unsupervised prototype terms, but no part-box losses.

Test geometry remains unchanged:
  original dog image -> official dog bbox crop -> direct Resize(224,224).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision import transforms
from tqdm import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_PARTS = ("head", "ear", "muzzle", "body", "leg", "tail")


@dataclass
class CFG:
    dogs_root: str = "./data/StanfordDogs"
    train_aug_root: str = "./data/stanforddogs_tesnet_aug_bboxsync"
    train_aug_manifest_file: str = "train_tesnet_aug_manifest.pt"
    gdino_box_dir: str = "./artifacts/hf_gdino_stanforddogs_6parts"
    gdino_test_file: str = "test_part_boxes_hf_gdino.pt"
    save_dir: str = "./runs/stanforddogs_shared_6parts_k50_dinov2_vitb14_tesnetaug"

    dino_model: str = "dinov2_vitb14"
    image_size: int = 224
    num_classes: int = 120
    parts: Tuple[str, ...] = DEFAULT_PARTS
    k_per_part: int = 50

    # Only the last two DINOv2 blocks and the final norm are eligible for tuning.
    # The first freeze epochs detach the visual backbone completely while router,
    # prototypes and class readout learn from its frozen patch features.
    unfreeze_last_blocks: int = 2
    unfreeze_norm: bool = True
    freeze_backbone_epochs: int = 10

    # Part-gated prototype evidence.
    score_mode: str = "resp_sum"
    score_scale: float = 8.0
    scan_topk: int = 5
    tau_part: float = 0.20
    tau_proto: float = 0.05
    null_logit_init: float = 0.0
    residual_scale: float = 0.20
    readout_mode: str = "nonneg"
    class_theta_init: float = 0.0
    proto_dropout: float = 0.0

    # Two-timescale prototype memory.
    ema_rho: float = 0.99
    ema_sem_mix: float = 0.35
    ema_min_mass: float = 1e-3
    ema_start_epoch: int = 1
    ema_stop_epoch: int = 0
    ema_every_steps: int = 1

    # Weak GroundingDINO semantic targets.
    box_target_gaussian: bool = True
    box_gaussian_sigma_scale: float = 0.50

    # Objective.
    label_smoothing: float = 0.0
    lambda_ce: float = 1.0
    lambda_route: float = 0.20
    route_final_ratio: float = 0.10
    route_decay_epochs: int = 20
    lambda_vis: float = 0.01
    lambda_proto_lb: float = 0.01
    lambda_proto_agree: float = 0.05
    proto_agree_direction: str = "symmetric"
    lambda_proto_div: float = 0.02
    proto_div_margin: float = 0.30
    lambda_cls_sparse: float = 0.0

    # Optimization.
    lr_backbone: float = 3e-6
    lr_router: float = 1e-4
    lr_proto: float = 1e-4
    lr_classifier: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0

    # Training.
    epochs: int = 50
    batch_size: int = 64
    max_train_batches: int = 0
    num_workers: int = 8
    seed: int = 42
    amp: bool = True

    # Resume.
    resume: bool = True
    resume_from: str = "last"
    reset_optimizer_on_resume: bool = False

    # Semantic memory bootstrap.
    bootstrap_memory: bool = True
    bootstrap_batches: int = 100
    bootstrap_max_tokens_per_part: int = 20000
    bootstrap_kmeans_iters: int = 20

    # I/O.
    allow_incomplete_boxes: bool = False
    eval_every: int = 1
    save_every: int = 1
    eps: float = 1e-9


cfg = CFG()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANK = 0
WORLD_SIZE = 1
LOCAL_RANK = 0


# -----------------------------------------------------------------------------
# Distributed and generic utilities
# -----------------------------------------------------------------------------
def setup_distributed() -> None:
    global DEVICE, RANK, WORLD_SIZE, LOCAL_RANK
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
    if WORLD_SIZE > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP was requested but CUDA is unavailable.")
        RANK = int(os.environ["RANK"])
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(LOCAL_RANK)
        DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
        dist.init_process_group(backend="nccl")
    else:
        RANK, LOCAL_RANK = 0, 0
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_distributed() -> bool:
    return WORLD_SIZE > 1 and dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return RANK == 0


def rank0_print(*args: Any, **kwargs: Any) -> None:
    if is_main_process():
        print(*args, **kwargs)


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def set_seed(seed: int) -> None:
    value = int(seed) + RANK
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def l2n(x: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_torch_load(path: str, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            value = value.detach().float().mean().item()
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def route_lambda(epoch: int) -> float:
    if cfg.route_decay_epochs <= 0:
        return cfg.lambda_route
    progress = min(max((epoch - 1) / float(cfg.route_decay_epochs), 0.0), 1.0)
    factor = 1.0 + progress * (cfg.route_final_ratio - 1.0)
    return cfg.lambda_route * factor


# -----------------------------------------------------------------------------
# Stanford Dogs data and HF GroundingDINO part-cache alignment
# -----------------------------------------------------------------------------
def matlab_string(value: Any) -> str:
    current = value
    while isinstance(current, np.ndarray):
        if current.size == 0:
            raise ValueError("Encountered an empty MATLAB string.")
        current = current.item() if current.size == 1 else current.flat[0]
    if isinstance(current, bytes):
        return current.decode("utf-8")
    return str(current)


def parse_split_mat(path: str) -> List[Tuple[str, int]]:
    data = loadmat(path)
    if "annotation_list" not in data or "labels" not in data:
        raise KeyError(f"{path} needs annotation_list and labels.")

    annotations = data["annotation_list"].squeeze()
    labels = np.asarray(data["labels"]).squeeze()
    if len(annotations) != len(labels):
        raise RuntimeError(f"Mismatch in {path}: {len(annotations)} annotations vs {len(labels)} labels.")

    samples: List[Tuple[str, int]] = []
    for annotation, label in zip(annotations, labels):
        rel_annotation = matlab_string(annotation).replace("\\", "/")
        samples.append((rel_annotation, int(np.asarray(label).item()) - 1))
    return samples


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
    width, height = image.size
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return image.crop((x1, y1, x2, y2))


def build_dogs_samples(dogs_root: str, split: str) -> List[Dict[str, Any]]:
    if split not in {"train", "test"}:
        raise ValueError(f"Unknown split: {split}")

    items = parse_split_mat(os.path.join(dogs_root, f"{split}_list.mat"))
    samples = [
        {
            "rel_annotation": rel_annotation,
            "label": label,
            "image_path": os.path.join(dogs_root, "Images", rel_annotation + ".jpg"),
            "annotation_path": os.path.join(dogs_root, "Annotation", rel_annotation),
        }
        for rel_annotation, label in items
    ]

    labels = [sample["label"] for sample in samples]
    if not labels or min(labels) < 0 or max(labels) >= cfg.num_classes:
        raise RuntimeError(
            f"Unexpected labels: min={min(labels) if labels else None}, "
            f"max={max(labels) if labels else None}, expected [0,{cfg.num_classes - 1}]."
        )
    return samples


def valid_xyxy(raw: Any) -> np.ndarray:
    if raw is None:
        return np.empty((0, 4), dtype=np.float32)
    try:
        if isinstance(raw, torch.Tensor):
            array = raw.detach().cpu().float().numpy()
        else:
            array = np.asarray(raw, dtype=np.float32)
    except Exception:
        return np.empty((0, 4), dtype=np.float32)

    if array.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if array.ndim == 1 and array.size == 4:
        array = array.reshape(1, 4)
    elif array.ndim >= 2 and array.shape[-1] == 4:
        array = array.reshape(-1, 4)
    else:
        return np.empty((0, 4), dtype=np.float32)

    valid = (
        np.isfinite(array).all(axis=1)
        & (array[:, 0] >= 0)
        & (array[:, 1] >= 0)
        & (array[:, 2] > array[:, 0])
        & (array[:, 3] > array[:, 1])
    )
    return array[valid].astype(np.float32, copy=False)


def load_aligned_part_boxes(samples: Sequence[Dict[str, Any]], cache_path: str) -> torch.Tensor:
    payload = safe_torch_load(cache_path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), dict):
        raise TypeError(f"Part cache must be a dict containing records: {cache_path}")

    meta = payload.get("meta", {})
    records: Dict[str, Any] = payload["records"]
    expected = int(meta.get("expected_images", 0))
    processed = int(meta.get("processed_images", len(records)))
    if not cfg.allow_incomplete_boxes and (expected <= 0 or processed < expected):
        raise RuntimeError(
            f"Part cache is incomplete: processed={processed}, expected={expected}. "
            "Wait for GDINO to finish, or explicitly pass --allow-incomplete-boxes."
        )

    file_parts_raw = meta.get("parts", [])
    file_parts = {
        str(item.get("name", "")).lower()
        for item in file_parts_raw
        if isinstance(item, dict)
    }
    missing_parts = [part for part in cfg.parts if part not in file_parts]
    if missing_parts:
        raise KeyError(
            f"Part cache does not declare requested parts {missing_parts}; declared={sorted(file_parts)}"
        )

    source_size = int(meta.get("resized_coordinate_size", cfg.image_size))
    scale = float(cfg.image_size) / max(1.0, float(source_size))

    temp: List[List[np.ndarray]] = []
    max_boxes = 1
    missing_records = 0
    per_part_boxes = {part: 0 for part in cfg.parts}

    for sample in samples:
        rel_annotation = str(sample["rel_annotation"])
        record = records.get(rel_annotation)
        image_parts: List[np.ndarray] = []

        if not isinstance(record, dict):
            missing_records += 1
            image_parts = [np.empty((0, 4), dtype=np.float32) for _ in cfg.parts]
        else:
            part_dict = record.get("parts", {})
            for part in cfg.parts:
                entry = part_dict.get(part, {}) if isinstance(part_dict, dict) else {}
                boxes = valid_xyxy(entry.get("boxes_xyxy_resize") if isinstance(entry, dict) else None)
                if scale != 1.0 and boxes.size:
                    boxes = boxes * scale
                image_parts.append(boxes)
                max_boxes = max(max_boxes, int(boxes.shape[0]))
                per_part_boxes[part] += int(boxes.shape[0])
        temp.append(image_parts)

    aligned = torch.full(
        (len(samples), len(cfg.parts), max_boxes, 4),
        -1.0,
        dtype=torch.float32,
    )
    for sample_index, image_parts in enumerate(temp):
        for part_index, boxes in enumerate(image_parts):
            if boxes.size == 0:
                continue
            count = min(max_boxes, boxes.shape[0])
            aligned[sample_index, part_index, :count] = torch.from_numpy(boxes[:count])

    valid = (
        (aligned[..., 0] >= 0)
        & (aligned[..., 2] > aligned[..., 0])
        & (aligned[..., 3] > aligned[..., 1])
    )
    rank0_print(
        f"[GDINO] {os.path.basename(cache_path)} matched={len(samples) - missing_records}/{len(samples)}; "
        f"source_resize={source_size} -> model_resize={cfg.image_size}; "
        f"valid_part_ratio={valid.any(dim=-1).float().mean().item():.4f}; "
        f"max_boxes_per_part={max_boxes}; selected_boxes={per_part_boxes}"
    )
    return aligned


class StanfordDogsWithPartBoxes(Dataset):
    def __init__(self, split: str, cache_path: str):
        self.samples = build_dogs_samples(cfg.dogs_root, split)
        self.boxes = load_aligned_part_boxes(self.samples, cache_path)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (cfg.image_size, cfg.image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample["image_path"]) as image:
            image = image.convert("RGB")
        image = crop_xyxy(image, parse_annotation_union_box(sample["annotation_path"]))
        return (
            self.transform(image),
            self.boxes[index],
            int(sample["label"]),
            str(sample["rel_annotation"]),
        )


# -----------------------------------------------------------------------------
# Offline TesNet-style train augmentation dataset
# -----------------------------------------------------------------------------
def load_tesnet_aug_manifest(path: str) -> Dict[str, Any]:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Augmentation manifest must be a dict: {path}")

    required = ("meta", "variants", "source_rel_annotations", "source_labels", "box_shards")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Augmentation manifest missing keys {missing}: {path}")
    return payload


def augmented_image_relative_path(rel_annotation: str, variant: Dict[str, Any]) -> str:
    parent, stem = os.path.split(rel_annotation)
    family = str(variant.get("family", ""))
    name = str(variant.get("name", ""))
    if family == "original":
        return os.path.join("train_cropped", parent, stem + ".jpg")
    return os.path.join("train_cropped_augmented", parent, f"{stem}_{name}.jpg")


def tensor_to_int_list(value: Any) -> List[int]:
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in np.asarray(value).reshape(-1).tolist()]


def load_tesnet_augmented_boxes(
    manifest: Dict[str, Any],
    aug_root: str,
) -> torch.Tensor:
    """Load all shard boxes into [N, V, P, M, 4] in manifest source order."""
    meta = manifest["meta"]
    source_rel = [str(x) for x in manifest["source_rel_annotations"]]
    variants = manifest["variants"]
    part_names = [str(x).lower() for x in meta.get("parts", [])]
    if tuple(part_names) != tuple(cfg.parts):
        raise ValueError(
            f"Augmentation manifest parts={part_names}, but requested --parts={list(cfg.parts)}."
        )

    n_sources = len(source_rel)
    n_variants = len(variants)
    expected_max_boxes = int(meta.get("max_boxes", 0))
    shard_paths = [os.path.join(aug_root, str(path)) for path in manifest["box_shards"]]
    if not shard_paths:
        raise RuntimeError("Augmentation manifest declares no box shards.")

    loaded_chunks: List[torch.Tensor] = []
    expected_start = 0
    max_boxes: Optional[int] = None

    for shard_path in shard_paths:
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(f"Missing augmentation box shard: {shard_path}")
        shard = safe_torch_load(shard_path, map_location="cpu")
        if not isinstance(shard, dict) or not isinstance(shard.get("boxes_xyxy_resize"), torch.Tensor):
            raise TypeError(f"Invalid augmentation box shard: {shard_path}")

        source_start = int(shard.get("source_start", -1))
        source_end = int(shard.get("source_end", -1))
        boxes = shard["boxes_xyxy_resize"].float().contiguous()

        if source_start != expected_start:
            raise RuntimeError(
                f"Shard order gap/overlap at {shard_path}: expected start={expected_start}, got={source_start}."
            )
        if source_end <= source_start or boxes.shape[0] != source_end - source_start:
            raise RuntimeError(f"Invalid source span in augmentation shard: {shard_path}")
        if boxes.ndim != 5 or boxes.shape[1] != n_variants or boxes.shape[2] != len(cfg.parts) or boxes.shape[-1] != 4:
            raise RuntimeError(
                f"Unexpected shard box shape {tuple(boxes.shape)} in {shard_path}; "
                f"expected [N,{n_variants},{len(cfg.parts)},M,4]."
            )

        if max_boxes is None:
            max_boxes = int(boxes.shape[3])
        elif int(boxes.shape[3]) != max_boxes:
            raise RuntimeError(f"Inconsistent max-box dimension across shards at {shard_path}.")

        shard_rel = [str(x) for x in shard.get("source_rel_annotations", [])]
        if shard_rel and shard_rel != source_rel[source_start:source_end]:
            raise RuntimeError(f"Source ordering mismatch in augmentation shard: {shard_path}")

        loaded_chunks.append(boxes)
        expected_start = source_end

    if expected_start != n_sources:
        raise RuntimeError(
            f"Augmentation shards cover {expected_start} sources, manifest declares {n_sources}."
        )

    all_boxes = torch.cat(loaded_chunks, dim=0).contiguous()
    if expected_max_boxes > 0 and int(all_boxes.shape[3]) != expected_max_boxes:
        raise RuntimeError(
            f"Manifest max_boxes={expected_max_boxes}, loaded max_boxes={all_boxes.shape[3]}."
        )
    return all_boxes


class StanfordDogsTesNetAugmented(Dataset):
    """All offline TesNet-style train images with their per-variant weak boxes.

    For nonlinear skew/distortion variants, the builder stores -1 for all boxes.
    `boxes_to_soft_targets` treats those parts as invalid, so no routing, visibility,
    prototype-agreement, or semantic-EMA supervision is applied to them.
    """

    def __init__(self, aug_root: str, manifest_file: str, only_original: bool = False):
        self.aug_root = os.path.abspath(aug_root)
        manifest_path = os.path.join(self.aug_root, manifest_file)
        self.manifest = load_tesnet_aug_manifest(manifest_path)
        self.variants: List[Dict[str, Any]] = [dict(item) for item in self.manifest["variants"]]
        self.source_rel_annotations = [str(item) for item in self.manifest["source_rel_annotations"]]
        self.source_labels = tensor_to_int_list(self.manifest["source_labels"])

        if len(self.source_rel_annotations) != len(self.source_labels):
            raise RuntimeError(
                f"Augmentation manifest has {len(self.source_rel_annotations)} sources but "
                f"{len(self.source_labels)} labels."
            )
        if not self.variants:
            raise RuntimeError("Augmentation manifest contains no variants.")

        meta = self.manifest["meta"]
        manifest_size = int(meta.get("image_size", cfg.image_size))
        if manifest_size != cfg.image_size:
            raise ValueError(
                f"Augmentation images are {manifest_size}x{manifest_size}; "
                f"requested --image-size={cfg.image_size}."
            )

        self.boxes = load_tesnet_augmented_boxes(self.manifest, self.aug_root)
        if self.boxes.shape[0] != len(self.source_rel_annotations):
            raise RuntimeError("Augmented box tensor / manifest source count mismatch.")

        if only_original:
            self.variant_indices = [
                index for index, variant in enumerate(self.variants)
                if str(variant.get("family", "")) == "original"
            ]
        else:
            self.variant_indices = list(range(len(self.variants)))

        if not self.variant_indices:
            raise RuntimeError("No selected augmentation variants.")

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        valid = (
            (self.boxes[..., 0] >= 0)
            & (self.boxes[..., 2] > self.boxes[..., 0])
            & (self.boxes[..., 3] > self.boxes[..., 1])
        )
        selected_valid = valid[:, self.variant_indices]
        supervised_parts = selected_valid.any(dim=-1).float().mean().item()
        nonlinear_indices = [
            idx for idx, variant in enumerate(self.variants)
            if str(variant.get("family", "")) in {"skew", "distortion"}
        ]
        if nonlinear_indices:
            nonlinear_valid = valid[:, nonlinear_indices].any(dim=-1).any().item()
            if nonlinear_valid:
                raise RuntimeError(
                    "Nonlinear skew/distortion variants unexpectedly contain valid weak boxes. "
                    "Rebuild the augmentation cache before training."
                )

        rank0_print(
            f"[AugData] manifest={os.path.basename(manifest_path)}; sources={len(self.source_rel_annotations)}; "
            f"selected_variants={len(self.variant_indices)}/{len(self.variants)}; "
            f"train_samples={len(self):,}; max_boxes={self.boxes.shape[3]}; "
            f"supervised_part_ratio={supervised_parts:.4f}; only_original={only_original}"
        )

    def __len__(self) -> int:
        return len(self.source_rel_annotations) * len(self.variant_indices)

    def __getitem__(self, index: int):
        variants_per_source = len(self.variant_indices)
        source_index = int(index) // variants_per_source
        variant_index = self.variant_indices[int(index) % variants_per_source]

        rel_annotation = self.source_rel_annotations[source_index]
        variant = self.variants[variant_index]
        image_path = os.path.join(
            self.aug_root,
            augmented_image_relative_path(rel_annotation, variant),
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if image.size != (cfg.image_size, cfg.image_size):
            raise RuntimeError(
                f"Offline augmented image must already be {cfg.image_size}x{cfg.image_size}, "
                f"got {image.size}: {image_path}"
            )

        return (
            self.transform(image),
            self.boxes[source_index, variant_index],
            int(self.source_labels[source_index]),
            f"{rel_annotation}::{variant.get('name', variant_index)}",
        )


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation without padding/duplicating samples across DDP ranks."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __iter__(self) -> Iterator[int]:
        return iter(range(RANK, len(self.dataset), WORLD_SIZE))

    def __len__(self) -> int:
        return (len(self.dataset) + WORLD_SIZE - 1 - RANK) // WORLD_SIZE


def make_loader(dataset: Dataset, train: bool) -> DataLoader:
    sampler: Optional[Sampler[int]]
    if is_distributed():
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=WORLD_SIZE,
                rank=RANK,
                shuffle=True,
                drop_last=False,
            )
            if train
            else DistributedEvalSampler(dataset)
        )
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(train and sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        drop_last=False,
    )


# -----------------------------------------------------------------------------
# Weak part-box targets
# -----------------------------------------------------------------------------
def boxes_to_soft_targets(
    boxes: torch.Tensor,
    grid_h: int,
    grid_w: int,
    image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert [B,P,M,4] boxes into patch targets.

    Invalid (-1) entries produce valid_bp=False.  Consequently no route,
    visibility, prototype-agreement, or semantic-memory supervision is applied
    for those image-part pairs.  This is used for all nonlinear augmentation
    variants and for genuinely absent detected parts.
    """
    if boxes.ndim == 3:
        boxes = boxes.unsqueeze(2)
    if boxes.ndim != 4:
        raise ValueError(f"Expected [B,P,M,4], got {tuple(boxes.shape)}")

    batch, parts, max_boxes, _ = boxes.shape
    device = boxes.device
    ys = (torch.arange(grid_h, device=device, dtype=torch.float32) + 0.5) * (image_size / grid_h)
    xs = (torch.arange(grid_w, device=device, dtype=torch.float32) + 0.5) * (image_size / grid_w)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    targets = torch.zeros((batch, parts, grid_h, grid_w), device=device, dtype=torch.float32)
    valid = torch.zeros((batch, parts), device=device, dtype=torch.bool)

    for batch_index in range(batch):
        for part_index in range(parts):
            combined = torch.zeros((grid_h, grid_w), device=device, dtype=torch.float32)
            for box_index in range(max_boxes):
                x1, y1, x2, y2 = boxes[batch_index, part_index, box_index]
                if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                    continue

                x1 = x1.clamp(0, image_size)
                y1 = y1.clamp(0, image_size)
                x2 = x2.clamp(0, image_size)
                y2 = y2.clamp(0, image_size)
                mask = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)

                # Keep a one-patch target for tiny but valid detector boxes.
                if not bool(mask.any()):
                    cx = 0.5 * (x1 + x2)
                    cy = 0.5 * (y1 + y2)
                    nearest = ((xx - cx).square() + (yy - cy).square()).flatten().argmin()
                    mask = torch.zeros_like(mask)
                    mask.flatten()[nearest] = True

                if cfg.box_target_gaussian:
                    cx = 0.5 * (x1 + x2)
                    cy = 0.5 * (y1 + y2)
                    sx = ((x2 - x1) * cfg.box_gaussian_sigma_scale).clamp_min(image_size / grid_w)
                    sy = ((y2 - y1) * cfg.box_gaussian_sigma_scale).clamp_min(image_size / grid_h)
                    weights = torch.exp(
                        -0.5 * (((xx - cx) / sx).square() + ((yy - cy) / sy).square())
                    ) * mask.float()
                else:
                    weights = mask.float()
                combined += weights

            if combined.sum() > 0:
                targets[batch_index, part_index] = combined / combined.sum().clamp_min(cfg.eps)
                valid[batch_index, part_index] = True

    return targets.flatten(2), valid


# -----------------------------------------------------------------------------
# DINOv2 feature extractor
# -----------------------------------------------------------------------------
def load_dinov2() -> nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", cfg.dino_model)
    if not hasattr(model, "blocks"):
        raise RuntimeError(f"Unexpected DINOv2 backbone: {cfg.dino_model}")
    return model


def extract_patch_tokens(backbone: nn.Module, images: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    output = backbone.forward_features(images)
    if not isinstance(output, dict) or output.get("x_norm_patchtokens") is None:
        raise RuntimeError("DINOv2 forward_features did not return x_norm_patchtokens.")
    tokens = output["x_norm_patchtokens"].float()

    count = int(tokens.shape[1])
    grid = int(round(math.sqrt(count)))
    if grid * grid != count:
        raise RuntimeError(f"Patch-token count must form a square grid, got {count}.")
    return tokens, grid, grid


def set_backbone_trainability(backbone: nn.Module) -> int:
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    blocks = list(backbone.blocks)
    if cfg.unfreeze_last_blocks < 0 or cfg.unfreeze_last_blocks > len(blocks):
        raise ValueError(
            f"--unfreeze-last-blocks must be in [0,{len(blocks)}], got {cfg.unfreeze_last_blocks}."
        )

    if cfg.unfreeze_last_blocks > 0:
        for block in blocks[-cfg.unfreeze_last_blocks :]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)

    if cfg.unfreeze_norm and hasattr(backbone, "norm"):
        for parameter in backbone.norm.parameters():
            parameter.requires_grad_(True)

    return sum(parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad)


# -----------------------------------------------------------------------------
# Part-gated prototype model
# -----------------------------------------------------------------------------
class SharedPartPrototypeModel(nn.Module):
    def __init__(self, backbone: nn.Module, dim: int, parts: int, k: int, classes: int):
        super().__init__()
        self.backbone = backbone
        self.dim = int(dim)
        self.parts = int(parts)
        self.k = int(k)
        self.classes = int(classes)

        self.part_queries = nn.Parameter(torch.randn(parts, dim) * 0.02)
        self.null_logits = nn.Parameter(torch.full((parts,), float(cfg.null_logit_init)))

        # M is the slow semantic memory; R is the gradient-trained fast residual.
        self.register_buffer("memory", l2n(torch.randn(parts, k, dim), dim=-1, eps=cfg.eps))
        self.proto_residual = nn.Parameter(torch.zeros(parts, k, dim))

        self.class_theta = nn.Parameter(
            torch.full((classes, parts, k), float(cfg.class_theta_init))
        )
        self.class_bias = nn.Parameter(torch.zeros(classes))

    def extract_tokens(self, images: torch.Tensor, detach_backbone: bool) -> Tuple[torch.Tensor, int, int]:
        if detach_backbone:
            with torch.no_grad():
                tokens, grid_h, grid_w = extract_patch_tokens(self.backbone, images)
            return tokens.detach(), grid_h, grid_w
        return extract_patch_tokens(self.backbone, images)

    def effective_prototypes(self) -> torch.Tensor:
        residual = cfg.residual_scale * torch.tanh(self.proto_residual.float())
        return l2n(self.memory.float() + residual, dim=-1, eps=cfg.eps)

    def class_weights(self) -> torch.Tensor:
        if cfg.readout_mode == "nonneg":
            return F.softplus(self.class_theta.float())
        if cfg.readout_mode == "signed":
            return self.class_theta.float()
        raise ValueError(f"Unknown readout mode: {cfg.readout_mode}")

    def forward(self, images: torch.Tensor, detach_backbone: bool = False) -> Dict[str, Any]:
        tokens, grid_h, grid_w = self.extract_tokens(images, detach_backbone=detach_backbone)
        normalized_tokens = l2n(tokens, dim=-1, eps=cfg.eps)

        part_logits = torch.einsum(
            "bnc,pc->bpn",
            normalized_tokens,
            l2n(self.part_queries.float(), dim=-1, eps=cfg.eps),
        ) / max(cfg.tau_part, 1e-6)

        null_logits = self.null_logits.float().view(1, self.parts, 1).expand(
            normalized_tokens.shape[0], -1, -1
        )
        part_prob_all = F.softmax(torch.cat([part_logits, null_logits], dim=-1), dim=-1)
        part_map = part_prob_all[..., :-1]
        visibility = 1.0 - part_prob_all[..., -1]

        prototypes = self.effective_prototypes()
        similarity = torch.einsum("bnc,pkc->bpnk", normalized_tokens, prototypes)
        proto_assign = F.softmax(similarity / max(cfg.tau_proto, 1e-6), dim=-1)
        responsibility = part_map.unsqueeze(-1) * proto_assign
        relu_similarity = F.relu(similarity)

        if cfg.score_mode == "resp_sum":
            raw_score = (responsibility * relu_similarity).sum(dim=2)
        elif cfg.score_mode == "scan_max":
            raw_score = relu_similarity.max(dim=2).values
        elif cfg.score_mode == "scan_topk":
            topk = min(max(1, cfg.scan_topk), relu_similarity.shape[2])
            raw_score = relu_similarity.topk(topk, dim=2).values.mean(dim=2)
        elif cfg.score_mode == "part_max":
            raw_score = (part_map.unsqueeze(-1) * relu_similarity).max(dim=2).values
        elif cfg.score_mode == "part_topk":
            gated = part_map.unsqueeze(-1) * relu_similarity
            topk = min(max(1, cfg.scan_topk), gated.shape[2])
            raw_score = gated.topk(topk, dim=2).values.mean(dim=2)
        else:
            raise ValueError(f"Unknown score mode: {cfg.score_mode}")

        proto_score = raw_score * cfg.score_scale
        if self.training and cfg.proto_dropout > 0:
            score_for_readout = F.dropout(proto_score, p=cfg.proto_dropout, training=True)
        else:
            score_for_readout = proto_score

        weights = self.class_weights()
        contributions = score_for_readout[:, None] * weights[None]
        part_evidence = contributions.sum(dim=-1)
        logits = self.class_bias.float().view(1, -1) + part_evidence.sum(dim=-1)

        return {
            "logits": logits,
            "tokens": normalized_tokens,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "part_map": part_map,
            "visibility": visibility,
            "prototypes": prototypes,
            "similarity": similarity,
            "proto_assign": proto_assign,
            "responsibility": responsibility,
            "utilization": responsibility.sum(dim=2),
            "proto_score": proto_score,
            "class_weight": weights,
            "contributions": contributions,
            "part_evidence": part_evidence,
        }

    @torch.no_grad()
    def ema_update_memory(
        self,
        tokens: torch.Tensor,
        part_map: torch.Tensor,
        proto_assign: torch.Tensor,
        q_sem: torch.Tensor,
    ) -> None:
        """EMA semantic memory: self-routed evidence mixed with box-routed evidence."""
        self_resp = part_map.unsqueeze(-1) * proto_assign
        self_mass = self_resp.sum(dim=(0, 2))
        self_num = torch.einsum("bpnk,bnc->pkc", self_resp, tokens)

        semantic_resp = q_sem.unsqueeze(-1) * proto_assign
        semantic_mass = semantic_resp.sum(dim=(0, 2))
        semantic_num = torch.einsum("bpnk,bnc->pkc", semantic_resp, tokens)

        if is_distributed():
            for value in (self_mass, self_num, semantic_mass, semantic_num):
                dist.all_reduce(value, op=dist.ReduceOp.SUM)

        self_target = l2n(
            self_num / self_mass.unsqueeze(-1).clamp_min(cfg.eps),
            dim=-1,
            eps=cfg.eps,
        )
        semantic_target = l2n(
            semantic_num / semantic_mass.unsqueeze(-1).clamp_min(cfg.eps),
            dim=-1,
            eps=cfg.eps,
        )
        semantic_valid = semantic_mass > cfg.ema_min_mass
        mixed_target = torch.where(
            semantic_valid.unsqueeze(-1),
            (1.0 - cfg.ema_sem_mix) * self_target + cfg.ema_sem_mix * semantic_target,
            self_target,
        )
        valid_update = self_mass > cfg.ema_min_mass
        updated_memory = l2n(
            cfg.ema_rho * self.memory.float() + (1.0 - cfg.ema_rho) * mixed_target,
            dim=-1,
            eps=cfg.eps,
        )
        self.memory.copy_(
            torch.where(valid_update.unsqueeze(-1), updated_memory, self.memory.float())
        )


# -----------------------------------------------------------------------------
# Losses and memory bootstrap
# -----------------------------------------------------------------------------
def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def compute_losses(
    output: Dict[str, Any],
    labels: torch.Tensor,
    q_sem: torch.Tensor,
    valid_bp: torch.Tensor,
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ce = F.cross_entropy(output["logits"], labels, label_smoothing=cfg.label_smoothing)

    part_map = output["part_map"].float()
    route_ce = -(q_sem * torch.log(part_map.clamp_min(cfg.eps))).sum(dim=-1)
    route = masked_mean(route_ce, valid_bp)

    visibility_loss = masked_mean(
        -torch.log(output["visibility"].float().clamp_min(cfg.eps)),
        valid_bp,
    )

    # Each child prototype's normalized spatial responsibility should remain
    # anchored to its parent part's local routing distribution.
    part_dist = part_map / part_map.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)
    proto_maps = output["responsibility"].permute(0, 1, 3, 2).float()
    proto_dist = proto_maps / proto_maps.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)
    part_expand = part_dist.unsqueeze(2).expand_as(proto_dist)

    kl_part_to_proto = (
        part_expand
        * (torch.log(part_expand.clamp_min(cfg.eps)) - torch.log(proto_dist.clamp_min(cfg.eps)))
    ).sum(dim=-1)
    kl_proto_to_part = (
        proto_dist
        * (torch.log(proto_dist.clamp_min(cfg.eps)) - torch.log(part_expand.clamp_min(cfg.eps)))
    ).sum(dim=-1)

    if cfg.proto_agree_direction == "part_to_proto":
        proto_agree_each = kl_part_to_proto
    elif cfg.proto_agree_direction == "proto_to_part":
        proto_agree_each = kl_proto_to_part
    elif cfg.proto_agree_direction == "symmetric":
        proto_agree_each = 0.5 * (kl_part_to_proto + kl_proto_to_part)
    else:
        raise ValueError(f"Unknown proto_agree_direction: {cfg.proto_agree_direction}")
    proto_agree = masked_mean(proto_agree_each.mean(dim=-1), valid_bp)

    utilization = output["utilization"].float().mean(dim=0)
    util_dist = utilization / utilization.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)
    proto_lb = -(
        torch.log(util_dist.clamp_min(cfg.eps)).mean(dim=-1) + math.log(util_dist.shape[-1])
    ).mean()

    prototypes = output["prototypes"].float()
    cosine = torch.einsum("pkc,plc->pkl", prototypes, prototypes)
    eye = torch.eye(cosine.shape[-1], device=cosine.device, dtype=torch.bool).unsqueeze(0)
    proto_div = F.relu(cosine - cfg.proto_div_margin).masked_select(~eye).mean()

    cls_sparse = output["class_weight"].abs().mean()

    total = (
        cfg.lambda_ce * ce
        + route_lambda(epoch) * route
        + cfg.lambda_vis * visibility_loss
        + cfg.lambda_proto_lb * proto_lb
        + cfg.lambda_proto_agree * proto_agree
        + cfg.lambda_proto_div * proto_div
        + cfg.lambda_cls_sparse * cls_sparse
    )
    return total, {
        "loss": finite_float(total),
        "ce": finite_float(ce),
        "route": finite_float(route),
        "vis": finite_float(visibility_loss),
        "proto_lb": finite_float(proto_lb),
        "proto_agree": finite_float(proto_agree),
        "proto_div": finite_float(proto_div),
        "cls_sparse": finite_float(cls_sparse),
        "visibility": finite_float(output["visibility"]),
    }


@torch.no_grad()
def run_kmeans(vectors: torch.Tensor, k: int, iterations: int) -> torch.Tensor:
    vectors = l2n(vectors.float(), dim=-1, eps=cfg.eps)
    if vectors.shape[0] < k:
        indices = torch.randint(vectors.shape[0], (k,), device=vectors.device)
        return vectors[indices]

    centers = vectors[torch.randperm(vectors.shape[0], device=vectors.device)[:k]].clone()
    for _ in range(max(1, iterations)):
        assignment = (vectors @ centers.t()).argmax(dim=1)
        next_centers: List[torch.Tensor] = []
        for center_index in range(k):
            cluster = vectors[assignment == center_index]
            next_centers.append(cluster.mean(dim=0) if cluster.numel() else centers[center_index])
        centers = l2n(torch.stack(next_centers), dim=-1, eps=cfg.eps)
    return centers


@torch.no_grad()
def bootstrap_memory(model: SharedPartPrototypeModel, loader: DataLoader) -> None:
    if not cfg.bootstrap_memory:
        rank0_print("[Bootstrap] skipped.")
        return

    rank0_print(f"[Bootstrap] collecting up to {cfg.bootstrap_batches} batches of weak part tokens.")
    collected: List[List[torch.Tensor]] = [[] for _ in range(model.parts)]
    model.eval()
    model.backbone.eval()

    iterator = tqdm(loader, desc="Bootstrap", dynamic_ncols=True, disable=not is_main_process())
    for step, (images, boxes, _, _) in enumerate(iterator):
        if cfg.bootstrap_batches > 0 and step >= cfg.bootstrap_batches:
            break
        images = images.to(DEVICE, non_blocking=True)
        boxes = boxes.to(DEVICE, non_blocking=True)
        tokens, grid_h, grid_w = model.extract_tokens(images, detach_backbone=True)
        tokens = l2n(tokens, dim=-1, eps=cfg.eps)
        q_sem, valid_bp = boxes_to_soft_targets(boxes, grid_h, grid_w, cfg.image_size)

        for part_index in range(model.parts):
            for batch_index in torch.where(valid_bp[:, part_index])[0].tolist():
                support = torch.where(q_sem[batch_index, part_index] > 0)[0]
                if support.numel() > 0:
                    collected[part_index].append(tokens[batch_index, support].detach().cpu())

    for part_index, chunks in enumerate(collected):
        if not chunks:
            rank0_print(f"[Bootstrap] {cfg.parts[part_index]}: no valid detector support; kept random memory.")
            continue
        vectors = torch.cat(chunks, dim=0)
        if vectors.shape[0] > cfg.bootstrap_max_tokens_per_part:
            indices = torch.randperm(vectors.shape[0])[:cfg.bootstrap_max_tokens_per_part]
            vectors = vectors[indices]
        centers = run_kmeans(vectors.to(DEVICE), model.k, cfg.bootstrap_kmeans_iters)
        model.memory[part_index].copy_(centers)
        rank0_print(f"[Bootstrap] {cfg.parts[part_index]}: {vectors.shape[0]} tokens -> {model.k} centers.")

    model.train()


# -----------------------------------------------------------------------------
# Optimizer, evaluation, checkpoints
# -----------------------------------------------------------------------------
def build_optimizer(model: SharedPartPrototypeModel) -> torch.optim.Optimizer:
    backbone_params = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    if not backbone_params:
        raise RuntimeError("No DINOv2 parameters were marked trainable.")
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr_backbone, "name": "backbone"},
            {"params": [model.part_queries, model.null_logits], "lr": cfg.lr_router, "name": "router"},
            {"params": [model.proto_residual], "lr": cfg.lr_proto, "name": "prototype"},
            {
                "params": [model.class_theta, model.class_bias],
                "lr": cfg.lr_classifier,
                "name": "classifier",
            },
        ],
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    autocast_enabled = cfg.amp and DEVICE.type == "cuda"

    for images, _, labels, _ in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=autocast_enabled):
            logits = model(images, detach_backbone=True)["logits"]
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())

    if is_distributed():
        stats = torch.tensor([correct, total], device=DEVICE, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        correct, total = int(stats[0].item()), int(stats[1].item())
    return correct / max(1, total)


def resolve_resume_path() -> Optional[str]:
    value = str(cfg.resume_from).strip()
    if not cfg.resume or value.lower() in {"", "none", "no"}:
        return None
    if value == "last":
        return os.path.join(cfg.save_dir, "last.pth")
    if value == "best":
        return os.path.join(cfg.save_dir, "best.pth")
    return os.path.abspath(os.path.expanduser(value))


def build_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def save_checkpoint(
    path: str,
    epoch: int,
    best_acc: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Any,
) -> None:
    if not is_main_process():
        return
    core = unwrap_model(model)
    payload = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model": core.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "config": asdict(cfg),
    }
    temporary_path = path + ".tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def append_metrics(record: Dict[str, Any]) -> None:
    with open(os.path.join(cfg.save_dir, "metrics.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


# -----------------------------------------------------------------------------
# CLI / run
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Part-gated shared prototype training on Stanford Dogs")

    parser.add_argument("--dogs-root", default=cfg.dogs_root)
    parser.add_argument("--train-aug-root", default=cfg.train_aug_root)
    parser.add_argument("--train-aug-manifest-file", default=cfg.train_aug_manifest_file)
    parser.add_argument("--gdino-box-dir", default=cfg.gdino_box_dir)
    parser.add_argument("--gdino-test-file", default=cfg.gdino_test_file)
    parser.add_argument("--save-dir", default=cfg.save_dir)

    parser.add_argument("--dino-model", default=cfg.dino_model)
    parser.add_argument("--image-size", type=int, default=cfg.image_size)
    parser.add_argument("--num-classes", type=int, default=cfg.num_classes)
    parser.add_argument("--parts", default=",".join(cfg.parts))
    parser.add_argument("--k-per-part", type=int, default=cfg.k_per_part)

    parser.add_argument("--unfreeze-last-blocks", type=int, default=cfg.unfreeze_last_blocks)
    parser.add_argument("--unfreeze-norm", action=argparse.BooleanOptionalAction, default=cfg.unfreeze_norm)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=cfg.freeze_backbone_epochs)

    parser.add_argument(
        "--score-mode",
        choices=["resp_sum", "scan_max", "scan_topk", "part_max", "part_topk"],
        default=cfg.score_mode,
    )
    parser.add_argument("--score-scale", type=float, default=cfg.score_scale)
    parser.add_argument("--scan-topk", type=int, default=cfg.scan_topk)
    parser.add_argument("--tau-part", type=float, default=cfg.tau_part)
    parser.add_argument("--tau-proto", type=float, default=cfg.tau_proto)
    parser.add_argument("--null-logit-init", type=float, default=cfg.null_logit_init)
    parser.add_argument("--residual-scale", type=float, default=cfg.residual_scale)
    parser.add_argument("--readout-mode", choices=["nonneg", "signed"], default=cfg.readout_mode)
    parser.add_argument("--class-theta-init", type=float, default=cfg.class_theta_init)
    parser.add_argument("--proto-dropout", type=float, default=cfg.proto_dropout)

    parser.add_argument("--ema-rho", type=float, default=cfg.ema_rho)
    parser.add_argument("--ema-sem-mix", type=float, default=cfg.ema_sem_mix)
    parser.add_argument("--ema-min-mass", type=float, default=cfg.ema_min_mass)
    parser.add_argument("--ema-start-epoch", type=int, default=cfg.ema_start_epoch)
    parser.add_argument("--ema-stop-epoch", type=int, default=cfg.ema_stop_epoch)
    parser.add_argument("--ema-every-steps", type=int, default=cfg.ema_every_steps)

    parser.add_argument("--box-target-gaussian", action=argparse.BooleanOptionalAction, default=cfg.box_target_gaussian)
    parser.add_argument("--box-gaussian-sigma-scale", type=float, default=cfg.box_gaussian_sigma_scale)

    parser.add_argument("--label-smoothing", type=float, default=cfg.label_smoothing)
    parser.add_argument("--lambda-ce", type=float, default=cfg.lambda_ce)
    parser.add_argument("--lambda-route", type=float, default=cfg.lambda_route)
    parser.add_argument("--route-final-ratio", type=float, default=cfg.route_final_ratio)
    parser.add_argument("--route-decay-epochs", type=int, default=cfg.route_decay_epochs)
    parser.add_argument("--lambda-vis", type=float, default=cfg.lambda_vis)
    parser.add_argument("--lambda-proto-lb", type=float, default=cfg.lambda_proto_lb)
    parser.add_argument("--lambda-proto-agree", type=float, default=cfg.lambda_proto_agree)
    parser.add_argument(
        "--proto-agree-direction",
        choices=["part_to_proto", "proto_to_part", "symmetric"],
        default=cfg.proto_agree_direction,
    )
    parser.add_argument("--lambda-proto-div", type=float, default=cfg.lambda_proto_div)
    parser.add_argument("--proto-div-margin", type=float, default=cfg.proto_div_margin)
    parser.add_argument("--lambda-cls-sparse", type=float, default=cfg.lambda_cls_sparse)

    parser.add_argument("--lr-backbone", type=float, default=cfg.lr_backbone)
    parser.add_argument("--lr-router", type=float, default=cfg.lr_router)
    parser.add_argument("--lr-proto", type=float, default=cfg.lr_proto)
    parser.add_argument("--lr-classifier", type=float, default=cfg.lr_classifier)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=cfg.grad_clip)

    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--max-train-batches", type=int, default=cfg.max_train_batches)
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=cfg.amp)

    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=cfg.resume)
    parser.add_argument("--resume-from", default=cfg.resume_from)
    parser.add_argument("--reset-optimizer-on-resume", action="store_true")

    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--bootstrap-batches", type=int, default=cfg.bootstrap_batches)
    parser.add_argument("--bootstrap-max-tokens-per-part", type=int, default=cfg.bootstrap_max_tokens_per_part)
    parser.add_argument("--bootstrap-kmeans-iters", type=int, default=cfg.bootstrap_kmeans_iters)

    parser.add_argument("--allow-incomplete-boxes", action="store_true")
    parser.add_argument("--eval-every", type=int, default=cfg.eval_every)
    parser.add_argument("--save-every", type=int, default=cfg.save_every)
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    global cfg
    values = vars(args).copy()
    parts = tuple(part.strip().lower() for part in values.pop("parts").split(",") if part.strip())
    if not parts:
        raise ValueError("--parts must contain at least one part.")
    values["parts"] = parts
    values["bootstrap_memory"] = not bool(values.pop("skip_bootstrap"))

    for key, value in values.items():
        setattr(cfg, key, value)

    cfg.dogs_root = os.path.abspath(os.path.expanduser(cfg.dogs_root))
    cfg.train_aug_root = os.path.abspath(os.path.expanduser(cfg.train_aug_root))
    cfg.gdino_box_dir = os.path.abspath(os.path.expanduser(cfg.gdino_box_dir))
    cfg.save_dir = os.path.abspath(os.path.expanduser(cfg.save_dir))


def preflight() -> None:
    required = [
        os.path.join(cfg.dogs_root, "Images"),
        os.path.join(cfg.dogs_root, "Annotation"),
        os.path.join(cfg.dogs_root, "test_list.mat"),
        os.path.join(cfg.train_aug_root, cfg.train_aug_manifest_file),
        os.path.join(cfg.gdino_box_dir, cfg.gdino_test_file),
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing required paths:\n  - " + "\n  - ".join(missing))


def main() -> None:
    args = parse_args()
    apply_args(args)
    setup_distributed()
    preflight()
    set_seed(cfg.seed)

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if is_main_process():
        ensure_dir(cfg.save_dir)
    barrier()

    rank0_print("[Run] Stanford Dogs part-gated shared prototype model with offline TesNet augmentation")
    rank0_print(
        f"[Run] DINO={cfg.dino_model}; classes={cfg.num_classes}; parts={list(cfg.parts)}; "
        f"K={cfg.k_per_part}; train=offline crop-resize-{cfg.image_size} augmentation; "
        f"test=official dog bbox crop -> Resize({cfg.image_size})"
    )
    rank0_print(
        f"[Run] world_size={WORLD_SIZE}; per_gpu_batch={cfg.batch_size}; "
        f"global_batch={cfg.batch_size * WORLD_SIZE}; epochs={cfg.epochs}"
    )

    train_set = StanfordDogsTesNetAugmented(
        cfg.train_aug_root,
        cfg.train_aug_manifest_file,
        only_original=False,
    )
    bootstrap_set = StanfordDogsTesNetAugmented(
        cfg.train_aug_root,
        cfg.train_aug_manifest_file,
        only_original=True,
    )
    test_set = StanfordDogsWithPartBoxes(
        "test", os.path.join(cfg.gdino_box_dir, cfg.gdino_test_file)
    )
    train_loader = make_loader(train_set, train=True)
    test_loader = make_loader(test_set, train=False)
    rank0_print(
        f"[Data] train_augmented={len(train_set):,} "
        f"bootstrap_original={len(bootstrap_set):,} test={len(test_set):,}"
    )

    backbone = load_dinov2().to(DEVICE)
    candidate_params = set_backbone_trainability(backbone)
    with torch.no_grad():
        dummy = torch.zeros((1, 3, cfg.image_size, cfg.image_size), device=DEVICE)
        tokens, grid_h, grid_w = extract_patch_tokens(backbone, dummy)
    dim = int(tokens.shape[-1])
    rank0_print(
        f"[Backbone] tokens={tokens.shape[1]} grid={grid_h}x{grid_w} dim={dim}; "
        f"eligible_last_blocks={cfg.unfreeze_last_blocks}; trainable_params={candidate_params:,}"
    )

    core_model = SharedPartPrototypeModel(
        backbone=backbone,
        dim=dim,
        parts=len(cfg.parts),
        k=cfg.k_per_part,
        classes=cfg.num_classes,
    ).to(DEVICE)

    # Bootstrap before DDP. Rank 0 uses all train images, then broadcasts M.
    if cfg.bootstrap_memory:
        if is_distributed():
            if is_main_process():
                bootstrap_loader = DataLoader(
                    bootstrap_set,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                    pin_memory=(DEVICE.type == "cuda"),
                    persistent_workers=(cfg.num_workers > 0),
                    drop_last=False,
                )
                bootstrap_memory(core_model, bootstrap_loader)
            dist.broadcast(core_model.memory, src=0)
            barrier()
        else:
            bootstrap_loader = make_loader(bootstrap_set, train=True)
            bootstrap_memory(core_model, bootstrap_loader)

    if is_distributed():
        model: nn.Module = DDP(
            core_model,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK,
            broadcast_buffers=True,
            # First freeze epochs deliberately detach the backbone.
            find_unused_parameters=True,
        )
    else:
        model = core_model

    optimizer = build_optimizer(core_model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.epochs),
        eta_min=0.0,
    )
    autocast_enabled = cfg.amp and DEVICE.type == "cuda"
    scaler = build_scaler(autocast_enabled)

    start_epoch = 1
    best_acc = -1.0
    resume_path = resolve_resume_path()
    if resume_path is not None:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Requested resume checkpoint not found: {resume_path}")
        checkpoint = safe_torch_load(resume_path, map_location="cpu")
        core_model.load_state_dict(checkpoint["model"], strict=True)
        best_acc = float(checkpoint.get("best_acc", -1.0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if not cfg.reset_optimizer_on_resume:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint.get("scheduler") is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if checkpoint.get("scaler") is not None and scaler.is_enabled():
                scaler.load_state_dict(checkpoint["scaler"])
        rank0_print(f"[Resume] {resume_path}; next_epoch={start_epoch}; best={best_acc:.4f}")

    if start_epoch > cfg.epochs:
        rank0_print("[Done] requested epoch count already completed.")
        cleanup_distributed()
        return

    for epoch in range(start_epoch, cfg.epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        detach_backbone = epoch <= cfg.freeze_backbone_epochs
        model.train()
        if detach_backbone:
            core_model.backbone.eval()

        accumulated: Dict[str, float] = {}
        seen, correct = 0, 0
        started = time.time()
        iterator = tqdm(
            train_loader,
            desc=f"Train {epoch}/{cfg.epochs}",
            dynamic_ncols=True,
            disable=not is_main_process(),
        )

        for batch_index, (images, boxes, labels, _) in enumerate(iterator):
            if cfg.max_train_batches > 0 and batch_index >= cfg.max_train_batches:
                break

            images = images.to(DEVICE, non_blocking=True)
            boxes = boxes.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=autocast_enabled):
                output = model(images, detach_backbone=detach_backbone)
                q_sem, valid_bp = boxes_to_soft_targets(
                    boxes,
                    int(output["grid_h"]),
                    int(output["grid_w"]),
                    cfg.image_size,
                )
                loss, stats = compute_losses(output, labels, q_sem, valid_bp, epoch)

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if (
                epoch >= cfg.ema_start_epoch
                and (cfg.ema_stop_epoch <= 0 or epoch <= cfg.ema_stop_epoch)
                and (batch_index + 1) % max(1, cfg.ema_every_steps) == 0
            ):
                core_model.ema_update_memory(
                    output["tokens"].detach(),
                    output["part_map"].detach(),
                    output["proto_assign"].detach(),
                    q_sem.detach(),
                )

            count = int(labels.numel())
            seen += count
            correct += int((output["logits"].detach().argmax(dim=1) == labels).sum().item())
            for key, value in stats.items():
                accumulated[key] = accumulated.get(key, 0.0) + value * count

            iterator.set_postfix(
                loss=f"{accumulated.get('loss', 0.0) / max(1, seen):.4f}",
                acc=f"{correct / max(1, seen):.4f}",
                vis=f"{accumulated.get('visibility', 0.0) / max(1, seen):.3f}",
            )

        scheduler.step()

        if is_distributed():
            stat_keys = sorted(accumulated.keys())
            packed = torch.tensor(
                [float(seen), float(correct)] + [float(accumulated[key]) for key in stat_keys],
                device=DEVICE,
                dtype=torch.float64,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            seen = int(packed[0].item())
            correct = int(packed[1].item())
            accumulated = {key: float(packed[2 + index].item()) for index, key in enumerate(stat_keys)}

        train_stats = {key: value / max(1, seen) for key, value in accumulated.items()}
        train_acc = correct / max(1, seen)
        test_acc: Optional[float] = None
        if epoch % max(1, cfg.eval_every) == 0 or epoch == cfg.epochs:
            test_acc = evaluate(model, test_loader)

        elapsed = time.time() - started
        rank0_print(
            f"[Epoch {epoch:03d}] train_loss={train_stats.get('loss', 0.0):.4f} "
            f"train_acc={train_acc:.4f} "
            f"test_acc={'-' if test_acc is None else f'{test_acc:.4f}'} "
            f"route={train_stats.get('route', 0.0):.4f} "
            f"agree={train_stats.get('proto_agree', 0.0):.4f} "
            f"time={elapsed:.1f}s"
        )

        record: Dict[str, Any] = {
            "epoch": epoch,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "elapsed_sec": elapsed,
            "backbone_detached": detach_backbone,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_router": optimizer.param_groups[1]["lr"],
            "lr_proto": optimizer.param_groups[2]["lr"],
            "lr_classifier": optimizer.param_groups[3]["lr"],
            **train_stats,
        }
        if is_main_process():
            append_metrics(record)

        if test_acc is not None and test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(
                os.path.join(cfg.save_dir, "best.pth"),
                epoch,
                best_acc,
                model,
                optimizer,
                scheduler,
                scaler,
            )
            rank0_print(f"[Best] epoch={epoch} acc={best_acc:.4f}")

        if epoch % max(1, cfg.save_every) == 0 or epoch == cfg.epochs:
            save_checkpoint(
                os.path.join(cfg.save_dir, "last.pth"),
                epoch,
                best_acc,
                model,
                optimizer,
                scheduler,
                scaler,
            )

    rank0_print(f"[Done] best_test_acc={best_acc:.4f}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
