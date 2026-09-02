#!/usr/bin/env python3
"""
No-GroundingDINO-semantic-cue wrapper for:
    train_cub_shared_part_proto_finetune_reg_vitb16_ddp.py

Put this file in the same cub_server_repo directory as the original training
script, then run it with the same CLI arguments.

What is removed:
  1) GDINO boxes are NOT loaded for original CUB train/test images.
  2) Offline-augmentation cache is read only for relpaths + class labels;
     its part boxes are discarded.
  3) q_sem is identically zero and valid_bp is identically False.
     Therefore route loss, visibility loss, and bbox-masked agreement receive
     no semantic supervision.
  4) Semantic EMA injection receives zero semantic mass, so the original EMA
     code automatically falls back to the self-routed target.
  5) Semantic bootstrap is disabled.

What is retained:
  - part queries and named banks
  - self-routed responsibility
  - EMA memory (ema_rho unchanged)
  - ema_sem_mix hyperparameter unchanged (but never activated because
    semantic mass is zero)
  - bounded residual
  - classification objective and all other architecture/training settings

This is intended for the joint "w/o GroundingDINO semantic cue" ablation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import train_cub_shared_part_proto_finetune_reg_vitb16_ddp as base


# -----------------------------------------------------------------------------
# 1) Data: preserve images/labels, discard all semantic part boxes.
# -----------------------------------------------------------------------------

def _invalid_boxes(n: int) -> torch.Tensor:
    return torch.full(
        (int(n), len(base.cfg.parts), 1, 4),
        -1.0,
        dtype=torch.float32,
    )


class CUBImageNoSemanticCue(Dataset):
    """Original CUB bbox-cropped images with dummy invalid part boxes."""

    def __init__(self, split: str, _gdino_path_ignored: str):
        self.samples = base.build_cub_samples(base.cfg.cub_root, split)
        self.boxes = _invalid_boxes(len(self.samples))

        mean, std = base.input_normalization()
        self.transform = base.transforms.Compose(
            [
                base.transforms.Resize(
                    (base.cfg.image_size, base.cfg.image_size),
                    interpolation=base.transforms.InterpolationMode.BICUBIC,
                ),
                base.transforms.ToTensor(),
                base.transforms.Normalize(mean, std),
            ]
        )

        base.rank0_print(
            f"[NoGDINO] {split}: {len(self.samples)} images; "
            "all semantic part boxes replaced by invalid dummy boxes."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        path = os.path.join(
            base.cfg.cub_root, "images", str(sample["relpath"])
        )
        with base.Image.open(path) as image:
            image = image.convert("RGB")
        image = base.crop_bbox(image, sample["bbox"])
        return (
            self.transform(image),
            self.boxes[idx],
            int(sample["label"]),
            str(sample["relpath"]),
        )


class OfflineAugmentedCUBNoSemanticCue(Dataset):
    """
    Offline NPPL images.

    The existing cache is used ONLY as an image manifest/class-label file.
    Any stored synchronized GDINO part boxes are ignored.
    """

    def __init__(self, aug_root: str, box_file: str):
        self.aug_root = os.path.abspath(aug_root)
        self.image_root = os.path.join(
            self.aug_root, base.cfg.offline_aug_image_dir
        )
        self.meta_path = os.path.join(self.aug_root, box_file)

        if not os.path.isdir(self.image_root):
            raise FileNotFoundError(
                f"Offline augmented image directory not found: {self.image_root}"
            )
        if not os.path.isfile(self.meta_path):
            raise FileNotFoundError(
                "Offline augmentation metadata file not found: "
                f"{self.meta_path}"
            )

        raw = base.safe_torch_load(self.meta_path, map_location="cpu")
        if not isinstance(raw, dict):
            raise TypeError(
                f"Offline augmentation metadata must be a dict: {self.meta_path}"
            )

        # IMPORTANT: only these two fields are consumed.
        relpaths = [str(v) for v in raw.get("relpaths", [])]
        labels_raw = raw.get("labels")
        if labels_raw is None:
            raise KeyError(
                f"Offline augmentation metadata is missing 'labels': {self.meta_path}"
            )
        labels = torch.as_tensor(labels_raw, dtype=torch.long).view(-1).tolist()

        if not relpaths:
            raise RuntimeError(
                f"Offline augmentation metadata has no relpaths: {self.meta_path}"
            )
        if len(relpaths) != len(labels):
            raise ValueError(
                f"Offline relpaths/labels mismatch: {len(relpaths)} vs {len(labels)}"
            )

        indices = list(range(len(relpaths)))
        if 0 < int(base.cfg.offline_aug_limit) < len(indices):
            rng = base.np.random.default_rng(int(base.cfg.seed))
            indices = sorted(
                rng.choice(
                    len(indices),
                    size=int(base.cfg.offline_aug_limit),
                    replace=False,
                ).tolist()
            )

        self.samples = [
            {"relpath": relpaths[i], "label": int(labels[i])}
            for i in indices
        ]
        if any(
            s["label"] < 0 or s["label"] >= base.cfg.num_classes
            for s in self.samples
        ):
            raise ValueError(
                "Offline augmentation metadata contains a class label "
                "outside [0, num_classes)."
            )

        # Discard every stored part box.
        self.boxes = _invalid_boxes(len(self.samples))
        self.supervised_images = 0
        self.unsupervised_images = len(self.samples)

        mean, std = base.input_normalization()
        self.transform = base.transforms.Compose(
            [
                base.transforms.Resize(
                    (base.cfg.image_size, base.cfg.image_size),
                    interpolation=base.transforms.InterpolationMode.BICUBIC,
                ),
                base.transforms.ToTensor(),
                base.transforms.Normalize(mean, std),
            ]
        )

        base.rank0_print(
            f"[NoGDINO][OfflineAug] images={len(self.samples)}; "
            "using relpaths/labels only; all stored semantic boxes ignored."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        path = os.path.join(self.image_root, str(sample["relpath"]))
        with base.Image.open(path) as image:
            image = image.convert("RGB")
        return (
            self.transform(image),
            self.boxes[idx],
            int(sample["label"]),
            f"aug::{sample['relpath']}",
        )


# -----------------------------------------------------------------------------
# 2) Semantic targets: force them to zero for every image/part.
# -----------------------------------------------------------------------------

def no_gdino_semantic_targets(
    boxes: torch.Tensor,
    grid_h: int,
    grid_w: int,
    image_size: int,
):
    del image_size
    batch = int(boxes.shape[0])
    parts = len(base.cfg.parts)
    tokens = int(grid_h) * int(grid_w)
    device = boxes.device

    q_sem = torch.zeros(
        (batch, parts, tokens),
        device=device,
        dtype=torch.float32,
    )
    valid_bp = torch.zeros(
        (batch, parts),
        device=device,
        dtype=torch.bool,
    )
    return q_sem, valid_bp


# -----------------------------------------------------------------------------
# 3) Preflight: no original GDINO cache is required.
#    Offline augmentation metadata is still required in append/only mode,
#    but its box fields are never consumed.
# -----------------------------------------------------------------------------

def no_gdino_preflight() -> None:
    required = [
        os.path.join(base.cfg.cub_root, "images.txt"),
        os.path.join(base.cfg.cub_root, "image_class_labels.txt"),
        os.path.join(base.cfg.cub_root, "train_test_split.txt"),
        os.path.join(base.cfg.cub_root, "bounding_boxes.txt"),
    ]

    if base.cfg.offline_aug_mode != "none":
        required.extend(
            [
                os.path.join(
                    base.cfg.offline_aug_root,
                    base.cfg.offline_aug_image_dir,
                ),
                os.path.join(
                    base.cfg.offline_aug_root,
                    base.cfg.offline_aug_box_file,
                ),
            ]
        )

    if base.cfg.backbone == "clip" and base.cfg.clip_init_checkpoint:
        required.append(base.cfg.clip_init_checkpoint)

    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required paths:\n" + "\n".join(missing)
        )


# -----------------------------------------------------------------------------
# 4) Make checkpoint config reflect that semantic bootstrap is absent.
#    Other hyperparameters (including ema_sem_mix, lambda_route, lambda_vis)
#    remain exactly as supplied. With q_sem=0 and valid_bp=False, the
#    GDINO-dependent terms evaluate to zero.
# -----------------------------------------------------------------------------

_original_apply_args = base.apply_args


def no_gdino_apply_args(args) -> None:
    _original_apply_args(args)
    base.cfg.bootstrap_memory = False
    base.rank0_print(
        "[NoGDINO] GroundingDINO semantic cue DISABLED. "
        "Semantic bootstrap disabled; route/visibility semantic targets are "
        "empty; EMA retains self-routed updates only."
    )


def no_gdino_bootstrap(*_args: Any, **_kwargs: Any) -> None:
    base.rank0_print(
        "[NoGDINO] semantic prototype bootstrap skipped."
    )


# Install patches before entering the original training main().
base.CUBImageWithPartBoxes = CUBImageNoSemanticCue
base.OfflineAugmentedCUBWithPartBoxes = OfflineAugmentedCUBNoSemanticCue
base.boxes_to_soft_targets = no_gdino_semantic_targets
base.preflight = no_gdino_preflight
base.apply_args = no_gdino_apply_args
base.bootstrap_memory = no_gdino_bootstrap


if __name__ == "__main__":
    base.main()
