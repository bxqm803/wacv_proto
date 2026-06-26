#!/usr/bin/env python3
"""
Evaluate Consistency (Con.) and Stability (Stb.) for the shared part-prototype
CUB model using its local resp evidence map and CUB's official keypoint labels.

Protocol:
  resp[b,p,n,k] = responsibility[b,p,n,k] * relu(sim[b,p,n,k])
  proto_score_raw[b,p,k] = sum_n resp[b,p,n,k]

A shared prototype (p,k) is assigned to its owner class
argmax_c class_weight[c,p,k], matching the class-conditioned evaluation domain
used by class-specific ProtoPNet metrics. Its spatial map remains raw resp:
the owner weight is used only to select the evaluation images and is NOT
multiplied into the map.

Con:
  A prototype is consistent if any official CUB keypoint is covered by its
  72x72 peak region (half_size=36) on at least 80% of owner-class test images
  where that keypoint is visible.

Stb:
  On each owner-class image, compare the 15D keypoint-coverage vectors from
  the original image and a normalized-space Gaussian-noise perturbation.
  The prototype score is the fraction of identical vectors; Stb averages it
  over all 300 prototypes.

Expected training source:
  train_cub_shared_part_proto_finetune_reg_vitb16_ddp.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def safe_load(path: str, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def read_text_map(path: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if row:
                k, v = row.split(maxsplit=1)
                out[int(k)] = v
    return out


def read_int_map(path: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if row:
                k, v = row.split()
                out[int(k)] = int(v)
    return out


def read_bboxes(path: str) -> Dict[int, Tuple[float, float, float, float]]:
    out: Dict[int, Tuple[float, float, float, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in f:
            vals = row.strip().split()
            if vals:
                out[int(vals[0])] = tuple(float(v) for v in vals[1:5])
    return out


def crop_xyxy(image_size: Tuple[int, int], bbox_xywh: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox_xywh
    width, height = image_size
    x1 = max(0, min(width - 1, int(math.floor(x))))
    y1 = max(0, min(height - 1, int(math.floor(y))))
    x2 = max(x1 + 1, min(width, int(math.ceil(x + w))))
    y2 = max(y1 + 1, min(height, int(math.ceil(y + h))))
    return x1, y1, x2, y2


def normalize_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    clean: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        key = str(key)
        if key.startswith("module."):
            key = key[len("module."):]
        clean[key] = value
    return clean


def import_training_module(path: str):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("shared_part_train", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import training script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["shared_part_train"] = module
    spec.loader.exec_module(module)
    return module


class CUBOfficialKeypointTest(Dataset):
    """Official CUB test split, official bird bbox crop, Resize(image_size)."""

    def __init__(
        self,
        cub_root: str,
        image_size: int,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        self.cub_root = os.path.abspath(cub_root)
        self.image_size = int(image_size)

        images = read_text_map(os.path.join(self.cub_root, "images.txt"))
        labels = read_int_map(os.path.join(self.cub_root, "image_class_labels.txt"))
        split = read_int_map(os.path.join(self.cub_root, "train_test_split.txt"))
        boxes = read_bboxes(os.path.join(self.cub_root, "bounding_boxes.txt"))

        self.part_names = read_text_map(os.path.join(self.cub_root, "parts", "parts.txt"))
        self.num_parts = len(self.part_names)

        locs: Dict[int, List[Tuple[int, float, float, int]]] = defaultdict(list)
        part_locs_path = os.path.join(self.cub_root, "parts", "part_locs.txt")
        with open(part_locs_path, "r", encoding="utf-8") as f:
            for row in f:
                vals = row.strip().split()
                if not vals:
                    continue
                image_id = int(vals[0])
                part_id = int(vals[1])
                x, y = float(vals[2]), float(vals[3])
                visible = int(vals[4])
                locs[image_id].append((part_id, x, y, visible))

        self.samples: List[Dict[str, Any]] = []
        for image_id in sorted(images):
            if split[image_id] != 0:
                continue
            self.samples.append(
                {
                    "image_id": image_id,
                    "relpath": images[image_id],
                    "label": labels[image_id] - 1,
                    "bbox": boxes[image_id],
                    "locs": locs.get(image_id, []),
                }
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = os.path.join(self.cub_root, "images", sample["relpath"])

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            x1, y1, x2, y2 = crop_xyxy(img.size, sample["bbox"])
            cropped = img.crop((x1, y1, x2, y2))
            tensor = self.transform(cropped)

        keypoints = torch.full((self.num_parts, 2), -1.0, dtype=torch.float32)
        visible = torch.zeros((self.num_parts,), dtype=torch.bool)
        crop_w = float(x2 - x1)
        crop_h = float(y2 - y1)

        for part_id, px, py, is_visible in sample["locs"]:
            j = int(part_id) - 1
            if not is_visible:
                continue
            # Match the exact official-bbox crop used by this pipeline.
            rx = (px - float(x1)) / crop_w * self.image_size
            ry = (py - float(y1)) / crop_h * self.image_size
            if 0.0 <= rx < self.image_size and 0.0 <= ry < self.image_size:
                keypoints[j] = torch.tensor([rx, ry], dtype=torch.float32)
                visible[j] = True

        return (
            tensor,
            torch.tensor(int(sample["label"]), dtype=torch.long),
            keypoints,
            visible,
            torch.tensor(int(sample["image_id"]), dtype=torch.long),
        )


@torch.inference_mode()
def response_coverage(
    model: torch.nn.Module,
    images: torch.Tensor,
    keypoints: torch.Tensor,
    image_size: int,
    half_size: int,
    amp: bool,
) -> torch.Tensor:
    """
    Returns bool [B, P*K, Q].
    Each map is the local resp summand, upsampled to image_size then localized
    by its maximum, following the official EvalProtoPNet metric convention.
    """
    device_type = images.device.type
    autocast_enabled = bool(amp and device_type == "cuda")

    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=autocast_enabled):
        out = model(images)

    sim = out.get("sim", out.get("similarity"))
    if sim is None:
        raise KeyError("Model output lacks both 'sim' and 'similarity'.")
    responsibility = out["responsibility"]

    # The exact spatial summand used by resp_sum:
    # proto_score_raw = sum_n responsibility * relu(sim)
    resp = responsibility.float() * F.relu(sim.float())  # [B,P,N,K]

    bsz, n_parts, n_tokens, k_per_part = resp.shape
    grid_h = int(out["grid_h"].item())
    grid_w = int(out["grid_w"].item())
    if n_tokens != grid_h * grid_w:
        raise RuntimeError(
            f"Token/grid mismatch: N={n_tokens}, grid={grid_h}x{grid_w}"
        )

    maps = (
        resp.permute(0, 1, 3, 2)
        .reshape(bsz * n_parts * k_per_part, 1, grid_h, grid_w)
        .contiguous()
    )

    # B=4 gives 1,200 maps for P=6,K=50: safe on a 40GB A100.
    upsampled = F.interpolate(
        maps,
        size=(image_size, image_size),
        mode="bicubic",
        align_corners=False,
    )
    flat_idx = upsampled.flatten(1).argmax(dim=1)

    peak_y = (flat_idx // image_size).view(bsz, n_parts * k_per_part)
    peak_x = (flat_idx % image_size).view(bsz, n_parts * k_per_part)

    x = keypoints[..., 0].unsqueeze(1)  # [B,1,Q]
    y = keypoints[..., 1].unsqueeze(1)

    x1 = (peak_x - int(half_size)).unsqueeze(-1)
    x2 = (peak_x + int(half_size)).clamp_max(image_size).unsqueeze(-1)
    y1 = (peak_y - int(half_size)).unsqueeze(-1)
    y2 = (peak_y + int(half_size)).clamp_max(image_size).unsqueeze(-1)

    # Keypoints are invalid (-1,-1) when invisible; visibility masking is
    # applied by the caller for Con and owner-class conditioning.
    return (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)


def load_model(train_module, checkpoint: Dict[str, Any], device: torch.device):
    saved_cfg = checkpoint.get("cfg", checkpoint.get("config", {}))
    if not isinstance(saved_cfg, dict):
        raise TypeError("Checkpoint must contain a dict under 'cfg' or 'config'.")

    # Reconstruct the original model configuration.
    for key, value in saved_cfg.items():
        if hasattr(train_module.cfg, key):
            setattr(train_module.cfg, key, value)

    # Use single-process evaluator device.
    train_module.DEVICE = device

    if not hasattr(train_module, "load_visual_backbone"):
        raise AttributeError("Training script lacks load_visual_backbone().")
    if not hasattr(train_module, "SharedPartPrototypeDINO"):
        raise AttributeError(
            "This evaluator expects SharedPartPrototypeDINO in the training script."
        )

    backbone = train_module.load_visual_backbone(train_module.cfg.dino_model).to(device)
    backbone.eval()

    with torch.no_grad():
        dummy = torch.zeros(
            (1, 3, int(train_module.cfg.image_size), int(train_module.cfg.image_size)),
            device=device,
        )
        tokens, _, _ = train_module.extract_patch_tokens(backbone, dummy)

    model = train_module.SharedPartPrototypeDINO(
        backbone=backbone,
        dim=int(tokens.shape[-1]),
        parts=len(train_module.cfg.parts),
        k=int(train_module.cfg.k_per_part),
        classes=int(train_module.cfg.num_classes),
    ).to(device)

    raw_state = checkpoint.get("model", checkpoint)
    if not isinstance(raw_state, dict):
        raise TypeError("Checkpoint model state is not a dict.")
    missing, unexpected = model.load_state_dict(normalize_state_dict(raw_state), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch.\n"
            f"Missing ({len(missing)}): {missing[:12]}\n"
            f"Unexpected ({len(unexpected)}): {unexpected[:12]}"
        )

    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "CUB Consistency/Stability for shared prototypes using resp maps"
    )
    p.add_argument(
        "--train-script",
        default="./train_cub_shared_part_proto_finetune_reg_vitb16_ddp.py",
        help="The exact training script used to create the checkpoint.",
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cub-root", default="./data/CUB_200_2011")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--half-size", type=int, default=36)
    p.add_argument("--part-thresh", type=float, default=0.80)
    p.add_argument("--noise-std", type=float, default=0.20)
    p.add_argument("--noise-clip", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_module = import_training_module(args.train_script)
    checkpoint = safe_load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)}")

    model = load_model(train_module, checkpoint, device)
    cfg = train_module.cfg
    image_size = int(cfg.image_size)
    parts = tuple(str(v) for v in cfg.parts)
    p_count = len(parts)
    k_count = int(cfg.k_per_part)
    m_count = p_count * k_count

    mean, std = train_module.input_normalization()
    dataset = CUBOfficialKeypointTest(
        cub_root=args.cub_root,
        image_size=image_size,
        mean=mean,
        std=std,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
    )

    if dataset.num_parts != 15:
        raise RuntimeError(f"Expected 15 official CUB keypoints, got {dataset.num_parts}.")
    if m_count != 300:
        print(f"[Warning] evaluating {m_count} prototypes, not 300.")

    # Shared-prototype adaptation:
    # assign prototype (p,k) to the class where its nonnegative readout weight
    # is maximal; this supplies the class-specific image set used by the
    # original Con/Stb protocol.
    owner_class = model.class_weights().detach().float().argmax(dim=0).reshape(-1).to(device)

    class_names = read_text_map(os.path.join(args.cub_root, "classes.txt"))
    part_names = [dataset.part_names[i + 1] for i in range(dataset.num_parts)]

    hit_count = torch.zeros((m_count, dataset.num_parts), dtype=torch.long)
    visible_count = torch.zeros((m_count, dataset.num_parts), dtype=torch.long)
    stable_count = torch.zeros((m_count,), dtype=torch.long)
    eval_count = torch.zeros((m_count,), dtype=torch.long)

    print(
        f"[Eval] test_images={len(dataset)} | image_size={image_size} | "
        f"grid from model output | P={p_count}, K={k_count} | "
        f"half_size={args.half_size}, part_thresh={args.part_thresh}"
    )
    print(
        "[Eval] spatial map = responsibility * ReLU(sim); "
        "class weight only selects the owner class."
    )

    progress = tqdm(loader, desc="Con/Stb (resp)", dynamic_ncols=True)
    for images, labels, keypoints, visible, _image_ids in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        keypoints = keypoints.to(device, non_blocking=True)
        visible = visible.to(device, non_blocking=True)

        coverage = response_coverage(
            model=model,
            images=images,
            keypoints=keypoints,
            image_size=image_size,
            half_size=args.half_size,
            amp=args.amp,
        )

        # Noise is applied in normalized tensor space exactly as in EvalProtoPNet.
        noise = torch.randn_like(images) * float(args.noise_std)
        noise = noise.clamp(min=-float(args.noise_clip), max=float(args.noise_clip))
        coverage_noise = response_coverage(
            model=model,
            images=images + noise,
            keypoints=keypoints,
            image_size=image_size,
            half_size=args.half_size,
            amp=args.amp,
        )

        # [B,M] selects images from each shared prototype's owner class.
        owner_mask = labels[:, None].eq(owner_class[None, :])
        valid = owner_mask[:, :, None] & visible[:, None, :]

        visible_count += valid.sum(dim=0).cpu().long()
        hit_count += (coverage & valid).sum(dim=0).cpu().long()

        same_coverage = coverage.eq(coverage_noise).all(dim=-1)  # [B,M]
        stable_count += (same_coverage & owner_mask).sum(dim=0).cpu().long()
        eval_count += owner_mask.sum(dim=0).cpu().long()

    denom = visible_count.clamp_min(1).float()
    per_part_ratio = hit_count.float() / denom
    dominant_ratio, dominant_part = per_part_ratio.max(dim=1)
    consistent = dominant_ratio.ge(float(args.part_thresh))
    per_proto_stability = stable_count.float() / eval_count.clamp_min(1).float()

    con = float(consistent.float().mean().item() * 100.0)
    stb = float(per_proto_stability.mean().item() * 100.0)

    print(f"\nConsistency (Con.): {con:.2f}%")
    print(f"Stability   (Stb.): {stb:.2f}%")

    owner_cpu = owner_class.cpu().tolist()
    csv_path = output_dir / "per_prototype_resp_con_stb.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "prototype_id",
                "part_group",
                "within_part_k",
                "owner_class_id_1based",
                "owner_class_name",
                "dominant_gt_keypoint",
                "consistency_ratio",
                "consistent_at_threshold",
                "stability",
                "owner_test_images",
            ]
            + [f"ratio_{name}" for name in part_names]
        )

        for flat_id in range(m_count):
            p = flat_id // k_count
            k = flat_id % k_count
            owner = int(owner_cpu[flat_id])
            row = [
                flat_id,
                parts[p],
                k,
                owner + 1,
                class_names.get(owner + 1, str(owner + 1)),
                part_names[int(dominant_part[flat_id].item())],
                float(dominant_ratio[flat_id].item()),
                int(consistent[flat_id].item()),
                float(per_proto_stability[flat_id].item()),
                int(eval_count[flat_id].item()),
            ]
            row += [float(v) for v in per_part_ratio[flat_id].tolist()]
            writer.writerow(row)

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "train_script": os.path.abspath(args.train_script),
        "resp_definition": "responsibility * ReLU(sim)",
        "shared_prototype_domain": "owner class = argmax_c class_weight[c,p,k]",
        "num_prototypes": m_count,
        "num_test_images": len(dataset),
        "image_size": image_size,
        "half_size": int(args.half_size),
        "part_threshold": float(args.part_thresh),
        "noise_std_normalized": float(args.noise_std),
        "noise_clip_normalized": float(args.noise_clip),
        "seed": int(args.seed),
        "consistency_percent": con,
        "stability_percent": stb,
        "per_prototype_csv": str(csv_path),
    }
    with open(output_dir / "summary_resp_con_stb.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Saved] {csv_path}")
    print(f"[Saved] {output_dir / 'summary_resp_con_stb.json'}")


if __name__ == "__main__":
    main()
