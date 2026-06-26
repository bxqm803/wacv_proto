#!/usr/bin/env python3
"""
Activation-conditioned Consistency (Con.) and Stability (Stb.) for the shared
part-prototype CUB model using local resp evidence maps and CUB official
keypoint labels.

For each image i and shared prototype j=(p,k), define the unscaled resp_sum
evidence
    e_{i,j} = sum_n responsibility_{i,p,n,k} * ReLU(sim_{i,p,n,k}).

Only image-prototype pairs with e_{i,j} > --resp-threshold are evaluated.
This avoids localizing arbitrary peaks for prototypes that are not used by an
image. No owner-class assignment is imposed because prototypes are shared.

Con:
  For each prototype, among its activated test images, a CUB keypoint is
  consistent when its peak-centered 72x72 region is covered on at least 80%
  of the activated images where that keypoint is visible.

Stb:
  For the same activation-selected image-prototype pairs, compare original and
  Gaussian-noise perturbed 15D keypoint-coverage vectors.

This is an activation-conditioned adaptation of EvalProtoPNet's Con/Stb
protocol, not the unfiltered class-conditioned protocol.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
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
def response_outputs(
    model: torch.nn.Module,
    images: torch.Tensor,
    keypoints: torch.Tensor,
    image_size: int,
    half_size: int,
    amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      coverage: bool [B, P*K, Q]
      resp_score: float [B, P*K], where score=sum_n responsibility*ReLU(sim).

    The spatial map is the same local summand used by resp_sum.
    """
    device_type = images.device.type
    autocast_enabled = bool(amp and device_type == "cuda")

    with torch.autocast(
        device_type=device_type,
        dtype=torch.float16,
        enabled=autocast_enabled,
    ):
        out = model(images)

    sim = out.get("sim", out.get("similarity"))
    if sim is None:
        raise KeyError("Model output lacks both 'sim' and 'similarity'.")
    responsibility = out["responsibility"]

    # Exact local resp summand used by score_mode=resp_sum.
    resp = responsibility.float() * F.relu(sim.float())  # [B,P,N,K]
    bsz, n_parts, n_tokens, k_per_part = resp.shape
    m_count = n_parts * k_per_part

    # The unscaled aggregate response used to decide whether a prototype was
    # activated on this image.
    resp_score = resp.sum(dim=2).reshape(bsz, m_count)  # [B,P*K]

    grid_h = int(out["grid_h"].item())
    grid_w = int(out["grid_w"].item())
    if n_tokens != grid_h * grid_w:
        raise RuntimeError(
            f"Token/grid mismatch: N={n_tokens}, grid={grid_h}x{grid_w}"
        )

    maps = (
        resp.permute(0, 1, 3, 2)
        .reshape(bsz * m_count, 1, grid_h, grid_w)
        .contiguous()
    )
    upsampled = F.interpolate(
        maps,
        size=(image_size, image_size),
        mode="bicubic",
        align_corners=False,
    )
    flat_idx = upsampled.flatten(1).argmax(dim=1)

    peak_y = (flat_idx // image_size).view(bsz, m_count)
    peak_x = (flat_idx % image_size).view(bsz, m_count)

    x = keypoints[..., 0].unsqueeze(1)  # [B,1,Q]
    y = keypoints[..., 1].unsqueeze(1)

    x1 = (peak_x - int(half_size)).unsqueeze(-1)
    x2 = (peak_x + int(half_size)).clamp_max(image_size).unsqueeze(-1)
    y1 = (peak_y - int(half_size)).unsqueeze(-1)
    y2 = (peak_y + int(half_size)).clamp_max(image_size).unsqueeze(-1)

    coverage = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
    return coverage, resp_score


def load_model(train_module, checkpoint: Dict[str, Any], device: torch.device):
    saved_cfg = checkpoint.get("cfg", checkpoint.get("config", {}))
    if not isinstance(saved_cfg, dict):
        raise TypeError("Checkpoint must contain a dict under 'cfg' or 'config'.")

    # Reconstruct the exact training-time configuration saved in the checkpoint.
    for key, value in saved_cfg.items():
        if hasattr(train_module.cfg, key):
            setattr(train_module.cfg, key, value)

    # Single-process evaluator device.
    train_module.DEVICE = device

    if not hasattr(train_module, "load_visual_backbone"):
        raise AttributeError("Training script lacks load_visual_backbone().")
    if not hasattr(train_module, "extract_patch_tokens"):
        raise AttributeError("Training script lacks extract_patch_tokens().")

    # The current CUB script names the model SharedPartPrototypeModel;
    # older variants used SharedPartPrototypeDINO. Support both names.
    model_cls = getattr(train_module, "SharedPartPrototypeModel", None)
    if model_cls is None:
        model_cls = getattr(train_module, "SharedPartPrototypeDINO", None)
    if model_cls is None:
        raise AttributeError(
            "Could not find SharedPartPrototypeModel or SharedPartPrototypeDINO "
            "in the supplied training script."
        )

    # Newer scripts build the visual backbone from cfg with no positional
    # arguments; older scripts require cfg.dino_model.
    loader = train_module.load_visual_backbone
    signature = inspect.signature(loader)
    required_positional = [
        p for p in signature.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    ]
    if len(required_positional) == 0:
        backbone = loader()
    elif len(required_positional) == 1:
        dino_model = getattr(train_module.cfg, "dino_model", None)
        if dino_model is None:
            raise AttributeError(
                "load_visual_backbone() requires a model name, but cfg.dino_model "
                "is unavailable in this training script."
            )
        backbone = loader(dino_model)
    else:
        raise TypeError(
            "Unsupported load_visual_backbone() signature: "
            f"{signature}"
        )

    backbone = backbone.to(device).eval()

    with torch.no_grad():
        dummy = torch.zeros(
            (1, 3, int(train_module.cfg.image_size), int(train_module.cfg.image_size)),
            device=device,
        )
        tokens, _, _ = train_module.extract_patch_tokens(backbone, dummy)

    model = model_cls(
        backbone=backbone,
        dim=int(tokens.shape[-1]),
        parts=len(train_module.cfg.parts),
        k=int(train_module.cfg.k_per_part),
        classes=int(train_module.cfg.num_classes),
    ).to(device)

    raw_state = checkpoint.get("model", checkpoint)
    if not isinstance(raw_state, dict):
        raise TypeError("Checkpoint model state is not a dict.")

    missing, unexpected = model.load_state_dict(
        normalize_state_dict(raw_state),
        strict=False,
    )
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch after rebuilding the exact training architecture.\n"
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
    p.add_argument(
        "--resp-threshold",
        type=float,
        default=0.05,
        help="Evaluate a prototype only where its unscaled resp_sum score is above this value.",
    )
    p.add_argument(
        "--min-active-images",
        type=int,
        default=1,
        help="A prototype needs at least this many activation-selected test images to enter macro averages.",
    )
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

    part_names = [dataset.part_names[i + 1] for i in range(dataset.num_parts)]

    # All test images are considered. A prototype contributes only on image i
    # if its unscaled aggregated response e_{i,p,k} exceeds resp_threshold.
    hit_count = torch.zeros((m_count, dataset.num_parts), dtype=torch.long)
    visible_count = torch.zeros((m_count, dataset.num_parts), dtype=torch.long)
    stable_count = torch.zeros((m_count,), dtype=torch.long)
    active_count = torch.zeros((m_count,), dtype=torch.long)
    noise_active_count = torch.zeros((m_count,), dtype=torch.long)

    print(
        f"[Eval] test_images={len(dataset)} | image_size={image_size} | "
        f"P={p_count}, K={k_count} | half_size={args.half_size}, "
        f"part_thresh={args.part_thresh}"
    )
    print(
        f"[Eval] activation-conditioned selection: "
        f"sum_n(resp) > {args.resp_threshold:.6g}; "
        "spatial map = responsibility * ReLU(sim)."
    )

    progress = tqdm(loader, desc="Con/Stb (active resp)", dynamic_ncols=True)
    for images, _labels, keypoints, visible, _image_ids in progress:
        images = images.to(device, non_blocking=True)
        keypoints = keypoints.to(device, non_blocking=True)
        visible = visible.to(device, non_blocking=True)

        coverage, resp_score = response_outputs(
            model=model,
            images=images,
            keypoints=keypoints,
            image_size=image_size,
            half_size=args.half_size,
            amp=args.amp,
        )

        # Selection is defined from the original image response. If the same
        # prototype becomes inactive after perturbation, its changed peak map
        # remains part of Stb and can lower the stability score.
        active_mask = resp_score.gt(float(args.resp_threshold))  # [B,M]

        noise = torch.randn_like(images) * float(args.noise_std)
        noise = noise.clamp(min=-float(args.noise_clip), max=float(args.noise_clip))
        coverage_noise, resp_score_noise = response_outputs(
            model=model,
            images=images + noise,
            keypoints=keypoints,
            image_size=image_size,
            half_size=args.half_size,
            amp=args.amp,
        )
        noise_active_mask = resp_score_noise.gt(float(args.resp_threshold))

        valid = active_mask[:, :, None] & visible[:, None, :]
        visible_count += valid.sum(dim=0).cpu().long()
        hit_count += (coverage & valid).sum(dim=0).cpu().long()

        same_coverage = coverage.eq(coverage_noise).all(dim=-1)  # [B,M]
        stable_count += (same_coverage & active_mask).sum(dim=0).cpu().long()
        active_count += active_mask.sum(dim=0).cpu().long()
        noise_active_count += (noise_active_mask & active_mask).sum(dim=0).cpu().long()

    denom = visible_count.clamp_min(1).float()
    per_part_ratio = hit_count.float() / denom
    dominant_ratio, dominant_part = per_part_ratio.max(dim=1)
    consistent = dominant_ratio.ge(float(args.part_thresh))
    per_proto_stability = stable_count.float() / active_count.clamp_min(1).float()
    noise_retention = noise_active_count.float() / active_count.clamp_min(1).float()

    eligible = active_count.ge(int(args.min_active_images))
    n_eligible = int(eligible.sum().item())
    if n_eligible == 0:
        raise RuntimeError(
            "No prototype has enough active test images. Lower --resp-threshold "
            "or --min-active-images."
        )

    con = float(consistent[eligible].float().mean().item() * 100.0)
    stb = float(per_proto_stability[eligible].mean().item() * 100.0)
    mean_active = float(active_count[eligible].float().mean().item())
    median_active = float(active_count[eligible].float().median().item())

    print(f"\nActivation-conditioned Consistency (Con.): {con:.2f}%")
    print(f"Activation-conditioned Stability   (Stb.): {stb:.2f}%")
    print(
        f"Eligible prototypes: {n_eligible}/{m_count} | "
        f"active images/prototype: mean={mean_active:.2f}, "
        f"median={median_active:.1f}"
    )

    csv_path = output_dir / "per_prototype_active_resp_con_stb.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "prototype_id",
                "part_group",
                "within_part_k",
                "active_test_images",
                "eligible_for_macro",
                "noise_resp_retention",
                "dominant_gt_keypoint",
                "consistency_ratio",
                "consistent_at_threshold",
                "stability",
            ]
            + [f"ratio_{name}" for name in part_names]
        )

        for flat_id in range(m_count):
            p = flat_id // k_count
            k = flat_id % k_count
            row = [
                flat_id,
                parts[p],
                k,
                int(active_count[flat_id].item()),
                int(eligible[flat_id].item()),
                float(noise_retention[flat_id].item()),
                part_names[int(dominant_part[flat_id].item())],
                float(dominant_ratio[flat_id].item()),
                int(consistent[flat_id].item()),
                float(per_proto_stability[flat_id].item()),
            ]
            row += [float(v) for v in per_part_ratio[flat_id].tolist()]
            writer.writerow(row)

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "train_script": os.path.abspath(args.train_script),
        "metric_name": "activation-conditioned Con/Stb",
        "resp_definition": "responsibility * ReLU(sim)",
        "activation_score": "sum_n responsibility * ReLU(sim)",
        "activation_domain": "all CUB test images; no owner-class restriction",
        "resp_threshold": float(args.resp_threshold),
        "min_active_images": int(args.min_active_images),
        "num_prototypes": m_count,
        "eligible_prototypes": n_eligible,
        "num_test_images": len(dataset),
        "image_size": image_size,
        "half_size": int(args.half_size),
        "part_threshold": float(args.part_thresh),
        "noise_std_normalized": float(args.noise_std),
        "noise_clip_normalized": float(args.noise_clip),
        "seed": int(args.seed),
        "consistency_percent": con,
        "stability_percent": stb,
        "mean_active_images_per_eligible_prototype": mean_active,
        "median_active_images_per_eligible_prototype": median_active,
        "per_prototype_csv": str(csv_path),
    }
    with open(output_dir / "summary_active_resp_con_stb.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Saved] {csv_path}")
    print(f"[Saved] {output_dir / 'summary_active_resp_con_stb.json'}")


if __name__ == "__main__":
    main()
