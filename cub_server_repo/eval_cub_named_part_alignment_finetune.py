#!/usr/bin/env python3
"""
Named-part alignment evaluation for the CUB shared part-prototype model.

This evaluator is intentionally built on top of the repository's existing
eval_cub_active_resp_consistency_stability.py so that it uses the same:

  - exact training-script import
  - checkpoint["config"] / checkpoint["cfg"] restoration
  - backbone/model reconstruction
  - official CUB bbox crop + Resize(image_size)
  - official CUB keypoint coordinates
  - prototype response definition:
        responsibility * ReLU(sim)

The metric differs from the existing generic Consistency metric.

Generic Consistency asks:
    "Does this prototype repeatedly hit SOME CUB keypoint?"

Named-Part Alignment asks:
    "A prototype belongs to a bank named beak/head/wing/body/tail/feet.
     Does it repeatedly hit the CUB keypoints assigned to THAT named bank?"

For prototype j=(p,k), on each test image i:
    resp_{i,p,n,k} = responsibility * ReLU(sim)
    score_{i,p,k}  = sum_n resp_{i,p,n,k}

Only image-prototype pairs with score > --resp-thresholds are considered
active, matching the repository's activation-conditioned evaluation.

For an active image:
  - the prototype response map is upsampled to image_size x image_size;
  - a square region of size (2*half_size)x(2*half_size) is centered at the
    response peak;
  - target-group visibility = at least one official CUB keypoint belonging
    to this prototype bank is visible in the cropped image;
  - target-group hit = at least one visible target-group keypoint is covered
    by the peak-centered region.

Reported metrics:
  1. conditional_named_part_alignment
       target hits / active images where target group is visible
  2. strict_named_part_alignment
       target hits / all active images
  3. target_visible_rate
       active images with target group visible / all active images
  4. named_part_consistency@T
       fraction of eligible prototypes with conditional alignment >= T
  5. target_is_dominant_rate
       fraction of eligible prototypes whose assigned named group is the
       highest-alignment semantic group among the six groups

CUB 15 official keypoints -> 6 named banks:
  beak : beak
  head : crown, forehead, left eye, right eye, nape, throat
  wing : left wing, right wing
  body : back, belly, breast
  tail : tail
  feet : left leg, right leg

Example:
python eval_cub_named_part_alignment.py \
  --train-script ./train_cub_shared_part_proto_finetune_reg_vitb16_ddp.py \
  --checkpoint ./runs/cub_vitb14_aggressive_woagree_e15_freeze6/best.pth \
  --cub-root ./data/CUB_200_2011 \
  --output-dir ./runs/cub_vitb14_aggressive_woagree_e15_freeze6/named_part_eval \
  --batch-size 8 \
  --resp-thresholds 0.05

Threshold sensitivity:
python eval_cub_named_part_alignment.py \
  --train-script ./train_cub_shared_part_proto_finetune_reg_vitb16_ddp.py \
  --checkpoint ./runs/.../best.pth \
  --cub-root ./data/CUB_200_2011 \
  --output-dir ./runs/.../named_part_eval_sweep \
  --resp-thresholds 0.0 0.02 0.05 0.10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Reuse the repository's existing, already-matched evaluation utilities.
try:
    import eval_cub_active_resp_consistency_stability as base_eval
except ImportError as exc:
    raise ImportError(
        "Could not import eval_cub_active_resp_consistency_stability.py.\n"
        "Place this script in cub_server_repo/ next to that evaluator, or add "
        "cub_server_repo to PYTHONPATH."
    ) from exc


BANK_TO_CUB_PART_NAMES: Dict[str, Tuple[str, ...]] = {
    "beak": ("beak",),
    "head": ("crown", "forehead", "left eye", "right eye", "nape", "throat"),
    "wing": ("left wing", "right wing"),
    "body": ("back", "belly", "breast"),
    "tail": ("tail",),
    "feet": ("left leg", "right leg"),
}


def normalize_name(name: str) -> str:
    return " ".join(
        str(name).strip().lower().replace("_", " ").replace("-", " ").split()
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CUB named-part alignment for shared part-prototype banks."
    )
    p.add_argument(
        "--train-script",
        default="./train_cub_shared_part_proto_finetune_reg_vitb16_ddp.py",
        help="Exact training script used to create the checkpoint.",
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cub-root", default="./data/CUB_200_2011")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--half-size",
        type=int,
        default=36,
        help="Half-size of the peak-centered response window. 36 => 72x72.",
    )
    p.add_argument(
        "--part-thresh",
        type=float,
        default=0.80,
        help="Threshold used for Named-Part Consistency@T.",
    )
    p.add_argument(
        "--resp-thresholds",
        type=float,
        nargs="+",
        default=[0.05],
        help=(
            "One or more activation thresholds on unscaled resp_sum. "
            "Use e.g. 0 0.02 0.05 0.1 for a sensitivity sweep."
        ),
    )
    p.add_argument(
        "--min-active-images",
        type=int,
        default=1,
        help="Minimum active images required for a prototype to enter macro averages.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def resolve_group_indices(
    dataset,
    model_parts: Sequence[str],
) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Resolve official CUB part-name strings to zero-based keypoint indices.
    """
    cub_names = [
        normalize_name(dataset.part_names[i + 1])
        for i in range(dataset.num_parts)
    ]
    name_to_idx = {name: i for i, name in enumerate(cub_names)}

    groups: Dict[str, List[int]] = {}
    normalized_parts: List[str] = []

    for raw_bank in model_parts:
        bank = normalize_name(raw_bank)
        normalized_parts.append(bank)

        if bank not in BANK_TO_CUB_PART_NAMES:
            raise KeyError(
                f"Unknown model part bank '{raw_bank}'. "
                f"Expected one of {list(BANK_TO_CUB_PART_NAMES)}."
            )

        wanted = [normalize_name(x) for x in BANK_TO_CUB_PART_NAMES[bank]]
        missing = [x for x in wanted if x not in name_to_idx]
        if missing:
            raise RuntimeError(
                f"Official CUB parts.txt does not contain {missing} needed by "
                f"bank '{bank}'. Available={cub_names}"
            )
        groups[bank] = [name_to_idx[x] for x in wanted]

    return normalized_parts, groups


@torch.inference_mode()
def batch_outputs(
    model: torch.nn.Module,
    images: torch.Tensor,
    image_size: int,
    half_size: int,
    amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      coverage  : bool  [B, M, Q] coverage of each of 15 official CUB keypoints
      resp_score: float [B, M]    unscaled sum_n resp
      logits    : float [B, C]

    M = P*K.
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

    # Exact local response summand used by the repository's resp_sum evaluator.
    resp = responsibility.float() * F.relu(sim.float())  # [B,P,N,K]
    bsz, n_parts, n_tokens, k_per_part = resp.shape
    m_count = n_parts * k_per_part

    resp_score = resp.sum(dim=2).reshape(bsz, m_count)  # [B,M]

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

    # We return peaks here; coverage is constructed in main because the
    # official keypoints are already provided by the shared CUB dataset.
    peaks = torch.stack([peak_x, peak_y], dim=-1)  # [B,M,2]

    logits = out.get("logits")
    if logits is None:
        raise KeyError("Model output lacks 'logits'.")

    return peaks, resp_score, logits.float()


def peak_keypoint_coverage(
    peaks: torch.Tensor,       # [B,M,2]
    keypoints: torch.Tensor,   # [B,Q,2]
    image_size: int,
    half_size: int,
) -> torch.Tensor:
    """
    Boolean [B,M,Q]: whether a GT keypoint falls inside the peak-centered box.
    """
    peak_x = peaks[..., 0]
    peak_y = peaks[..., 1]

    x = keypoints[..., 0].unsqueeze(1)  # [B,1,Q]
    y = keypoints[..., 1].unsqueeze(1)

    x1 = (peak_x - int(half_size)).clamp_min(0).unsqueeze(-1)
    x2 = (peak_x + int(half_size)).clamp_max(image_size).unsqueeze(-1)
    y1 = (peak_y - int(half_size)).clamp_min(0).unsqueeze(-1)
    y2 = (peak_y + int(half_size)).clamp_max(image_size).unsqueeze(-1)

    return (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)


def safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(num, dtype=torch.float32)
    mask = den > 0
    out[mask] = num[mask].float() / den[mask].float()
    return out


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    if not (0.0 <= args.part_thresh <= 1.0):
        raise ValueError("--part-thresh must be in [0,1].")
    if args.min_active_images < 1:
        raise ValueError("--min-active-images must be >= 1.")

    thresholds = sorted(set(float(x) for x in args.resp_thresholds))
    if not thresholds:
        raise ValueError("--resp-thresholds cannot be empty.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # This is the key compatibility path:
    # 1) import exact training script
    # 2) restore checkpoint["config"] into train_module.cfg
    # 3) rebuild backbone + prototype model
    train_module = base_eval.import_training_module(args.train_script)
    checkpoint = base_eval.safe_load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)}")

    model = base_eval.load_model(train_module, checkpoint, device)
    cfg = train_module.cfg

    image_size = int(cfg.image_size)
    parts_raw = tuple(str(v) for v in cfg.parts)
    p_count = len(parts_raw)
    k_count = int(cfg.k_per_part)
    m_count = p_count * k_count

    mean, std = train_module.input_normalization()
    dataset = base_eval.CUBOfficialKeypointTest(
        cub_root=args.cub_root,
        image_size=image_size,
        mean=mean,
        std=std,
    )
    if dataset.num_parts != 15:
        raise RuntimeError(
            f"Expected 15 official CUB keypoints, got {dataset.num_parts}."
        )

    parts, group_indices = resolve_group_indices(dataset, parts_raw)

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
    )

    print("=" * 100)
    print("CUB NAMED-PART ALIGNMENT")
    print("=" * 100)
    print(f"checkpoint    : {os.path.abspath(args.checkpoint)}")
    print(f"train script  : {os.path.abspath(args.train_script)}")
    print(f"epoch         : {checkpoint.get('epoch')}")
    print(f"saved best acc: {checkpoint.get('best_acc')}")
    print(f"parts         : {parts}")
    print(f"K/part        : {k_count}")
    print(f"image size    : {image_size}")
    print(f"window        : {2 * args.half_size} x {2 * args.half_size}")
    print(f"thresholds    : {thresholds}")
    print("response      : responsibility * ReLU(sim)")
    print()

    for bank in parts:
        names = BANK_TO_CUB_PART_NAMES[bank]
        print(f"  {bank:>5s} <- {', '.join(names)}")
    print()

    # We maintain counts separately for every response threshold.
    stats: Dict[float, Dict[str, torch.Tensor]] = {}
    for thr in thresholds:
        stats[thr] = {
            "active": torch.zeros(m_count, dtype=torch.long),
            "target_visible": torch.zeros(m_count, dtype=torch.long),
            "target_hit": torch.zeros(m_count, dtype=torch.long),
            # Semantic-group diagnostics [M,P].
            "group_visible": torch.zeros((m_count, p_count), dtype=torch.long),
            "group_hit": torch.zeros((m_count, p_count), dtype=torch.long),
        }

    total = 0
    correct = 0

    progress = tqdm(loader, desc="Named-part alignment", dynamic_ncols=True)
    for images, labels, keypoints, visible, _image_ids in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        keypoints = keypoints.to(device, non_blocking=True)
        visible = visible.to(device, non_blocking=True)

        peaks, resp_score, logits = batch_outputs(
            model=model,
            images=images,
            image_size=image_size,
            half_size=args.half_size,
            amp=args.amp,
        )
        coverage = peak_keypoint_coverage(
            peaks=peaks,
            keypoints=keypoints,
            image_size=image_size,
            half_size=args.half_size,
        )  # [B,M,Q]

        total += int(labels.numel())
        correct += int((logits.argmax(dim=1) == labels).sum().item())

        bsz = images.shape[0]

        # For each of the six semantic groups, precompute per-image/prototype:
        # visible? hit?
        group_visible_batch: List[torch.Tensor] = []
        group_hit_batch: List[torch.Tensor] = []

        for bank in parts:
            idxs = group_indices[bank]

            # [B] at least one GT keypoint in this named group is visible.
            gv = visible[:, idxs].any(dim=1)

            # [B,M] at least one *visible* GT keypoint in this group is
            # covered by this prototype's response region.
            vis_sub = visible[:, idxs].unsqueeze(1)       # [B,1,Qg]
            cov_sub = coverage[:, :, idxs]                # [B,M,Qg]
            gh = (cov_sub & vis_sub).any(dim=2)           # [B,M]

            group_visible_batch.append(gv)
            group_hit_batch.append(gh)

        # [B,P]
        gv_all = torch.stack(group_visible_batch, dim=1)
        # [B,M,P]
        gh_all = torch.stack(group_hit_batch, dim=2)

        for thr in thresholds:
            active = resp_score.gt(thr)                    # [B,M]
            s = stats[thr]

            s["active"] += active.sum(dim=0).cpu().long()

            # All six groups: denominator/hits for diagnostic purity/dominance.
            valid_group = active[:, :, None] & gv_all[:, None, :]  # [B,M,P]
            s["group_visible"] += valid_group.sum(dim=0).cpu().long()
            s["group_hit"] += (gh_all & active[:, :, None]).sum(dim=0).cpu().long()

            # Assigned target group for each prototype.
            # Prototype flat id j=(p,k) is assigned to group p.
            for p in range(p_count):
                lo = p * k_count
                hi = (p + 1) * k_count

                active_pk = active[:, lo:hi]               # [B,K]
                target_vis = gv_all[:, p].unsqueeze(1)     # [B,1]
                target_hit = gh_all[:, lo:hi, p]           # [B,K]

                s["target_visible"][lo:hi] += (
                    active_pk & target_vis
                ).sum(dim=0).cpu().long()

                s["target_hit"][lo:hi] += (
                    active_pk & target_hit
                ).sum(dim=0).cpu().long()

    acc = correct / max(1, total)
    print(f"\nRecomputed test accuracy: {acc * 100.0:.3f}% ({correct}/{total})")

    all_threshold_summaries: Dict[str, Any] = {}

    csv_path = output_dir / "per_prototype_named_part_alignment.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = [
            "resp_threshold",
            "prototype_id",
            "part_group",
            "within_part_k",
            "active_test_images",
            "eligible_for_macro",
            "target_visible_active_images",
            "target_visible_rate",
            "target_hits",
            "strict_named_part_alignment",
            "conditional_named_part_alignment",
            "named_part_consistent_at_threshold",
            "dominant_semantic_group",
            "dominant_group_alignment",
            "target_is_dominant",
        ]
        header += [f"alignment_to_{bank}" for bank in parts]
        writer.writerow(header)

        for thr in thresholds:
            s = stats[thr]
            active = s["active"]
            target_visible = s["target_visible"]
            target_hit = s["target_hit"]

            eligible = active.ge(int(args.min_active_images))

            strict = safe_div(target_hit, active)
            conditional = safe_div(target_hit, target_visible)
            target_visible_rate = safe_div(target_visible, active)

            group_ratio = safe_div(s["group_hit"], s["group_visible"])  # [M,P]
            dominant_ratio, dominant_group_idx = group_ratio.max(dim=1)

            target_group_idx = torch.arange(m_count) // k_count
            target_is_dominant = dominant_group_idx.eq(target_group_idx)

            named_consistent = conditional.ge(float(args.part_thresh))

            n_eligible = int(eligible.sum().item())
            if n_eligible == 0:
                print(
                    f"\n[threshold={thr:g}] no eligible prototypes; "
                    "lower threshold/min-active-images."
                )
                continue

            macro_cond = float(conditional[eligible].mean().item())
            macro_strict = float(strict[eligible].mean().item())
            macro_vis = float(target_visible_rate[eligible].mean().item())
            consistency = float(
                named_consistent[eligible].float().mean().item()
            )
            dominant_rate = float(
                target_is_dominant[eligible].float().mean().item()
            )
            mean_active = float(active[eligible].float().mean().item())
            median_active = float(active[eligible].float().median().item())

            per_bank: Dict[str, Any] = {}
            for p, bank in enumerate(parts):
                lo = p * k_count
                hi = (p + 1) * k_count
                bank_eligible = eligible[lo:hi]

                if int(bank_eligible.sum()) == 0:
                    bank_summary = {
                        "eligible_prototypes": 0,
                        "conditional_alignment_percent": None,
                        "strict_alignment_percent": None,
                        "named_part_consistency_percent": None,
                        "target_is_dominant_percent": None,
                    }
                else:
                    bank_summary = {
                        "eligible_prototypes": int(bank_eligible.sum().item()),
                        "conditional_alignment_percent": float(
                            conditional[lo:hi][bank_eligible].mean().item() * 100.0
                        ),
                        "strict_alignment_percent": float(
                            strict[lo:hi][bank_eligible].mean().item() * 100.0
                        ),
                        "named_part_consistency_percent": float(
                            named_consistent[lo:hi][bank_eligible]
                            .float()
                            .mean()
                            .item()
                            * 100.0
                        ),
                        "target_is_dominant_percent": float(
                            target_is_dominant[lo:hi][bank_eligible]
                            .float()
                            .mean()
                            .item()
                            * 100.0
                        ),
                    }
                per_bank[bank] = bank_summary

            summary = {
                "resp_threshold": float(thr),
                "eligible_prototypes": n_eligible,
                "num_prototypes": m_count,
                "conditional_named_part_alignment_percent": macro_cond * 100.0,
                "strict_named_part_alignment_percent": macro_strict * 100.0,
                "target_visible_rate_percent": macro_vis * 100.0,
                f"named_part_consistency@{args.part_thresh:g}_percent": (
                    consistency * 100.0
                ),
                "target_is_dominant_percent": dominant_rate * 100.0,
                "mean_active_images_per_eligible_prototype": mean_active,
                "median_active_images_per_eligible_prototype": median_active,
                "per_bank": per_bank,
            }
            all_threshold_summaries[str(thr)] = summary

            print("\n" + "-" * 100)
            print(f"resp threshold = {thr:g}")
            print("-" * 100)
            print(f"Eligible prototypes             : {n_eligible}/{m_count}")
            print(f"Conditional Named-Part Align.   : {macro_cond * 100.0:.2f}%")
            print(f"Strict Named-Part Align.        : {macro_strict * 100.0:.2f}%")
            print(f"Target-part visible rate        : {macro_vis * 100.0:.2f}%")
            print(
                f"Named-Part Consistency@{args.part_thresh:g}      : "
                f"{consistency * 100.0:.2f}%"
            )
            print(f"Assigned group is dominant      : {dominant_rate * 100.0:.2f}%")
            print(
                f"Active images/proto             : "
                f"mean={mean_active:.2f}, median={median_active:.1f}"
            )

            print("\nPer bank (conditional alignment / consistency):")
            for bank in parts:
                b = per_bank[bank]
                if b["eligible_prototypes"] == 0:
                    print(f"  {bank:>5s}: no eligible prototypes")
                else:
                    print(
                        f"  {bank:>5s}: "
                        f"align={b['conditional_alignment_percent']:.2f}%  "
                        f"cons={b['named_part_consistency_percent']:.2f}%  "
                        f"target-dominant={b['target_is_dominant_percent']:.2f}%"
                    )

            for flat_id in range(m_count):
                p = flat_id // k_count
                k = flat_id % k_count

                row = [
                    float(thr),
                    flat_id,
                    parts[p],
                    k,
                    int(active[flat_id].item()),
                    int(eligible[flat_id].item()),
                    int(target_visible[flat_id].item()),
                    float(target_visible_rate[flat_id].item()),
                    int(target_hit[flat_id].item()),
                    float(strict[flat_id].item()),
                    float(conditional[flat_id].item()),
                    int(named_consistent[flat_id].item()),
                    parts[int(dominant_group_idx[flat_id].item())],
                    float(dominant_ratio[flat_id].item()),
                    int(target_is_dominant[flat_id].item()),
                ]
                row += [float(v) for v in group_ratio[flat_id].tolist()]
                writer.writerow(row)

    final_summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "train_script": os.path.abspath(args.train_script),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_acc": checkpoint.get("best_acc"),
        "recomputed_test_accuracy_percent": acc * 100.0,
        "metric_name": "activation-conditioned named-part alignment",
        "resp_definition": "responsibility * ReLU(sim)",
        "activation_score": "sum_n responsibility * ReLU(sim)",
        "activation_domain": "all CUB test images; no owner-class restriction",
        "parts": parts,
        "cub_keypoint_mapping": {
            bank: list(BANK_TO_CUB_PART_NAMES[bank])
            for bank in parts
        },
        "num_test_images": len(dataset),
        "num_prototypes": m_count,
        "image_size": image_size,
        "half_size": int(args.half_size),
        "response_window_size": int(2 * args.half_size),
        "named_part_consistency_threshold": float(args.part_thresh),
        "min_active_images": int(args.min_active_images),
        "threshold_results": all_threshold_summaries,
        "per_prototype_csv": str(csv_path),
    }

    json_path = output_dir / "summary_named_part_alignment.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print("\n" + "=" * 100)
    print(f"[Saved] {csv_path}")
    print(f"[Saved] {json_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
