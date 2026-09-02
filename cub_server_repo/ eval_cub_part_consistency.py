#!/usr/bin/env python3
"""
CUB bank-aware Part Consistency using RESPONSIBILITY only.

Protocol
--------
1) Input/keypoint geometry:
   raw image -> official CUB bbox crop -> Resize(S,S)
   official keypoints are transformed with the exact same crop+resize geometry.

2) Prototype activity:
   active(i,p,k) iff max_n responsibility[i,p,n,k] > --responsibility-threshold

3) Prototype localization map:
   responsibility[i,p,n,k] only
   (no ReLU(sim), no evidence score)

4) Ordinary Consistency:
   For each prototype, compute hit rate for each of the 15 CUB keypoints,
   then take the maximum over all 15.

5) Part Consistency (bank-aware):
   For a prototype in semantic bank p, compute the same per-keypoint hit rates,
   but take the maximum ONLY over keypoints assigned to bank p.

   CUB mapping:
     beak -> beak
     head -> crown, forehead, left eye, right eye, nape, throat
     wing -> left wing, right wing
     body -> back, belly, breast
     tail -> tail
     feet -> left leg, right leg

A prototype is consistent when its score >= --part-thresh.

This preserves the original keypoint-based Consistency protocol while requiring
the best-matching keypoint to belong to the prototype's assigned semantic bank.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import eval_cub_active_resp_consistency_stability as base_eval


BANK_TO_CUB_PART_NAMES = {
    "beak": ("beak",),
    "head": ("crown", "forehead", "left eye", "right eye", "nape", "throat"),
    "wing": ("left wing", "right wing"),
    "body": ("back", "belly", "breast"),
    "tail": ("tail",),
    "feet": ("left leg", "right leg"),
}


def norm_name(s: str) -> str:
    return " ".join(str(s).lower().replace("_", " ").replace("-", " ").split())


def detect_backbone(checkpoint):
    raw = checkpoint.get("model", checkpoint)
    if not isinstance(raw, dict):
        return None
    keys = []
    for k in raw:
        k = str(k)
        if k.startswith("module."):
            k = k[7:]
        keys.append(k)

    if any(k.startswith(("backbone.blocks.", "backbone.patch_embed.",
                         "backbone.cls_token", "backbone.pos_embed",
                         "backbone.mask_token")) for k in keys):
        return "dino"
    if any(k.startswith("backbone.vision_model.") for k in keys):
        return "clip"
    return None


def repair_checkpoint_config(checkpoint):
    detected = detect_backbone(checkpoint)
    if detected is None:
        return checkpoint

    ckpt = dict(checkpoint)
    if isinstance(checkpoint.get("config"), dict):
        cfg_key = "config"
    elif isinstance(checkpoint.get("cfg"), dict):
        cfg_key = "cfg"
    else:
        return checkpoint

    saved = dict(checkpoint[cfg_key])
    old = saved.get("backbone")
    if old != detected:
        print(f"[Config repair] backbone {old!r} -> {detected!r}")
        saved["backbone"] = detected
        ckpt[cfg_key] = saved
    return ckpt


@torch.inference_mode()
def responsibility_coverage(model, images, keypoints, image_size, half_size, amp):
    device_type = images.device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.float16,
        enabled=bool(amp and device_type == "cuda"),
    ):
        out = model(images)

    r = out["responsibility"].float()  # [B,P,N,K]
    bsz, p_count, n_tokens, k_count = r.shape
    m_count = p_count * k_count

    # Paper's active criterion.
    max_resp = r.amax(dim=2).reshape(bsz, m_count)

    gh = int(out["grid_h"].item())
    gw = int(out["grid_w"].item())
    if n_tokens != gh * gw:
        raise RuntimeError(f"N={n_tokens}, grid={gh}x{gw}")

    maps = (
        r.permute(0, 1, 3, 2)
         .reshape(bsz * m_count, 1, gh, gw)
         .contiguous()
    )
    up = F.interpolate(
        maps,
        size=(image_size, image_size),
        mode="bicubic",
        align_corners=False,
    )

    flat_idx = up.flatten(1).argmax(dim=1)
    py = (flat_idx // image_size).view(bsz, m_count)
    px = (flat_idx % image_size).view(bsz, m_count)

    x = keypoints[..., 0].unsqueeze(1)  # [B,1,Q]
    y = keypoints[..., 1].unsqueeze(1)

    x1 = (px - half_size).clamp_min(0).unsqueeze(-1)
    x2 = (px + half_size).clamp_max(image_size - 1).unsqueeze(-1)
    y1 = (py - half_size).clamp_min(0).unsqueeze(-1)
    y2 = (py + half_size).clamp_max(image_size - 1).unsqueeze(-1)

    coverage = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
    return coverage, max_resp


def parse_args():
    p = argparse.ArgumentParser("CUB responsibility-based bank-aware Part Consistency")
    p.add_argument(
        "--train-script",
        default="./train_cub_shared_part_proto_finetune_reg_vitb16_ddp.py",
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cub-root", default="./data/CUB_200_2011")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--responsibility-threshold", type=float, default=0.10)
    p.add_argument("--half-size", type=int, default=36)
    p.add_argument("--part-thresh", type=float, default=0.80)
    p.add_argument("--min-active-images", type=int, default=1)
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_module = base_eval.import_training_module(args.train_script)
    checkpoint = base_eval.safe_load(args.checkpoint, map_location="cpu")
    checkpoint = repair_checkpoint_config(checkpoint)

    model = base_eval.load_model(train_module, checkpoint, device)
    cfg = train_module.cfg

    image_size = int(cfg.image_size)
    parts = [norm_name(x) for x in cfg.parts]
    k_count = int(cfg.k_per_part)
    p_count = len(parts)
    m_count = p_count * k_count

    mean, std = train_module.input_normalization()
    dataset = base_eval.CUBOfficialKeypointTest(
        cub_root=args.cub_root,
        image_size=image_size,
        mean=mean,
        std=std,
    )

    if dataset.num_parts != 15:
        raise RuntimeError(f"Expected 15 CUB keypoints, got {dataset.num_parts}")

    official_names = [
        norm_name(dataset.part_names[i + 1])
        for i in range(dataset.num_parts)
    ]
    name_to_idx = {name: i for i, name in enumerate(official_names)}

    bank_indices = {}
    for bank in parts:
        if bank not in BANK_TO_CUB_PART_NAMES:
            raise KeyError(
                f"Unknown bank '{bank}'. Expected {list(BANK_TO_CUB_PART_NAMES)}"
            )
        wanted = [norm_name(x) for x in BANK_TO_CUB_PART_NAMES[bank]]
        missing = [x for x in wanted if x not in name_to_idx]
        if missing:
            raise RuntimeError(
                f"Missing official CUB keypoints for bank {bank}: {missing}"
            )
        bank_indices[bank] = [name_to_idx[x] for x in wanted]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    hit_count = torch.zeros((m_count, 15), dtype=torch.long)
    visible_count = torch.zeros((m_count, 15), dtype=torch.long)
    active_count = torch.zeros(m_count, dtype=torch.long)

    print("=" * 100)
    print("CUB BANK-AWARE PART CONSISTENCY")
    print("=" * 100)
    print(f"checkpoint             : {args.checkpoint}")
    print(f"geometry               : official bbox crop -> Resize({image_size},{image_size})")
    print("spatial map            : responsibility only")
    print(
        f"active criterion       : max_n responsibility > "
        f"{args.responsibility_threshold:g}"
    )
    print(f"local region           : {2*args.half_size}x{2*args.half_size}")
    print(f"consistency threshold  : {args.part_thresh:g}")
    print("bank mapping:")
    for bank in parts:
        print(f"  {bank:>5s} <- {', '.join(BANK_TO_CUB_PART_NAMES[bank])}")

    model.eval()

    for images, _labels, keypoints, visible, _image_ids in tqdm(
        loader, desc="Part consistency", dynamic_ncols=True
    ):
        images = images.to(device, non_blocking=True)
        keypoints = keypoints.to(device, non_blocking=True)
        visible = visible.to(device, non_blocking=True)

        coverage, max_resp = responsibility_coverage(
            model=model,
            images=images,
            keypoints=keypoints,
            image_size=image_size,
            half_size=args.half_size,
            amp=args.amp,
        )

        active = max_resp.gt(args.responsibility_threshold)  # [B,M]
        active_count += active.sum(dim=0).cpu().long()

        valid = active[:, :, None] & visible[:, None, :]
        hits = valid & coverage

        visible_count += valid.sum(dim=0).cpu().long()
        hit_count += hits.sum(dim=0).cpu().long()

    ratios = torch.zeros((m_count, 15), dtype=torch.float32)
    mask = visible_count > 0
    ratios[mask] = hit_count[mask].float() / visible_count[mask].float()

    # Ordinary consistency: best of all 15 official keypoints.
    ordinary_best, ordinary_idx = ratios.max(dim=1)

    # Bank-aware part consistency: best keypoint only within assigned bank.
    part_best = torch.zeros(m_count, dtype=torch.float32)
    part_best_idx = torch.zeros(m_count, dtype=torch.long)

    for p, bank in enumerate(parts):
        lo, hi = p * k_count, (p + 1) * k_count
        idxs = bank_indices[bank]
        local = ratios[lo:hi][:, idxs]
        vals, local_arg = local.max(dim=1)
        part_best[lo:hi] = vals
        idx_tensor = torch.tensor(idxs, dtype=torch.long)
        part_best_idx[lo:hi] = idx_tensor[local_arg]

    eligible = active_count.ge(args.min_active_images)
    n_eligible = int(eligible.sum().item())

    ordinary_consistent = ordinary_best.ge(args.part_thresh)
    part_consistent = part_best.ge(args.part_thresh)

    ordinary_con = (
        ordinary_consistent[eligible].float().mean().item()
        if n_eligible else float("nan")
    )
    part_con = (
        part_consistent[eligible].float().mean().item()
        if n_eligible else float("nan")
    )

    print("\n" + "-" * 100)
    print(f"Eligible prototypes       : {n_eligible}/{m_count}")
    print(f"Ordinary Consistency      : {100*ordinary_con:.2f}%")
    print(f"Part Consistency          : {100*part_con:.2f}%")
    print(
        f"Active images/proto       : mean="
        f"{active_count[eligible].float().mean().item():.2f}, "
        f"median={active_count[eligible].float().median().item():.1f}"
        if n_eligible else "Active images/proto       : n/a"
    )

    print("\nPer bank Part Consistency:")
    per_bank = {}
    for p, bank in enumerate(parts):
        lo, hi = p * k_count, (p + 1) * k_count
        e = eligible[lo:hi]
        n = int(e.sum().item())
        val = (
            part_consistent[lo:hi][e].float().mean().item()
            if n else float("nan")
        )
        per_bank[bank] = {
            "eligible": n,
            "part_consistency_percent": 100 * val if n else None,
        }
        print(
            f"  {bank:>5s}: "
            f"{100*val:.2f}% ({int(part_consistent[lo:hi][e].sum())}/{n})"
            if n else f"  {bank:>5s}: no eligible prototypes"
        )
    print("-" * 100)

    csv_path = out_dir / "per_prototype_part_consistency.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "prototype_id", "bank", "k", "active_images",
            "ordinary_best_keypoint", "ordinary_best_rate",
            "part_best_keypoint", "part_best_rate",
            "ordinary_consistent", "part_consistent",
        ])
        for j in range(m_count):
            p = j // k_count
            k = j % k_count
            oq = int(ordinary_idx[j].item())
            pq = int(part_best_idx[j].item())
            w.writerow([
                j, parts[p], k, int(active_count[j].item()),
                official_names[oq], float(ordinary_best[j].item()),
                official_names[pq], float(part_best[j].item()),
                int(ordinary_consistent[j].item()),
                int(part_consistent[j].item()),
            ])

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "spatial_map": "responsibility",
        "active_criterion": "max_n responsibility > threshold",
        "responsibility_threshold": args.responsibility_threshold,
        "geometry": f"official bbox crop -> Resize({image_size},{image_size})",
        "half_size": args.half_size,
        "part_thresh": args.part_thresh,
        "eligible_prototypes": n_eligible,
        "num_prototypes": m_count,
        "ordinary_consistency_percent": 100 * ordinary_con,
        "part_consistency_percent": 100 * part_con,
        "per_bank": per_bank,
        "bank_mapping": {
            k: list(v) for k, v in BANK_TO_CUB_PART_NAMES.items()
        },
    }

    json_path = out_dir / "summary_part_consistency.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Saved] {csv_path}")
    print(f"[Saved] {json_path}")


if __name__ == "__main__":
    main()
