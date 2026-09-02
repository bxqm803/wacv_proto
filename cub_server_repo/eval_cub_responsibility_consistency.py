
"""
CUB prototype Consistency using RESPONSIBILITY only.

Correct protocol for the paper:
1) Input image:
      original image -> official CUB bird bbox crop -> Resize(S,S)
2) Official CUB part keypoints:
      original-image coordinates -> subtract exact crop origin
      -> scale by the exact crop width/height to Resize(S,S)
3) Prototype selection:
      active(i,p,k) iff max_n responsibility[i,p,n,k] > threshold
4) Prototype spatial map:
      responsibility[i,p,n,k] only
   (NO ReLU(sim), NO resp_sum activation score)
5) Consistency:
      for each prototype and each of 15 CUB keypoints, compute the hit rate
      among active images where that keypoint is visible.
      Prototype consistency score = max over 15 keypoints.
      Con. = fraction of eligible prototypes whose score >= part_thresh.

This matches the intended criterion described in the paper:
    max_n r_{p,n,k} > 0.1
where r is the token-to-prototype responsibility.
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


# ---------------------------------------------------------------------
# Old-checkpoint metadata repair
# ---------------------------------------------------------------------
_original_load_model = base_eval.load_model


def _detect_backbone(checkpoint):
    raw = checkpoint.get("model", checkpoint)
    if not isinstance(raw, dict):
        return None

    keys = []
    for k in raw:
        k = str(k)
        if k.startswith("module."):
            k = k[7:]
        keys.append(k)

    if any(
        k.startswith(
            (
                "backbone.blocks.",
                "backbone.patch_embed.",
                "backbone.cls_token",
                "backbone.pos_embed",
                "backbone.mask_token",
            )
        )
        for k in keys
    ):
        return "dino"

    if any(k.startswith("backbone.vision_model.") for k in keys):
        return "clip"

    return None


def _repair_checkpoint_config(checkpoint):
    detected = _detect_backbone(checkpoint)
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


# ---------------------------------------------------------------------
# Responsibility-only localization
# ---------------------------------------------------------------------
@torch.inference_mode()
def responsibility_outputs(
    model: torch.nn.Module,
    images: torch.Tensor,
    keypoints: torch.Tensor,
    image_size: int,
    half_size: int,
    amp: bool,
):
    """
    Returns
    -------
    coverage : bool [B, P*K, 15]
        Whether the peak-centered box covers each resized CUB keypoint.
    max_resp : float [B, P*K]
        max_n responsibility_{p,n,k}, used ONLY for active selection.
    """
    device_type = images.device.type
    autocast_enabled = bool(amp and device_type == "cuda")

    with torch.autocast(
        device_type=device_type,
        dtype=torch.float16,
        enabled=autocast_enabled,
    ):
        out = model(images)

    responsibility = out["responsibility"].float()  # [B,P,N,K]

    bsz, n_parts, n_tokens, k_per_part = responsibility.shape
    m_count = n_parts * k_per_part

    # Paper criterion: max_n r_{p,n,k}
    max_resp = responsibility.amax(dim=2).reshape(bsz, m_count)

    grid_h = int(out["grid_h"].item())
    grid_w = int(out["grid_w"].item())
    if n_tokens != grid_h * grid_w:
        raise RuntimeError(
            f"Token/grid mismatch: N={n_tokens}, grid={grid_h}x{grid_w}"
        )

    # Spatial map = responsibility ONLY.
    maps = (
        responsibility
        .permute(0, 1, 3, 2)
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

    # CUBOfficialKeypointTest already returns keypoints AFTER:
    # official bbox crop -> exact resize to image_size.
    x = keypoints[..., 0].unsqueeze(1)  # [B,1,15]
    y = keypoints[..., 1].unsqueeze(1)

    x1 = (peak_x - int(half_size)).clamp_min(0).unsqueeze(-1)
    x2 = (peak_x + int(half_size)).clamp_max(image_size - 1).unsqueeze(-1)
    y1 = (peak_y - int(half_size)).clamp_min(0).unsqueeze(-1)
    y2 = (peak_y + int(half_size)).clamp_max(image_size - 1).unsqueeze(-1)

    coverage = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)

    return coverage, max_resp


def parse_args():
    p = argparse.ArgumentParser(
        "CUB Consistency using max responsibility and responsibility heatmaps"
    )
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

    p.add_argument(
        "--responsibility-threshold",
        type=float,
        default=0.10,
        help="Active iff max_n responsibility > this value.",
    )
    p.add_argument(
        "--half-size",
        type=int,
        default=36,
        help="36 gives the standard 72x72 local region.",
    )
    p.add_argument(
        "--part-thresh",
        type=float,
        default=0.80,
        help="Prototype is consistent if its best keypoint hit rate >= this.",
    )
    p.add_argument(
        "--min-active-images",
        type=int,
        default=1,
    )
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_module = base_eval.import_training_module(args.train_script)
    checkpoint = base_eval.safe_load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected checkpoint dict.")

    checkpoint = _repair_checkpoint_config(checkpoint)
    model = _original_load_model(train_module, checkpoint, device)

    cfg = train_module.cfg
    image_size = int(cfg.image_size)
    parts = tuple(str(x) for x in cfg.parts)
    p_count = len(parts)
    k_count = int(cfg.k_per_part)
    m_count = p_count * k_count

    mean, std = train_module.input_normalization()

    # IMPORTANT:
    # this dataset performs:
    # raw image -> CUB official bbox crop -> Resize(image_size)
    # and applies the same crop+resize geometry to official keypoints.
    dataset = base_eval.CUBOfficialKeypointTest(
        cub_root=args.cub_root,
        image_size=image_size,
        mean=mean,
        std=std,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    q_count = dataset.num_parts
    if q_count != 15:
        raise RuntimeError(f"Expected 15 CUB keypoints, got {q_count}")

    part_names = [dataset.part_names[i + 1] for i in range(q_count)]

    hit_count = torch.zeros((m_count, q_count), dtype=torch.long)
    visible_count = torch.zeros((m_count, q_count), dtype=torch.long)
    active_count = torch.zeros((m_count,), dtype=torch.long)

    print("=" * 100)
    print("CUB RESPONSIBILITY-ONLY CONSISTENCY")
    print("=" * 100)
    print(f"checkpoint             : {args.checkpoint}")
    print(f"image preprocessing    : official bbox crop -> Resize({image_size},{image_size})")
    print(f"keypoint preprocessing : original coords -> crop coords -> resized coords")
    print(f"spatial map            : responsibility ONLY")
    print(
        f"active criterion       : max_n responsibility > "
        f"{args.responsibility_threshold:g}"
    )
    print(f"local region           : {2*args.half_size}x{2*args.half_size}")
    print(f"consistency threshold  : {args.part_thresh:g}")
    print()

    model.eval()

    for images, _labels, keypoints, visible, _image_ids in tqdm(
        loader, desc="Con (responsibility)", dynamic_ncols=True
    ):
        images = images.to(device, non_blocking=True)
        keypoints = keypoints.to(device, non_blocking=True)
        visible = visible.to(device, non_blocking=True)

        coverage, max_resp = responsibility_outputs(
            model=model,
            images=images,
            keypoints=keypoints,
            image_size=image_size,
            half_size=args.half_size,
            amp=args.amp,
        )

        active = max_resp.gt(args.responsibility_threshold)  # [B,M]

        active_count += active.sum(dim=0).cpu().long()

        # only visible CUB keypoints count in each keypoint denominator
        valid = active[:, :, None] & visible[:, None, :]  # [B,M,Q]
        hits = valid & coverage

        visible_count += valid.sum(dim=0).cpu().long()
        hit_count += hits.sum(dim=0).cpu().long()

    ratio = torch.zeros((m_count, q_count), dtype=torch.float32)
    valid_den = visible_count > 0
    ratio[valid_den] = (
        hit_count[valid_den].float() / visible_count[valid_den].float()
    )

    best_ratio, best_q = ratio.max(dim=1)

    eligible = active_count.ge(args.min_active_images)
    num_eligible = int(eligible.sum().item())

    consistent = best_ratio.ge(args.part_thresh)
    con = (
        consistent[eligible].float().mean().item()
        if num_eligible > 0
        else float("nan")
    )

    print("\n" + "-" * 100)
    print(f"Eligible prototypes : {num_eligible}/{m_count}")
    print(f"Consistency (Con.)  : {100.0 * con:.2f}%")
    if num_eligible:
        print(
            f"Active images/proto : mean="
            f"{active_count[eligible].float().mean().item():.2f}, "
            f"median={active_count[eligible].float().median().item():.1f}"
        )
    print("-" * 100)

    csv_path = output_dir / "per_prototype_responsibility_consistency.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "prototype_id",
                "bank",
                "k",
                "active_images",
                "best_keypoint",
                "best_hit_rate",
                "consistent",
            ]
        )
        for j in range(m_count):
            p = j // k_count
            k = j % k_count
            q = int(best_q[j].item())
            w.writerow(
                [
                    j,
                    parts[p],
                    k,
                    int(active_count[j].item()),
                    part_names[q],
                    float(best_ratio[j].item()),
                    int(consistent[j].item()),
                ]
            )

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "metric": "responsibility-only CUB Consistency",
        "image_preprocessing": f"official bbox crop -> Resize({image_size},{image_size})",
        "keypoint_preprocessing": "original coords -> crop coords -> resized coords",
        "spatial_map": "responsibility",
        "active_criterion": "max_n responsibility > threshold",
        "responsibility_threshold": args.responsibility_threshold,
        "half_size": args.half_size,
        "part_thresh": args.part_thresh,
        "eligible_prototypes": num_eligible,
        "num_prototypes": m_count,
        "consistency_percent": 100.0 * con,
    }

    json_path = output_dir / "summary_responsibility_consistency.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Saved] {csv_path}")
    print(f"[Saved] {json_path}")


if __name__ == "__main__":
    main()
