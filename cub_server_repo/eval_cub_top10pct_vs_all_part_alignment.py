#!/usr/bin/env python3
"""
Evaluate CUB named-part alignment by selecting prototypes WITHIN EACH PART
according to total responsibility.

Default comparison:
  - top 10% prototypes per part per image (5 of 50)
  - all prototypes per part per image (50 of 50)

For image i, part p, prototype k:
    r[i,p,n,k] = responsibility
    usage_score[i,p,k] = sum_n r[i,p,n,k]

Within each part and image, prototypes are ranked by usage_score.

For every selected prototype use:
  1) upsample its responsibility map r[i,p,:,k]
  2) find the peak
  3) form the standard 72x72 local region (half-size 36)
  4) count a hit iff the region contains ANY VISIBLE official CUB keypoint
     belonging to that prototype's assigned semantic part.

CUB semantic groups:
  beak -> beak
  head -> crown, forehead, left eye, right eye, nape, throat
  wing -> left wing OR right wing
  body -> back OR belly OR breast
  tail -> tail
  feet -> left leg OR right leg

Input and keypoints use the same geometry:
  original image -> official CUB bbox crop -> Resize(S,S)

This script reports BOTH:
  * Top-10%-within-part alignment
  * All-prototype alignment

No responsibility threshold is used.
"""

from __future__ import annotations

import argparse
import json
import math
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


def parse_fractions(s: str):
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        v = float(x)
        if not (0 < v <= 1):
            raise ValueError(f"fraction must be in (0,1], got {v}")
        vals.append(v)
    if not vals:
        raise ValueError("No valid fractions.")
    return vals


def parse_args():
    p = argparse.ArgumentParser(
        "CUB named-part alignment: top fraction vs all prototypes"
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
        "--fractions",
        default="0.1,1.0",
        help="Fractions of prototypes selected within each part/image. "
             "Default: 0.1,1.0",
    )
    p.add_argument("--half-size", type=int, default=36)
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


@torch.inference_mode()
def forward_batch(model, images, image_size, amp):
    device_type = images.device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.float16,
        enabled=bool(amp and device_type == "cuda"),
    ):
        out = model(images)

    r = out["responsibility"].float()  # [B,P,N,K]

    B, P, N, K = r.shape
    gh = int(out["grid_h"].item())
    gw = int(out["grid_w"].item())

    if N != gh * gw:
        raise RuntimeError(f"Token/grid mismatch: N={N}, grid={gh}x{gw}")

    # Total responsibility mass = how much this prototype contributes
    # within its part on this image.
    usage = r.sum(dim=2)  # [B,P,K]

    # Upsample every prototype responsibility map once.
    maps = (
        r.permute(0, 1, 3, 2)
         .reshape(B * P * K, 1, gh, gw)
         .contiguous()
    )
    up = F.interpolate(
        maps,
        size=(image_size, image_size),
        mode="bicubic",
        align_corners=False,
    )

    flat = up.flatten(1).argmax(dim=1)
    peak_y = (flat // image_size).reshape(B, P, K)
    peak_x = (flat % image_size).reshape(B, P, K)

    return usage, peak_x, peak_y


def main():
    args = parse_args()
    fractions = parse_fractions(args.fractions)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_module = base_eval.import_training_module(args.train_script)
    checkpoint = base_eval.safe_load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected checkpoint dict.")

    checkpoint = repair_checkpoint_config(checkpoint)
    model = base_eval.load_model(train_module, checkpoint, device)

    cfg = train_module.cfg
    image_size = int(cfg.image_size)
    parts = [norm_name(x) for x in cfg.parts]
    K = int(cfg.k_per_part)
    P = len(parts)

    mean, std = train_module.input_normalization()
    dataset = base_eval.CUBOfficialKeypointTest(
        cub_root=args.cub_root,
        image_size=image_size,
        mean=mean,
        std=std,
    )

    if dataset.num_parts != 15:
        raise RuntimeError(f"Expected 15 official CUB keypoints, got {dataset.num_parts}")

    official_names = [
        norm_name(dataset.part_names[i + 1])
        for i in range(dataset.num_parts)
    ]
    name_to_idx = {name: i for i, name in enumerate(official_names)}

    bank_indices = {}
    for bank in parts:
        if bank not in BANK_TO_CUB_PART_NAMES:
            raise KeyError(
                f"Unknown semantic bank '{bank}'. "
                f"Expected one of {list(BANK_TO_CUB_PART_NAMES)}"
            )
        wanted = [norm_name(x) for x in BANK_TO_CUB_PART_NAMES[bank]]
        missing = [x for x in wanted if x not in name_to_idx]
        if missing:
            raise RuntimeError(
                f"Missing official CUB keypoints for {bank}: {missing}"
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

    # Stats per fraction and semantic bank.
    stats = {}
    proto_use = {}
    proto_hit = {}

    for frac in fractions:
        key = f"{frac:g}"
        stats[key] = {
            bank: {"uses": 0, "hits": 0, "visible_images": 0}
            for bank in parts
        }
        proto_use[key] = torch.zeros((P, K), dtype=torch.long)
        proto_hit[key] = torch.zeros((P, K), dtype=torch.long)

    print("=" * 100)
    print("CUB NAMED-PART ALIGNMENT: WITHIN-PART PROTOTYPE SELECTION")
    print("=" * 100)
    print(f"checkpoint      : {args.checkpoint}")
    print(f"geometry        : official bbox crop -> Resize({image_size},{image_size})")
    print("usage score     : sum_n responsibility")
    print("localization    : responsibility-map peak")
    print(f"local region    : {2*args.half_size}x{2*args.half_size}")
    print(f"fractions       : {fractions}")
    print(f"K per part      : {K}")
    for frac in fractions:
        nsel = min(K, max(1, int(math.ceil(K * frac))))
        print(f"  fraction={frac:g} -> select {nsel}/{K} prototypes per part/image")
    print()

    model.eval()

    for images, _labels, keypoints, visible, _image_ids in tqdm(
        loader, desc="NPA", dynamic_ncols=True
    ):
        images = images.to(device, non_blocking=True)
        keypoints = keypoints.to(device, non_blocking=True)
        visible = visible.to(device, non_blocking=True)

        usage, peak_x, peak_y = forward_batch(
            model=model,
            images=images,
            image_size=image_size,
            amp=args.amp,
        )

        B = images.shape[0]

        for p, bank in enumerate(parts):
            qidx = bank_indices[bank]

            # This semantic part is evaluable if ANY assigned official
            # keypoint is visible in this image.
            group_visible = visible[:, qidx].any(dim=1)  # [B]

            # Coordinates / visibility for the group's keypoints.
            gx = keypoints[:, qidx, 0]  # [B,Qp]
            gy = keypoints[:, qidx, 1]
            gv = visible[:, qidx]       # [B,Qp]

            for frac in fractions:
                key = f"{frac:g}"
                nsel = min(K, max(1, int(math.ceil(K * frac))))

                if nsel == K:
                    selected = torch.arange(
                        K, device=device
                    ).view(1, K).expand(B, K)
                else:
                    selected = usage[:, p, :].topk(
                        k=nsel, dim=1, largest=True, sorted=False
                    ).indices  # [B,nsel]

                px = peak_x[:, p, :].gather(1, selected)  # [B,nsel]
                py = peak_y[:, p, :].gather(1, selected)

                x1 = (px - args.half_size).clamp_min(0).unsqueeze(-1)
                x2 = (px + args.half_size).clamp_max(image_size - 1).unsqueeze(-1)
                y1 = (py - args.half_size).clamp_min(0).unsqueeze(-1)
                y2 = (py + args.half_size).clamp_max(image_size - 1).unsqueeze(-1)

                # [B,nsel,Qp], only visible keypoints can create a hit.
                inside = (
                    (gx[:, None, :] >= x1)
                    & (gx[:, None, :] <= x2)
                    & (gy[:, None, :] >= y1)
                    & (gy[:, None, :] <= y2)
                    & gv[:, None, :]
                )

                hit = inside.any(dim=2)  # [B,nsel]

                # Ignore images where this semantic group has no visible GT
                # landmark at all.
                valid = group_visible[:, None].expand_as(hit)

                valid_sel = selected[valid]
                valid_hit = hit[valid]

                n_uses = int(valid.sum().item())
                n_hits = int(valid_hit.sum().item())

                stats[key][bank]["uses"] += n_uses
                stats[key][bank]["hits"] += n_hits
                stats[key][bank]["visible_images"] += int(group_visible.sum().item())

                # Per-prototype counts.
                if n_uses:
                    use_ids = valid_sel.detach().cpu()
                    hit_ids = valid_sel[valid_hit].detach().cpu()

                    proto_use[key][p].scatter_add_(
                        0,
                        use_ids,
                        torch.ones_like(use_ids, dtype=torch.long),
                    )
                    if hit_ids.numel():
                        proto_hit[key][p].scatter_add_(
                            0,
                            hit_ids,
                            torch.ones_like(hit_ids, dtype=torch.long),
                        )

    print("\n" + "=" * 100)

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "geometry": f"official bbox crop -> Resize({image_size},{image_size})",
        "usage_score": "sum_n responsibility",
        "localization_map": "responsibility",
        "half_size": args.half_size,
        "k_per_part": K,
        "results": {},
    }

    for frac in fractions:
        key = f"{frac:g}"
        nsel = min(K, max(1, int(math.ceil(K * frac))))

        print(f"SELECTION = {100*frac:.1f}% ({nsel}/{K} prototypes per part/image)")
        print("-" * 100)

        part_alignments = []
        total_hits = 0
        total_uses = 0

        frac_result = {
            "fraction": frac,
            "selected_per_part_image": nsel,
            "per_part": {},
        }

        for p, bank in enumerate(parts):
            uses = stats[key][bank]["uses"]
            hits = stats[key][bank]["hits"]
            align = hits / uses if uses else float("nan")

            used_proto_count = int((proto_use[key][p] > 0).sum().item())
            mean_uses = float(proto_use[key][p].float().mean().item())
            median_uses = float(proto_use[key][p].float().median().item())

            print(
                f"{bank:>5s}: alignment={100*align:6.2f}% "
                f"hits={hits:7d}/{uses:7d} "
                f"used_proto={used_proto_count:2d}/{K} "
                f"mean_uses/proto={mean_uses:7.2f} "
                f"median={median_uses:6.1f}"
            )

            if uses:
                part_alignments.append(align)
                total_hits += hits
                total_uses += uses

            frac_result["per_part"][bank] = {
                "alignment_percent": 100 * align if uses else None,
                "hits": hits,
                "uses": uses,
                "used_prototypes": used_proto_count,
                "num_prototypes": K,
                "mean_uses_per_prototype": mean_uses,
                "median_uses_per_prototype": median_uses,
            }

        macro = sum(part_alignments) / len(part_alignments) if part_alignments else float("nan")
        micro = total_hits / total_uses if total_uses else float("nan")

        print("-" * 100)
        print(f"Macro Named-Part Alignment : {100*macro:.2f}%")
        print(f"Micro Named-Part Alignment : {100*micro:.2f}%")
        print()

        frac_result["macro_named_part_alignment_percent"] = 100 * macro
        frac_result["micro_named_part_alignment_percent"] = 100 * micro
        frac_result["total_hits"] = total_hits
        frac_result["total_uses"] = total_uses
        summary["results"][key] = frac_result

    json_path = out_dir / "top_fraction_vs_all_named_part_alignment.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Saved] {json_path}")


if __name__ == "__main__":
    main()
