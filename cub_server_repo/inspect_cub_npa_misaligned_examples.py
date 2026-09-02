#!/usr/bin/env python3
"""
Save high-confidence misaligned examples for the current CUB NPA protocol.

This script matches eval_cub_top10pct_vs_all_part_alignment.py:
  - select top 10% prototypes (default 5/50) WITHIN each part and image
    using usage_score = sum_n responsibility
  - localize each selected prototype with the peak of its responsibility map
  - use the standard 72x72 response region (half-size=36)
  - a hit requires ANY visible CUB keypoint assigned to that semantic bank
    to fall inside the response region

For wing/body/tail/feet, the script keeps the highest-usage WRONG examples
from distinct test images and saves:
  1) cropped input + target keypoints + response box
  2) responsibility heatmap overlay + the same annotations

It also writes a CSV with prototype index, usage score, peak location,
nearest target-keypoint distance, and nearest semantic group.

Run this script from the same directory as:
  - eval_cub_top10pct_vs_all_part_alignment.py
  - eval_cub_active_resp_consistency_stability.py
  - your training script
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import eval_cub_active_resp_consistency_stability as base_eval
import eval_cub_top10pct_vs_all_part_alignment as npa


def parse_args():
    p = argparse.ArgumentParser("Inspect CUB NPA false-alignment examples")
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
    p.add_argument("--fraction", type=float, default=0.10)
    p.add_argument("--half-size", type=int, default=36)
    p.add_argument("--num-examples", type=int, default=5)
    p.add_argument(
        "--parts",
        default="wing,body,tail,feet",
        help="Comma-separated banks to inspect.",
    )
    p.add_argument(
        "--candidate-pool",
        type=int,
        default=50,
        help="Keep this many highest-usage distinct-image candidates per part "
             "before taking the final examples.",
    )
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def scalar_image_id(x):
    if torch.is_tensor(x):
        return int(x.item())
    return int(x)


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

    usage = r.sum(dim=2)  # [B,P,K]

    # Match the evaluator exactly for peak localization.
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

    return r, usage, peak_x, peak_y, gh, gw


def denormalize_image(img_chw, mean, std):
    img = img_chw.detach().float().cpu()
    mean_t = torch.tensor(mean, dtype=img.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=img.dtype).view(-1, 1, 1)
    img = img * std_t + mean_t
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def upsample_map(r_flat, gh, gw, image_size):
    x = r_flat.detach().float().view(1, 1, gh, gw)
    x = F.interpolate(
        x,
        size=(image_size, image_size),
        mode="bicubic",
        align_corners=False,
    )[0, 0]
    # Responsibility is non-negative before bicubic interpolation; clamp
    # interpolation undershoot for visualization only.
    x = x.clamp_min(0)
    m = float(x.max().item())
    if m > 0:
        x = x / m
    return x.cpu().numpy()


def group_min_distance(px, py, keypoints, visible, indices):
    dists = []
    for q in indices:
        if bool(visible[q]):
            x = float(keypoints[q, 0])
            y = float(keypoints[q, 1])
            dists.append(math.hypot(px - x, py - y))
    return min(dists) if dists else float("inf")


def nearest_semantic_group(px, py, keypoints, visible, parts, bank_indices):
    best_name = None
    best_dist = float("inf")
    for bank in parts:
        d = group_min_distance(
            px, py, keypoints, visible, bank_indices[bank]
        )
        if d < best_dist:
            best_name = bank
            best_dist = d
    return best_name, best_dist


def prune_candidates(pool, max_items):
    # pool: image_id -> candidate
    if len(pool) <= max_items:
        return pool
    keep = sorted(
        pool.items(),
        key=lambda kv: kv[1]["usage"],
        reverse=True,
    )[:max_items]
    return dict(keep)


def draw_example(candidate, out_path, bank, target_indices, all_parts, bank_indices,
                 half_size, image_size):
    image = candidate["image"]
    heat = candidate["heatmap"]
    keypoints = candidate["keypoints"]
    visible = candidate["visible"]
    px = candidate["peak_x"]
    py = candidate["peak_y"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    for ax in axes:
        ax.imshow(image)
        ax.set_xlim(0, image_size - 1)
        ax.set_ylim(image_size - 1, 0)
        ax.axis("off")

        # Target-bank visible GT keypoints.
        tx, ty = [], []
        for q in target_indices:
            if bool(visible[q]):
                tx.append(float(keypoints[q, 0]))
                ty.append(float(keypoints[q, 1]))
        if tx:
            ax.scatter(tx, ty, marker="*", s=90, label=f"{bank} GT")

        # Prototype peak + 72x72 evaluation region.
        ax.scatter([px], [py], marker="x", s=80, linewidths=2, label="resp. peak")
        x1 = max(0, px - half_size)
        y1 = max(0, py - half_size)
        x2 = min(image_size - 1, px + half_size)
        y2 = min(image_size - 1, py + half_size)
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=2,
            )
        )

    axes[0].set_title(
        f"Input | target={bank}\n"
        f"image={candidate['image_id']}  proto={candidate['proto_idx']}"
    )

    axes[1].imshow(heat, alpha=0.50)
    axes[1].set_title(
        f"Responsibility map | usage={candidate['usage']:.5f}\n"
        f"nearest target={candidate['target_dist']:.1f}px, "
        f"nearest group={candidate['nearest_group']} "
        f"({candidate['nearest_group_dist']:.1f}px)"
    )

    # Put a compact legend on the first panel only.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_contact_sheet(paths, out_path, bank):
    if not paths:
        return
    from PIL import Image, ImageOps, ImageDraw

    ims = [Image.open(p).convert("RGB") for p in paths]
    target_w = max(im.width for im in ims)
    resized = []
    for im in ims:
        if im.width != target_w:
            h = round(im.height * target_w / im.width)
            im = im.resize((target_w, h))
        resized.append(im)

    pad = 12
    title_h = 42
    total_h = title_h + sum(im.height for im in resized) + pad * (len(resized) + 1)
    canvas = Image.new("RGB", (target_w + 2 * pad, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 12), f"{bank}: high-confidence NPA failures", fill="black")

    y = title_h + pad
    for im in resized:
        canvas.paste(im, (pad, y))
        y += im.height + pad

    canvas.save(out_path)


def main():
    args = parse_args()

    if not (0 < args.fraction <= 1):
        raise ValueError("--fraction must be in (0,1].")

    inspect_parts = [
        npa.norm_name(x)
        for x in args.parts.split(",")
        if x.strip()
    ]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    train_module = base_eval.import_training_module(args.train_script)
    checkpoint = base_eval.safe_load(args.checkpoint, map_location="cpu")
    checkpoint = npa.repair_checkpoint_config(checkpoint)
    model = base_eval.load_model(train_module, checkpoint, device)
    model.eval()

    cfg = train_module.cfg
    image_size = int(cfg.image_size)
    parts = [npa.norm_name(x) for x in cfg.parts]
    K = int(cfg.k_per_part)
    P = len(parts)

    for bank in inspect_parts:
        if bank not in parts:
            raise ValueError(f"Requested part {bank!r} not in model parts {parts}")

    mean, std = train_module.input_normalization()
    dataset = base_eval.CUBOfficialKeypointTest(
        cub_root=args.cub_root,
        image_size=image_size,
        mean=mean,
        std=std,
    )

    official_names = [
        npa.norm_name(dataset.part_names[i + 1])
        for i in range(dataset.num_parts)
    ]
    name_to_idx = {name: i for i, name in enumerate(official_names)}

    bank_indices = {}
    for bank in parts:
        wanted = [
            npa.norm_name(x)
            for x in npa.BANK_TO_CUB_PART_NAMES[bank]
        ]
        bank_indices[bank] = [name_to_idx[x] for x in wanted]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    nsel = min(K, max(1, int(math.ceil(K * args.fraction))))
    pools = {bank: {} for bank in inspect_parts}  # bank -> image_id -> best candidate

    print("=" * 100)
    print("CUB NPA FAILURE INSPECTION")
    print("=" * 100)
    print(f"selection       : top {100*args.fraction:.1f}% = {nsel}/{K} prototypes per part/image")
    print("usage score     : sum_n responsibility")
    print("localization    : responsibility-map peak")
    print(f"response region : {2*args.half_size}x{2*args.half_size}")
    print(f"parts           : {inspect_parts}")
    print(f"save per part   : {args.num_examples}")
    print()

    for images, _labels, keypoints, visible, image_ids in tqdm(
        loader, desc="Find failures", dynamic_ncols=True
    ):
        images_dev = images.to(device, non_blocking=True)
        keypoints_dev = keypoints.to(device, non_blocking=True)
        visible_dev = visible.to(device, non_blocking=True)

        r, usage, peak_x, peak_y, gh, gw = forward_batch(
            model, images_dev, image_size, args.amp
        )

        B = images.shape[0]

        for bank in inspect_parts:
            p = parts.index(bank)
            qidx = bank_indices[bank]
            selected = usage[:, p, :].topk(
                k=nsel, dim=1, largest=True, sorted=True
            ).indices

            gx = keypoints_dev[:, qidx, 0]
            gy = keypoints_dev[:, qidx, 1]
            gv = visible_dev[:, qidx]
            group_visible = gv.any(dim=1)

            px = peak_x[:, p, :].gather(1, selected)
            py = peak_y[:, p, :].gather(1, selected)

            x1 = (px - args.half_size).clamp_min(0).unsqueeze(-1)
            x2 = (px + args.half_size).clamp_max(image_size - 1).unsqueeze(-1)
            y1 = (py - args.half_size).clamp_min(0).unsqueeze(-1)
            y2 = (py + args.half_size).clamp_max(image_size - 1).unsqueeze(-1)

            inside = (
                (gx[:, None, :] >= x1)
                & (gx[:, None, :] <= x2)
                & (gy[:, None, :] >= y1)
                & (gy[:, None, :] <= y2)
                & gv[:, None, :]
            )
            hit = inside.any(dim=2)

            for b in range(B):
                if not bool(group_visible[b]):
                    continue

                image_id = scalar_image_id(image_ids[b])
                best_for_image = None

                for j in range(nsel):
                    if bool(hit[b, j]):
                        continue

                    k = int(selected[b, j].item())
                    score = float(usage[b, p, k].item())

                    # For each image keep the strongest wrong selected prototype.
                    if best_for_image is not None and score <= best_for_image["usage"]:
                        continue

                    px0 = int(peak_x[b, p, k].item())
                    py0 = int(peak_y[b, p, k].item())

                    kp_cpu = keypoints[b].detach().cpu()
                    vis_cpu = visible[b].detach().cpu()

                    target_dist = group_min_distance(
                        px0, py0, kp_cpu, vis_cpu, qidx
                    )
                    nearest_group, nearest_group_dist = nearest_semantic_group(
                        px0, py0, kp_cpu, vis_cpu, parts, bank_indices
                    )

                    best_for_image = {
                        "bank": bank,
                        "image_id": image_id,
                        "proto_idx": k,
                        "usage": score,
                        "peak_x": px0,
                        "peak_y": py0,
                        "target_dist": float(target_dist),
                        "nearest_group": nearest_group,
                        "nearest_group_dist": float(nearest_group_dist),
                        "image": denormalize_image(images[b], mean, std),
                        "keypoints": kp_cpu.numpy(),
                        "visible": vis_cpu.numpy().astype(bool),
                        "heatmap": upsample_map(
                            r[b, p, :, k], gh, gw, image_size
                        ),
                    }

                if best_for_image is not None:
                    old = pools[bank].get(image_id)
                    if old is None or best_for_image["usage"] > old["usage"]:
                        pools[bank][image_id] = best_for_image

            pools[bank] = prune_candidates(
                pools[bank], max(args.candidate_pool, args.num_examples)
            )

    rows = []

    for bank in inspect_parts:
        bank_dir = out_root / bank
        bank_dir.mkdir(parents=True, exist_ok=True)

        chosen = sorted(
            pools[bank].values(),
            key=lambda c: c["usage"],
            reverse=True,
        )[:args.num_examples]

        print(f"\n[{bank}] found {len(pools[bank])} retained distinct-image failures; saving {len(chosen)}")

        saved_paths = []
        for rank, c in enumerate(chosen, start=1):
            name = (
                f"{rank:02d}_image{c['image_id']:05d}"
                f"_proto{c['proto_idx']:02d}"
                f"_usage{c['usage']:.5f}.png"
            )
            path = bank_dir / name
            draw_example(
                c,
                path,
                bank,
                bank_indices[bank],
                parts,
                bank_indices,
                args.half_size,
                image_size,
            )
            saved_paths.append(path)

            rows.append({
                "part": bank,
                "rank": rank,
                "image_id": c["image_id"],
                "prototype": c["proto_idx"],
                "usage": c["usage"],
                "peak_x": c["peak_x"],
                "peak_y": c["peak_y"],
                "nearest_target_distance_px": c["target_dist"],
                "nearest_semantic_group": c["nearest_group"],
                "nearest_semantic_group_distance_px": c["nearest_group_dist"],
                "file": str(path),
            })

            print(
                f"  {rank:02d}. image={c['image_id']:5d} proto={c['proto_idx']:2d} "
                f"usage={c['usage']:.5f} target_dist={c['target_dist']:.1f}px "
                f"nearest={c['nearest_group']} ({c['nearest_group_dist']:.1f}px)"
            )

        make_contact_sheet(
            saved_paths,
            out_root / f"{bank}_contact_sheet.png",
            bank,
        )

    csv_path = out_root / "misaligned_examples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "part", "rank", "image_id", "prototype", "usage",
            "peak_x", "peak_y", "nearest_target_distance_px",
            "nearest_semantic_group", "nearest_semantic_group_distance_px",
            "file",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 100)
    print(f"[Saved examples] {out_root}")
    print(f"[Saved summary ] {csv_path}")
    print("Each part also has a *_contact_sheet.png for quick inspection.")


if __name__ == "__main__":
    main()
