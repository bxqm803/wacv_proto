#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import eval_cub_active_resp_consistency_stability as base_eval


def parse_args():
    p = argparse.ArgumentParser("Visualize random CUB part-query heatmaps")
    p.add_argument("--train-script",
                   default="./train_cub_shared_part_proto_finetune_reg_vitb16_ddp.py")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cub-root", default="./data/CUB_200_2011")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-images", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha", type=float, default=0.55)
    return p.parse_args()


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


def unnormalize_image(x, mean, std):
    mean_t = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
    img = x.detach().cpu() * std_t + mean_t
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


@torch.inference_mode()
def get_part_maps(model, image, image_size):
    out = model(image.unsqueeze(0))
    part_map = out["part_map"].float()[0]      # [P,N]
    visibility = out["visibility"].float()[0] # [P]

    h = int(out["grid_h"].item())
    w = int(out["grid_w"].item())
    p_count, n_tokens = part_map.shape
    if n_tokens != h * w:
        raise RuntimeError(f"N={n_tokens}, grid={h}x{w}")

    # Conditional spatial distribution given non-null routing.
    spatial = part_map / part_map.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    spatial = spatial.reshape(p_count, 1, h, w)
    up = F.interpolate(
        spatial, size=(image_size, image_size),
        mode="bicubic", align_corners=False
    )[:, 0]

    # Per-part visualization normalization; visibility remains in title.
    up = up - up.amin(dim=(1, 2), keepdim=True)
    up = up / up.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    return up.cpu().numpy(), visibility.cpu().numpy()


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
    parts = [str(v) for v in cfg.parts]
    mean, std = train_module.input_normalization()

    dataset = base_eval.CUBOfficialKeypointTest(
        cub_root=args.cub_root,
        image_size=image_size,
        mean=mean,
        std=std,
    )

    rng = random.Random(args.seed)
    n = min(args.num_images, len(dataset))
    indices = rng.sample(range(len(dataset)), n)

    model.eval()

    for order, idx in enumerate(indices, 1):
        image, label, _keypoints, _visible, image_id = dataset[idx]
        heatmaps, visibility = get_part_maps(
            model, image.to(device), image_size
        )
        rgb = unnormalize_image(image, mean, std)

        n_panels = 1 + len(parts)
        ncols = 4
        nrows = int(np.ceil(n_panels / ncols))

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 4 * nrows)
        )
        axes = np.asarray(axes).reshape(-1)

        axes[0].imshow(rgb)
        axes[0].set_title(
            f"Original | image_id={int(image_id)} | class={int(label)}"
        )
        axes[0].axis("off")

        for p_idx, part in enumerate(parts):
            ax = axes[p_idx + 1]
            ax.imshow(rgb)
            ax.imshow(heatmaps[p_idx], alpha=args.alpha)
            ax.set_title(f"{part} | vis={visibility[p_idx]:.3f}")
            ax.axis("off")

        for ax in axes[n_panels:]:
            ax.axis("off")

        fig.tight_layout()
        save_path = out_dir / (
            f"{order:02d}_image{int(image_id):05d}_class{int(label):03d}.png"
        )
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {save_path}")


if __name__ == "__main__":
    main()
