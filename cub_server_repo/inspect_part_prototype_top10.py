#!/usr/bin/env python3
"""Save top-10 images per semantic part for a selected prototype.

Output layout:
  output_dir/
    beak/
      rank01_....jpg
      ...
    head/
    wing/
    body/
    tail/
    feet/

Each saved image shows:
  - the resized bird crop used by the model
  - the selected part GDINO bbox in red
  - the strongest prototype patch in yellow (optional)
  - a small text tag with rank / value / prototype id / label / pred
"""

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm


def import_train_module(path: str):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("train_proto_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import training script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Save per-part top prototype images only")
    p.add_argument("--train-script", default="./train_cub_shared_part_proto_finetune_reg.py")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cub-root", default=os.environ.get("CUB_ROOT", "./data/CUB_200_2011"))
    p.add_argument("--gdino-box-dir", default="./artifacts/gdino_part_boxes_gtbbox_warp224_sep")
    p.add_argument("--gdino-train-file", default="train_part_boxes_gtbbox_warp518.pt")
    p.add_argument("--gdino-test-file", default="test_part_boxes_gtbbox_warp518.pt")
    p.add_argument("--output-dir", default="./runs/proto_top10_images")
    p.add_argument("--dino-model", default="dinov2_vitb14")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--parts", default="beak,head,wing,body,tail,feet")
    p.add_argument("--k-per-part", type=int, default=50)
    p.add_argument("--score-mode", default="resp_sum",
                   choices=["resp_sum", "scan_max", "scan_topk", "part_max", "part_topk"])
    p.add_argument("--score-scale", type=float, default=8.0)
    p.add_argument("--scan-topk", type=int, default=5)
    p.add_argument("--readout-mode", default="nonneg", choices=["nonneg", "signed"])
    p.add_argument("--select-by", default="usage", choices=["usage", "score", "contribution"],
                   help="How to choose one prototype per part.")
    p.add_argument("--rank-by", default="score", choices=["usage", "score", "contribution"],
                   help="How to rank top images once the prototype is chosen.")
    p.add_argument("--contribution-class", default="pred", choices=["pred", "true"])
    p.add_argument("--contribution-abs", action="store_true")
    p.add_argument("--splits", default="train,test")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-draw-top-patch", action="store_true",
                   help="Do not draw the strongest patch location.")
    return p.parse_args()


def configure_module(mod, args: argparse.Namespace) -> None:
    cfg = mod.cfg
    cfg.cub_root = os.path.abspath(os.path.expanduser(args.cub_root))
    cfg.gdino_box_dir = os.path.abspath(os.path.expanduser(args.gdino_box_dir))
    cfg.gdino_train_file = args.gdino_train_file
    cfg.gdino_test_file = args.gdino_test_file
    cfg.dino_model = args.dino_model
    cfg.image_size = args.image_size
    cfg.parts = tuple([x.strip().lower() for x in args.parts.split(",") if x.strip()])
    cfg.k_per_part = args.k_per_part
    cfg.score_mode = args.score_mode
    cfg.score_scale = args.score_scale
    cfg.scan_topk = args.scan_topk
    cfg.readout_mode = args.readout_mode
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.amp = not args.no_amp
    if hasattr(cfg, "proto_dropout"):
        cfg.proto_dropout = 0.0
    if hasattr(cfg, "resume"):
        cfg.resume = False


def load_class_names(cub_root: str) -> Dict[int, str]:
    path = os.path.join(cub_root, "classes.txt")
    names: Dict[int, str] = {}
    if not os.path.isfile(path):
        return names
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split(maxsplit=1)
            names[int(idx) - 1] = name
    return names


def build_model_and_load(mod, args: argparse.Namespace, train_ds) -> torch.nn.Module:
    device = mod.DEVICE
    print(f"[DINO] loading {mod.cfg.dino_model}")
    backbone = mod.load_dinov2(mod.cfg.dino_model).to(device)
    mod.set_dino_trainability(backbone, 0, False)

    with torch.no_grad():
        x0, _, _, _ = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)))
        x0 = x0.to(device)
        feats = backbone.forward_features(x0)
        if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
            dim = int(feats["x_norm_patchtokens"].shape[-1])
        else:
            raise RuntimeError("DINOv2 output missing x_norm_patchtokens")

    model = mod.SharedPartPrototypeDINO(
        backbone,
        dim=dim,
        parts=len(mod.cfg.parts),
        k=mod.cfg.k_per_part,
        classes=mod.cfg.num_classes,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[Load] missing keys={len(missing)} unexpected keys={len(unexpected)}")
    print(f"[Load] checkpoint={args.ckpt}")
    model.eval()
    return model


def make_loader(mod, ds):
    return DataLoader(
        ds,
        batch_size=mod.cfg.batch_size,
        shuffle=False,
        num_workers=mod.cfg.num_workers,
        pin_memory=(mod.DEVICE == "cuda"),
        drop_last=False,
    )


def as_metric_array(name: str, score: np.ndarray, usage: np.ndarray, contrib: np.ndarray) -> np.ndarray:
    if name == "score":
        return score
    if name == "usage":
        return usage
    if name == "contribution":
        return contrib
    raise ValueError(name)


def get_crop_image(mod, sample: Dict[str, Any]) -> Image.Image:
    path = os.path.join(mod.cfg.cub_root, "images", sample["relpath"])
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = mod.crop_bbox(img, sample["bbox"])
        img = img.resize((mod.cfg.image_size, mod.cfg.image_size), Image.BICUBIC)
    return img


def draw_sample_image(
    mod,
    ds,
    local_idx: int,
    part_idx: int,
    proto_idx: int,
    value: float,
    split: str,
    pred: int,
    label: int,
    class_names: Dict[int, str],
    top_patch: int,
    grid_hw: Tuple[int, int],
    out_path: str,
    draw_top_patch: bool,
) -> None:
    sample = ds.samples[local_idx]
    img = get_crop_image(mod, sample)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    box = ds.boxes[local_idx, part_idx].cpu().numpy().astype(float).tolist()
    if box[0] >= 0 and box[2] > box[0] and box[3] > box[1]:
        draw.rectangle(box, outline=(255, 0, 0), width=3)

    if draw_top_patch and top_patch >= 0:
        h, w = grid_hw
        r = int(top_patch) // w
        c = int(top_patch) % w
        cell_w = mod.cfg.image_size / float(w)
        cell_h = mod.cfg.image_size / float(h)
        patch_box = [c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h]
        draw.rectangle(patch_box, outline=(255, 255, 0), width=3)

    label_name = class_names.get(label, str(label))
    pred_name = class_names.get(pred, str(pred))
    text = f"r={Path(out_path).stem[:6]} | {mod.cfg.parts[part_idx]} k={proto_idx} | v={value:.4g}\nlabel={label_name}\npred={pred_name}"
    lines = text.split("\n")
    tw = max(draw.textlength(line, font=font) for line in lines) + 6
    th = 12 * len(lines) + 6
    draw.rectangle([0, 0, tw, th], fill=(255, 255, 255))
    draw.multiline_text((3, 3), text, fill=(0, 0, 0), font=font, spacing=1)
    img.save(out_path, quality=95)


def main() -> None:
    args = parse_args()
    mod = import_train_module(args.train_script)
    configure_module(mod, args)
    device = mod.DEVICE
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_names = [x.strip() for x in args.splits.split(",") if x.strip()]
    datasets = {}
    loaders = {}
    for split in split_names:
        gdino_file = mod.cfg.gdino_train_file if split == "train" else mod.cfg.gdino_test_file
        gdino_path = os.path.join(mod.cfg.gdino_box_dir, gdino_file)
        ds = mod.CUBImageWithPartBoxes(split, gdino_path)
        datasets[split] = ds
        loaders[split] = make_loader(mod, ds)

    first_ds = datasets[split_names[0]]
    model = build_model_and_load(mod, args, first_ds)
    class_names = load_class_names(mod.cfg.cub_root)

    all_scores: List[np.ndarray] = []
    all_usage: List[np.ndarray] = []
    all_contrib: List[np.ndarray] = []
    all_patch: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []
    grid_hw = None

    for split in split_names:
        ds = datasets[split]
        loader = loaders[split]
        offset = 0
        for images, boxes, y, relpaths in tqdm(loader, desc=f"Scan {split}", ncols=120):
            images = images.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(mod.cfg.amp and device == "cuda")):
                out = model(images)
            logits = out["logits"].float()
            pred = logits.argmax(dim=1)
            score = out["proto_score"].detach().float()
            usage = out["utilization"].detach().float()
            cw = out["class_weight"].detach().float()
            cls = pred if args.contribution_class == "pred" else y
            contrib = score * cw[cls]
            if args.contribution_abs:
                contrib = contrib.abs()

            relu_sim = F.relu(out["sim"].detach().float())
            gated = out["part_map"].detach().float().unsqueeze(-1) * relu_sim
            patch = gated.argmax(dim=2).to(torch.int16)
            if grid_hw is None:
                grid_hw = (int(out["grid_h"].item()), int(out["grid_w"].item()))

            all_scores.append(score.cpu().numpy())
            all_usage.append(usage.cpu().numpy())
            all_contrib.append(contrib.cpu().numpy())
            all_patch.append(patch.cpu().numpy())

            bs = y.shape[0]
            for i in range(bs):
                local_idx = offset + i
                meta.append({
                    "split": split,
                    "local_idx": int(local_idx),
                    "relpath": str(relpaths[i]),
                    "label": int(y[i].item()),
                    "pred": int(pred[i].item()),
                })
            offset += bs

    scores = np.concatenate(all_scores, axis=0)
    usage = np.concatenate(all_usage, axis=0)
    contrib = np.concatenate(all_contrib, axis=0)
    patch_ids = np.concatenate(all_patch, axis=0)

    select_metric = as_metric_array(args.select_by, scores, usage, contrib)
    rank_metric = as_metric_array(args.rank_by, scores, usage, contrib)
    agg = select_metric.sum(axis=0)  # P,K
    chosen_k = agg.argmax(axis=1).astype(int)

    top_n = int(args.top_n)
    parts = list(mod.cfg.parts)
    draw_top_patch = not args.no_draw_top_patch

    for p_idx, part in enumerate(parts):
        k_idx = int(chosen_k[p_idx])
        part_dir = out_dir / part
        part_dir.mkdir(parents=True, exist_ok=True)

        vals = rank_metric[:, p_idx, k_idx]
        order = np.argsort(-vals)[:top_n]

        for rank, global_idx in enumerate(order, start=1):
            m = meta[int(global_idx)]
            split = m["split"]
            ds = datasets[split]
            local_idx = int(m["local_idx"])
            value = float(vals[global_idx])
            patch_id = int(patch_ids[global_idx, p_idx, k_idx])
            safe_rel = m["relpath"].replace("/", "__").replace(" ", "_")
            img_name = f"rank{rank:02d}_proto{k_idx:03d}_{split}_{safe_rel}.jpg"
            img_path = part_dir / img_name
            draw_sample_image(
                mod, ds, local_idx, p_idx, k_idx, value, split,
                int(m["pred"]), int(m["label"]), class_names,
                patch_id, grid_hw or (16, 16), str(img_path), draw_top_patch,
            )

        print(f"[Saved] {part}: proto k={k_idx} -> {part_dir}")

    print("[Done]")
    print(f"Images saved under: {out_dir}")


if __name__ == "__main__":
    main()
