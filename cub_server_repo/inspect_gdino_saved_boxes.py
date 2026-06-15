#!/usr/bin/env python3
"""Inspect saved GroundingDINO part boxes for CUB.

This script checks whether saved part boxes are missing, oversized, out of range,
or badly distributed. It also saves visualization images with the saved boxes
drawn on the exact CUB GT-bbox crop geometry.

Output:
  output_dir/
    stats.json
    stats.txt
    area_hist.csv
    random_vis/
      *.jpg
    largest_boxes/
      *.jpg
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


DEFAULT_PARTS = ("beak", "head", "wing", "body", "tail", "feet")


def read_kv_text(path: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                k, v = line.split(maxsplit=1)
                out[int(k)] = v
    return out


def read_bboxes(path: str) -> Dict[int, Tuple[float, float, float, float]]:
    out: Dict[int, Tuple[float, float, float, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.strip().split()
            if vals:
                out[int(vals[0])] = tuple(float(x) for x in vals[1:5])
    return out


def build_relpath_to_cub_bbox(cub_root: str) -> Dict[str, Tuple[float, float, float, float]]:
    paths = read_kv_text(os.path.join(cub_root, "images.txt"))
    bboxes = read_bboxes(os.path.join(cub_root, "bounding_boxes.txt"))
    return {rp: bboxes[i] for i, rp in paths.items()}


def crop_bbox(image: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
    x, y, w, h = bbox
    width, height = image.size
    x1 = max(0, min(width - 1, int(np.floor(x))))
    y1 = max(0, min(height - 1, int(np.floor(y))))
    x2 = max(x1 + 1, min(width, int(np.ceil(x + w))))
    y2 = max(y1 + 1, min(height, int(np.ceil(y + h))))
    return image.crop((x1, y1, x2, y2))


def parse_one_sample_part_boxes(elem: Any, parts_in_file: List[str]) -> np.ndarray:
    out = np.full((len(parts_in_file), 4), -1.0, dtype=np.float32)
    if elem is None:
        return out

    if isinstance(elem, dict):
        for j, name in enumerate(parts_in_file):
            val = elem.get(name)
            if val is None:
                continue
            arr = np.asarray(val, dtype=np.float32)
            if arr.ndim == 1 and arr.size == 4:
                out[j] = arr
            elif arr.ndim == 2 and arr.shape[-1] == 4 and arr.shape[0] > 0:
                valid = arr[(arr[:, 0] >= 0) & (arr[:, 2] > arr[:, 0]) & (arr[:, 3] > arr[:, 1])]
                if len(valid):
                    out[j] = np.array(
                        [valid[:, 0].min(), valid[:, 1].min(), valid[:, 2].max(), valid[:, 3].max()],
                        dtype=np.float32,
                    )
        return out

    arr = np.asarray(elem, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[-1] == 4:
        m = min(len(parts_in_file), arr.shape[0])
        out[:m] = arr[:m]
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        # multi boxes for one sample: P,M,4 -> union per part
        m = min(len(parts_in_file), arr.shape[0])
        for p in range(m):
            bx = arr[p]
            valid = (bx[:, 0] >= 0) & (bx[:, 2] > bx[:, 0]) & (bx[:, 3] > bx[:, 1])
            if valid.any():
                v = bx[valid]
                out[p] = np.array([v[:, 0].min(), v[:, 1].min(), v[:, 2].max(), v[:, 3].max()], dtype=np.float32)
    return out


def load_part_boxes(gd: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    parts_in_file = [str(x).lower() for x in gd.get("parts", list(DEFAULT_PARTS))]
    pb = gd["part_boxes_xyxy_pix"]
    p_need = len(parts_in_file)

    if isinstance(pb, torch.Tensor):
        pb = pb.detach().cpu().float()
        if pb.ndim == 3 and pb.shape[-1] == 4:
            out = torch.full((pb.shape[0], p_need, 4), -1.0, dtype=torch.float32)
            out[:, : min(p_need, pb.shape[1])] = pb[:, : min(p_need, pb.shape[1])]
            return out.numpy(), parts_in_file

        if pb.ndim == 4 and pb.shape[-1] == 4:
            b = pb.reshape(pb.shape[0], pb.shape[1], -1, 4)
            valid = (b[..., 0] >= 0) & (b[..., 2] > b[..., 0]) & (b[..., 3] > b[..., 1])
            big = torch.tensor(1e9, dtype=torch.float32)
            neg = torch.tensor(-1e9, dtype=torch.float32)
            x1 = torch.where(valid, b[..., 0], big).min(dim=2).values
            y1 = torch.where(valid, b[..., 1], big).min(dim=2).values
            x2 = torch.where(valid, b[..., 2], neg).max(dim=2).values
            y2 = torch.where(valid, b[..., 3], neg).max(dim=2).values
            any_valid = valid.any(dim=2)
            merged = torch.stack([x1, y1, x2, y2], dim=-1)
            merged[~any_valid] = -1.0
            out = torch.full((pb.shape[0], p_need, 4), -1.0, dtype=torch.float32)
            out[:, : min(p_need, merged.shape[1])] = merged[:, : min(p_need, merged.shape[1])]
            return out.numpy(), parts_in_file

        raise ValueError(f"Unexpected tensor shape: {tuple(pb.shape)}")

    if isinstance(pb, (list, tuple)):
        out = np.full((len(pb), p_need, 4), -1.0, dtype=np.float32)
        for i, elem in enumerate(pb):
            out[i] = parse_one_sample_part_boxes(elem, parts_in_file)
        return out, parts_in_file

    raise TypeError(f"Unsupported part_boxes_xyxy_pix type: {type(pb)}")


def box_valid(boxes: np.ndarray) -> np.ndarray:
    return (boxes[..., 0] >= 0) & (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])


def draw_boxes_on_crop(
    cub_root: str,
    relpath_to_bbox: Dict[str, Tuple[float, float, float, float]],
    relpath: str,
    boxes: np.ndarray,
    parts: List[str],
    img_size: int,
    out_path: Path,
    title: str,
) -> None:
    image_path = os.path.join(cub_root, "images", relpath)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = crop_bbox(img, relpath_to_bbox[relpath])
        img = img.resize((img_size, img_size), Image.BICUBIC)

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # deterministic simple colors
    colors = [
        (255, 0, 0),      # red
        (255, 128, 0),    # orange
        (0, 180, 0),      # green
        (0, 128, 255),    # blue
        (180, 0, 255),    # purple
        (255, 0, 180),    # pink
        (0, 0, 0),
    ]

    for p, part in enumerate(parts):
        b = boxes[p].astype(float).tolist()
        if b[0] < 0 or b[2] <= b[0] or b[3] <= b[1]:
            continue
        x1, y1, x2, y2 = b
        color = colors[p % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.rectangle([x1, max(0, y1 - 12), x1 + 48, y1], fill=(255, 255, 255))
        draw.text((x1 + 2, max(0, y1 - 11)), part[:8], fill=color, font=font)

    lines = [title, relpath]
    tw = max(draw.textlength(line, font=font) for line in lines) + 6
    th = 12 * len(lines) + 6
    draw.rectangle([0, 0, tw, th], fill=(255, 255, 255))
    draw.multiline_text((3, 3), "\n".join(lines), fill=(0, 0, 0), font=font, spacing=1)
    img.save(out_path, quality=95)


def summarize_one(pt_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    gd = torch.load(pt_path, map_location="cpu")
    if "relpaths" not in gd or "part_boxes_xyxy_pix" not in gd:
        raise KeyError(f"Bad GDINO cache: {pt_path}")

    relpaths = [str(x) for x in gd["relpaths"]]
    img_size = int(gd.get("img_size", args.assume_img_size))
    boxes, parts = load_part_boxes(gd)
    n, p, _ = boxes.shape

    valid = box_valid(boxes)
    clipped = boxes.copy()
    clipped[..., [0, 2]] = np.clip(clipped[..., [0, 2]], 0, img_size)
    clipped[..., [1, 3]] = np.clip(clipped[..., [1, 3]], 0, img_size)

    w = np.clip(clipped[..., 2] - clipped[..., 0], 0, None)
    h = np.clip(clipped[..., 3] - clipped[..., 1], 0, None)
    area = (w * h) / float(img_size * img_size)
    area[~valid] = np.nan

    out_of_range = valid & (
        (boxes[..., 0] < 0) | (boxes[..., 1] < 0) |
        (boxes[..., 2] > img_size) | (boxes[..., 3] > img_size)
    )
    full_like = valid & (area >= args.large_area_threshold)
    tiny = valid & (area <= args.tiny_area_threshold)

    stats = {
        "file": pt_path,
        "num_images": n,
        "parts": parts,
        "img_size": img_size,
        "raw_box_shape": list(boxes.shape),
        "thresholds": {
            "large_area_threshold": args.large_area_threshold,
            "tiny_area_threshold": args.tiny_area_threshold,
        },
        "per_part": {},
    }

    for j, part in enumerate(parts):
        a = area[:, j]
        v = valid[:, j]
        vals = a[v]
        if vals.size == 0:
            row = {
                "valid": 0,
                "missing": int((~v).sum()),
                "valid_ratio": 0.0,
            }
        else:
            row = {
                "valid": int(v.sum()),
                "missing": int((~v).sum()),
                "valid_ratio": float(v.mean()),
                "area_mean": float(np.nanmean(vals)),
                "area_median": float(np.nanmedian(vals)),
                "area_p90": float(np.nanpercentile(vals, 90)),
                "area_p95": float(np.nanpercentile(vals, 95)),
                "area_max": float(np.nanmax(vals)),
                "large_count": int(full_like[:, j].sum()),
                "large_ratio": float(full_like[:, j].sum() / max(1, v.sum())),
                "tiny_count": int(tiny[:, j].sum()),
                "tiny_ratio": float(tiny[:, j].sum() / max(1, v.sum())),
                "out_of_range_count": int(out_of_range[:, j].sum()),
                "out_of_range_ratio": float(out_of_range[:, j].sum() / max(1, v.sum())),
            }
        stats["per_part"][part] = row

    return {
        "gd": gd,
        "relpaths": relpaths,
        "boxes": boxes,
        "parts": parts,
        "img_size": img_size,
        "valid": valid,
        "area": area,
        "stats": stats,
    }


def write_stats_txt(path: Path, all_stats: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    for st in all_stats:
        lines.append(f"FILE: {st['file']}")
        lines.append(f"  num_images={st['num_images']} img_size={st['img_size']} box_shape={st['raw_box_shape']}")
        lines.append("  part              valid  missing  area_mean  area_med  p90    p95    max    large%  tiny%  oor%")
        for part, r in st["per_part"].items():
            if r.get("valid", 0) == 0:
                lines.append(f"  {part:<16} {0:>5}  {r.get('missing',0):>7}  NA")
            else:
                lines.append(
                    f"  {part:<16} {r['valid']:>5}  {r['missing']:>7}  "
                    f"{r['area_mean']:.3f}      {r['area_median']:.3f}    {r['area_p90']:.3f}  {r['area_p95']:.3f}  {r['area_max']:.3f}  "
                    f"{100*r['large_ratio']:.1f}    {100*r['tiny_ratio']:.1f}   {100*r['out_of_range_ratio']:.1f}"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("Inspect saved GroundingDINO CUB part boxes")
    parser.add_argument("--cub-root", default="./data/CUB_200_2011")
    parser.add_argument("--gdino-box-dir", default="./artifacts/gdino_part_boxes_gtbbox_warp224_sep")
    parser.add_argument("--train-file", default="train_part_boxes_gtbbox_warp518.pt")
    parser.add_argument("--test-file", default="test_part_boxes_gtbbox_warp518.pt")
    parser.add_argument("--output-dir", default="./runs/gdino_box_inspect")
    parser.add_argument("--assume-img-size", type=int, default=224,
                        help="Used only if the pt file does not contain img_size.")
    parser.add_argument("--large-area-threshold", type=float, default=0.45,
                        help="Boxes with area ratio above this are suspiciously large.")
    parser.add_argument("--tiny-area-threshold", type=float, default=0.002,
                        help="Boxes with area ratio below this are suspiciously tiny.")
    parser.add_argument("--num-random-vis", type=int, default=80)
    parser.add_argument("--num-largest-vis-per-part", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    random_dir = out_dir / "random_vis"
    largest_dir = out_dir / "largest_boxes"
    out_dir.mkdir(parents=True, exist_ok=True)
    random_dir.mkdir(parents=True, exist_ok=True)
    largest_dir.mkdir(parents=True, exist_ok=True)

    files = [
        os.path.join(args.gdino_box_dir, args.train_file),
        os.path.join(args.gdino_box_dir, args.test_file),
    ]
    relpath_to_bbox = build_relpath_to_cub_bbox(args.cub_root)

    results = []
    for pt_path in files:
        print(f"[Load] {pt_path}")
        res = summarize_one(pt_path, args)
        results.append(res)

    stats_list = [r["stats"] for r in results]
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_list, f, indent=2)
    write_stats_txt(out_dir / "stats.txt", stats_list)

    # area histogram csv
    with open(out_dir / "area_hist.csv", "w", encoding="utf-8") as f:
        f.write("file,part,bin_left,bin_right,count\n")
        bins = np.array([0, .002, .005, .01, .02, .05, .10, .20, .30, .45, .60, .80, 1.01], dtype=np.float32)
        for res in results:
            fname = os.path.basename(res["stats"]["file"])
            area = res["area"]
            valid = res["valid"]
            parts = res["parts"]
            for j, part in enumerate(parts):
                vals = area[:, j][valid[:, j]]
                hist, edges = np.histogram(vals, bins=bins)
                for c, l, rr in zip(hist, edges[:-1], edges[1:]):
                    f.write(f"{fname},{part},{float(l):.4f},{float(rr):.4f},{int(c)}\n")

    # Print summary
    print("\n==== Box stats ====")
    print((out_dir / "stats.txt").read_text(encoding="utf-8"))

    # random visualization
    for res in results:
        relpaths = res["relpaths"]
        boxes = res["boxes"]
        parts = res["parts"]
        img_size = res["img_size"]
        fname = os.path.splitext(os.path.basename(res["stats"]["file"]))[0]
        idxs = list(range(len(relpaths)))
        random.shuffle(idxs)
        idxs = idxs[: min(args.num_random_vis, len(idxs))]
        for rank, i in enumerate(idxs, start=1):
            relpath = relpaths[i]
            if relpath not in relpath_to_bbox:
                continue
            out_path = random_dir / f"{fname}_random{rank:03d}_{relpath.replace('/', '__')}.jpg"
            draw_boxes_on_crop(
                args.cub_root, relpath_to_bbox, relpath,
                boxes[i], parts, img_size, out_path,
                title=f"{fname} random {rank}",
            )

    # largest suspicious box visualization per part
    for res in results:
        relpaths = res["relpaths"]
        boxes = res["boxes"]
        parts = res["parts"]
        img_size = res["img_size"]
        area = res["area"]
        valid = res["valid"]
        fname = os.path.splitext(os.path.basename(res["stats"]["file"]))[0]
        for j, part in enumerate(parts):
            part_dir = largest_dir / fname / part
            part_dir.mkdir(parents=True, exist_ok=True)
            vals = area[:, j].copy()
            vals[~valid[:, j]] = -1
            order = np.argsort(-vals)[: args.num_largest_vis_per_part]
            for rank, i in enumerate(order, start=1):
                if vals[i] < 0:
                    continue
                relpath = relpaths[int(i)]
                if relpath not in relpath_to_bbox:
                    continue
                out_path = part_dir / f"rank{rank:02d}_area{vals[i]:.3f}_{relpath.replace('/', '__')}.jpg"
                draw_boxes_on_crop(
                    args.cub_root, relpath_to_bbox, relpath,
                    boxes[int(i)], parts, img_size, out_path,
                    title=f"{fname} {part} area={vals[i]:.3f}",
                )

    print("[Done]")
    print(f"stats:        {out_dir / 'stats.txt'}")
    print(f"random_vis:   {random_dir}")
    print(f"largest_vis:  {largest_dir}")


if __name__ == "__main__":
    main()
