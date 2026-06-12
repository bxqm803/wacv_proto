#!/usr/bin/env python3
"""Generate GroundingDINO part boxes for CUB bird-bbox crops.

Each CUB image is cropped to its ground-truth bird bounding box, warped to
518x518, and queried with six named bird parts. The output format is consumed by
train_cub_semantic_part_additive_proto_server.py.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


PARTS = ("beak", "head", "wing", "body", "tail", "feet")
PROMPTS = ("bird beak", "bird head", "bird wing", "bird body", "bird tail", "bird foot")
ALIASES = {
    "beak": ("beak", "bill"),
    "head": ("head",),
    "wing": ("wing",),
    "body": ("body", "torso"),
    "tail": ("tail",),
    "feet": ("foot", "feet", "leg"),
}


def read_kv_text(path: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                k, v = line.split(maxsplit=1)
                out[int(k)] = v
    return out


def read_kv_int(path: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                k, v = line.split()
                out[int(k)] = int(v)
    return out


def read_bboxes(path: str) -> Dict[int, Tuple[float, float, float, float]]:
    out: Dict[int, Tuple[float, float, float, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.strip().split()
            if vals:
                out[int(vals[0])] = tuple(float(x) for x in vals[1:5])
    return out


@dataclass(frozen=True)
class Sample:
    image_id: int
    relpath: str
    bbox: Tuple[float, float, float, float]


def build_samples(cub_root: str, split: str) -> List[Sample]:
    paths = read_kv_text(os.path.join(cub_root, "images.txt"))
    split_map = read_kv_int(os.path.join(cub_root, "train_test_split.txt"))
    bboxes = read_bboxes(os.path.join(cub_root, "bounding_boxes.txt"))
    want_train = split == "train"
    out: List[Sample] = []
    for image_id in sorted(paths):
        if (split_map[image_id] == 1) == want_train:
            out.append(Sample(image_id, paths[image_id], bboxes[image_id]))
    return out


def crop_bbox(image: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
    x, y, w, h = bbox
    width, height = image.size
    x1 = max(0, min(width - 1, int(np.floor(x))))
    y1 = max(0, min(height - 1, int(np.floor(y))))
    x2 = max(x1 + 1, min(width, int(np.ceil(x + w))))
    y2 = max(y1 + 1, min(height, int(np.ceil(y + h))))
    return image.crop((x1, y1, x2, y2))


def load_crop(cub_root: str, sample: Sample, image_size: int) -> Image.Image:
    path = os.path.join(cub_root, "images", sample.relpath)
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = crop_bbox(image, sample.bbox)
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        return image.copy()


def part_from_label(label) -> int:
    if isinstance(label, (list, tuple)):
        text = " ".join(str(x) for x in label).lower()
    else:
        text = str(label).lower()
    for i, part in enumerate(PARTS):
        if any(alias in text for alias in ALIASES[part]):
            return i
    return -1


def output_path(output_dir: str, split: str) -> str:
    return os.path.join(output_dir, f"{split}_part_boxes_gtbbox_warp518.pt")


def save_state(
    path: str,
    relpaths: Sequence[str],
    boxes: torch.Tensor,
    scores: torch.Tensor,
    processed: int,
    image_size: int,
    model_id: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(
        {
            "relpaths": list(relpaths),
            "parts": list(PARTS),
            "prompts": list(PROMPTS),
            "part_boxes_xyxy_pix": boxes,
            "part_scores": scores,
            "processed": int(processed),
            "img_size": int(image_size),
            "model_id": model_id,
            "crop": "CUB ground-truth bbox, warped to square",
        },
        tmp,
    )
    os.replace(tmp, path)


def build_split(processor, model, args: argparse.Namespace, split: str) -> None:
    samples = build_samples(args.cub_root, split)
    relpaths = [x.relpath for x in samples]
    path = output_path(args.output_dir, split)

    boxes = torch.full(
        (len(samples), len(PARTS), args.max_boxes_per_part, 4),
        -1.0,
        dtype=torch.float32,
    )
    scores = torch.full(
        (len(samples), len(PARTS), args.max_boxes_per_part),
        -1.0,
        dtype=torch.float32,
    )
    start = 0

    if os.path.isfile(path) and args.resume and not args.overwrite:
        old = torch.load(path, map_location="cpu")
        if old.get("relpaths") == relpaths and tuple(old["part_boxes_xyxy_pix"].shape) == tuple(boxes.shape):
            boxes.copy_(old["part_boxes_xyxy_pix"])
            if "part_scores" in old and tuple(old["part_scores"].shape) == tuple(scores.shape):
                scores.copy_(old["part_scores"])
            start = int(old.get("processed", 0))
            print(f"[{split}] resume at {start}/{len(samples)}")
        else:
            raise RuntimeError(f"Existing cache is incompatible: {path}. Use --overwrite.")
    elif os.path.isfile(path) and not args.overwrite:
        print(f"[{split}] output exists; skipping. Use --overwrite or --resume.")
        return

    for begin in tqdm(range(start, len(samples), args.batch_size), desc=f"GDINO {split}", ncols=120):
        end = min(begin + args.batch_size, len(samples))
        images = [load_crop(args.cub_root, samples[i], args.image_size) for i in range(begin, end)]
        text_labels = [list(PROMPTS) for _ in images]
        inputs = processor(images=images, text=text_labels, return_tensors="pt", padding=True)
        inputs = {k: v.to(args.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            amp_enabled = (not args.no_amp) and args.device.startswith("cuda")
            with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.float16 if args.device.startswith("cuda") else torch.bfloat16, enabled=amp_enabled):
                outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[(args.image_size, args.image_size)] * len(images),
        )

        for local_i, result in enumerate(results):
            global_i = begin + local_i
            per_part: List[List[Tuple[float, torch.Tensor]]] = [[] for _ in PARTS]
            for box, score, label in zip(result["boxes"].cpu(), result["scores"].cpu(), result["labels"]):
                p = part_from_label(label)
                if p >= 0:
                    per_part[p].append((float(score.item()), box.float()))

            for p, candidates in enumerate(per_part):
                candidates.sort(key=lambda item: item[0], reverse=True)
                for j, (score, box) in enumerate(candidates[: args.max_boxes_per_part]):
                    boxes[global_i, p, j] = box
                    scores[global_i, p, j] = score

        processed = end
        if processed % args.save_every < args.batch_size or processed == len(samples):
            save_state(path, relpaths, boxes, scores, processed, args.image_size, args.model_id)

    save_state(path, relpaths, boxes, scores, len(samples), args.image_size, args.model_id)
    valid = (boxes[..., 0] >= 0).any(dim=2)
    print(f"[{split}] saved {path}")
    print("[detections] " + " | ".join(f"{p}:{int(valid[:, i].sum())}/{len(samples)}" for i, p in enumerate(PARTS)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cub-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--image-size", type=int, default=518)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--box-threshold", type=float, default=0.15)
    p.add_argument("--text-threshold", type=float, default=0.15)
    p.add_argument("--max-boxes-per-part", type=int, default=4)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--split", choices=["all", "train", "test"], default="all")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.cub_root = os.path.abspath(os.path.expanduser(args.cub_root))
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    if not os.path.isdir(args.cub_root):
        raise FileNotFoundError(args.cub_root)
    if args.max_boxes_per_part < 1:
        raise ValueError("--max-boxes-per-part must be positive")

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id)
    model = model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    splits = ("train", "test") if args.split == "all" else (args.split,)
    for split in splits:
        build_split(processor, model, args, split)


if __name__ == "__main__":
    main()
