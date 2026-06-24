#!/usr/bin/env python3
"""Full Stanford Dogs part-box generation with Hugging Face Grounding DINO.

For each image:
  official Stanford Dogs union dog bbox -> dog crop
  six fixed prompts on that crop:
    head (top-1), ear (top-2), muzzle (top-1),
    body (top-1), leg (top-4), tail (top-1)
  candidates with area / dog-crop area > 0.80 are discarded.

The output is resumable and contains selected boxes in:
  - original-image xyxy coordinates
  - official-dog-crop xyxy coordinates
  - 224x224 resized dog-crop xyxy coordinates
"""

from __future__ import annotations

import argparse
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers.utils import logging as hf_logging


PARTS = (
    {"name": "head",   "prompt": "dog head.",   "topk": 1},
    {"name": "ear",    "prompt": "dog ear.",    "topk": 2},
    {"name": "muzzle", "prompt": "dog muzzle.", "topk": 1},
    {"name": "body",   "prompt": "dog torso.",  "topk": 1},
    {"name": "leg",    "prompt": "dog leg.",    "topk": 4},
    {"name": "tail",   "prompt": "dog tail.",   "topk": 1},
)


def matlab_string(value: Any) -> str:
    current = value
    while isinstance(current, np.ndarray):
        if current.size == 0:
            raise ValueError("Encountered an empty MATLAB string.")
        current = current.item() if current.size == 1 else current.flat[0]
    return current.decode("utf-8") if isinstance(current, bytes) else str(current)


def parse_split_mat(path: str) -> List[Tuple[str, int]]:
    data = loadmat(path)
    if "annotation_list" not in data or "labels" not in data:
        raise KeyError(f"Missing annotation_list or labels in {path}.")
    annotations = data["annotation_list"].squeeze()
    labels = np.asarray(data["labels"]).squeeze()
    if len(annotations) != len(labels):
        raise RuntimeError(f"Split mismatch: {len(annotations)} annotations vs {len(labels)} labels.")
    return [
        (matlab_string(annotation).replace("\\", "/"), int(np.asarray(label).item()) - 1)
        for annotation, label in zip(annotations, labels)
    ]


def parse_annotation_union_box(path: str) -> Tuple[int, int, int, int]:
    root = ET.parse(path).getroot()
    boxes: List[Tuple[int, int, int, int]] = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = int(float(bndbox.findtext("xmin", "0")))
        ymin = int(float(bndbox.findtext("ymin", "0")))
        xmax = int(float(bndbox.findtext("xmax", "0")))
        ymax = int(float(bndbox.findtext("ymax", "0")))
        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))
    if not boxes:
        raise RuntimeError(f"No valid dog bbox in {path}")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def crop_xyxy(image: Image.Image, box: Sequence[int]) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [int(v) for v in box]
    image_w, image_h = image.size
    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(x1 + 1, min(image_w, x2))
    y2 = max(y1 + 1, min(image_h, y2))
    return image.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)


def post_process(processor, outputs, input_ids, target_sizes, box_threshold, text_threshold):
    try:
        return processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
    except TypeError:
        return processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )


def empty_boxes() -> torch.Tensor:
    return torch.empty((0, 4), dtype=torch.float32)


def select_part_boxes(result: Dict[str, Any], crop_w: int, crop_h: int, topk: int, max_area_ratio: float) -> Dict[str, Any]:
    boxes = result["boxes"].detach().float().cpu().numpy().reshape(-1, 4)
    scores = result["scores"].detach().float().cpu().numpy().reshape(-1)
    labels = [str(label) for label in result.get("labels", [])]

    selected_boxes: List[List[float]] = []
    selected_scores: List[float] = []
    selected_labels: List[str] = []
    selected_area_ratios: List[float] = []
    crop_area = max(1.0, float(crop_w * crop_h))

    for index in np.argsort(-scores):
        x1, y1, x2, y2 = [float(v) for v in boxes[index]]
        x1 = max(0.0, min(float(crop_w), x1))
        y1 = max(0.0, min(float(crop_h), y1))
        x2 = max(0.0, min(float(crop_w), x2))
        y2 = max(0.0, min(float(crop_h), y2))
        width, height = x2 - x1, y2 - y1
        if width < 2.0 or height < 2.0:
            continue
        area_ratio = (width * height) / crop_area
        if area_ratio > max_area_ratio:
            continue
        selected_boxes.append([x1, y1, x2, y2])
        selected_scores.append(float(scores[index]))
        selected_labels.append(labels[index] if index < len(labels) else "")
        selected_area_ratios.append(float(area_ratio))
        if len(selected_boxes) >= topk:
            break

    boxes_crop = torch.tensor(selected_boxes, dtype=torch.float32)
    if boxes_crop.numel() == 0:
        boxes_crop = empty_boxes()
    else:
        boxes_crop = boxes_crop.reshape(-1, 4)

    return {
        "boxes_xyxy_crop": boxes_crop,
        "scores": torch.tensor(selected_scores, dtype=torch.float32),
        "labels": selected_labels,
        "area_ratios": torch.tensor(selected_area_ratios, dtype=torch.float32),
    }


def boxes_to_original(boxes_crop: torch.Tensor, dog_box: Sequence[int]) -> torch.Tensor:
    if boxes_crop.numel() == 0:
        return empty_boxes()
    offset = torch.tensor([dog_box[0], dog_box[1], dog_box[0], dog_box[1]], dtype=torch.float32)
    return boxes_crop + offset


def boxes_to_resize(boxes_crop: torch.Tensor, crop_w: int, crop_h: int, resize_size: int) -> torch.Tensor:
    if boxes_crop.numel() == 0:
        return empty_boxes()
    scale = torch.tensor(
        [
            resize_size / max(1.0, float(crop_w)),
            resize_size / max(1.0, float(crop_h)),
            resize_size / max(1.0, float(crop_w)),
            resize_size / max(1.0, float(crop_h)),
        ],
        dtype=torch.float32,
    )
    return boxes_crop * scale


def safe_torch_load(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_save(payload: Dict[str, Any], output_path: str) -> None:
    tmp_path = output_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, output_path)


def build_payload(records: Dict[str, Dict[str, Any]], args, split: str, expected_images: int) -> Dict[str, Any]:
    selected_count_by_part = {}
    no_box_count_by_part = {}
    for part in PARTS:
        name = part["name"]
        selected_count_by_part[name] = int(sum(record["parts"][name]["boxes_xyxy_crop"].shape[0] for record in records.values()))
        no_box_count_by_part[name] = int(sum(record["parts"][name]["boxes_xyxy_crop"].shape[0] == 0 for record in records.values()))

    return {
        "meta": {
            "dataset": "Stanford Dogs",
            "split": split,
            "detector": args.model_id,
            "global_crop": "official Stanford Dogs union dog bbox",
            "parts": [{"name": p["name"], "prompt": p["prompt"], "topk": p["topk"]} for p in PARTS],
            "box_threshold": float(args.box_threshold),
            "text_threshold": float(args.text_threshold),
            "max_area_ratio": float(args.max_area_ratio),
            "resized_coordinate_size": int(args.resize_size),
            "expected_images": int(expected_images),
            "processed_images": int(len(records)),
            "selected_box_count_by_part": selected_count_by_part,
            "no_box_count_by_part": no_box_count_by_part,
            "bbox_coordinate_system": {
                "boxes_xyxy_original": "original image pixels",
                "boxes_xyxy_crop": "official dog crop pixels",
                "boxes_xyxy_resize": "official dog crop directly resized to resize_size x resize_size",
            },
        },
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Full Stanford Dogs part bbox extraction with HF Grounding DINO")
    parser.add_argument("--dogs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--images-per-batch", type=int, default=4)
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--max-area-ratio", type=float, default=0.80)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.dogs_root = str(Path(args.dogs_root).expanduser().resolve())
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    required = [
        Path(args.dogs_root) / "Images",
        Path(args.dogs_root) / "Annotation",
        Path(args.dogs_root) / f"{args.split}_list.mat",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n  - " + "\n  - ".join(missing))
    if args.images_per_batch < 1:
        raise ValueError("--images-per-batch must be >= 1")
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")
    if not (0.0 <= args.box_threshold <= 1.0 and 0.0 <= args.text_threshold <= 1.0):
        raise ValueError("thresholds must be in [0, 1]")
    if not (0.0 < args.max_area_ratio <= 1.0):
        raise ValueError("--max-area-ratio must be in (0, 1]")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_part_boxes_hf_gdino.pt"

    records: Dict[str, Dict[str, Any]] = {}
    if args.resume and output_path.is_file():
        old_payload = safe_torch_load(str(output_path))
        if isinstance(old_payload, dict) and isinstance(old_payload.get("records"), dict):
            records = old_payload["records"]
            print(f"[Resume] loaded {len(records)} completed {args.split} records.")

    hf_logging.set_verbosity_error()
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print(f"[HF] loading {args.model_id} on {args.device}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(args.device).eval()

    dogs_root = Path(args.dogs_root)
    images_root = dogs_root / "Images"
    annotations_root = dogs_root / "Annotation"
    split_items = parse_split_mat(str(dogs_root / f"{args.split}_list.mat"))
    pending = [(rel, label) for rel, label in split_items if rel not in records]

    print(
        f"[Run] split={args.split}, total={len(split_items)}, remaining={len(pending)}, "
        f"images_per_batch={args.images_per_batch}, GDINO_pairs_per_forward={args.images_per_batch * len(PARTS)}"
    )

    start_time = time.time()
    processed_since_save = 0
    progress = tqdm(range(0, len(pending), args.images_per_batch), desc=f"HF GDINO {args.split}", dynamic_ncols=True)

    for start in progress:
        group = pending[start : start + args.images_per_batch]
        items: List[Dict[str, Any]] = []
        jobs: List[Dict[str, Any]] = []

        for rel_annotation, label in group:
            image_path = images_root / f"{rel_annotation}.jpg"
            annotation_path = annotations_root / rel_annotation
            with Image.open(image_path) as source:
                full_image = source.convert("RGB")
            image_w, image_h = full_image.size
            dog_box = parse_annotation_union_box(str(annotation_path))
            dog_crop, dog_box = crop_xyxy(full_image, dog_box)
            crop_w, crop_h = dog_crop.size

            item = {
                "rel_annotation": rel_annotation,
                "label": int(label),
                "image_size_wh": [int(image_w), int(image_h)],
                "dog_box": dog_box,
                "crop_w": crop_w,
                "crop_h": crop_h,
                "parts": {},
            }
            items.append(item)
            for part in PARTS:
                jobs.append({
                    "item": item,
                    "part": part,
                    "image": dog_crop,
                    "prompt": part["prompt"],
                    "target_size_hw": (crop_h, crop_w),
                })

        # Each image is replicated once for each part prompt; all pairs are evaluated together.
        batch_inputs = processor(
            images=[job["image"] for job in jobs],
            text=[job["prompt"] for job in jobs],
            return_tensors="pt",
            padding=True,
        ).to(args.device)

        with torch.inference_mode():
            outputs = model(**batch_inputs)

        results = post_process(
            processor,
            outputs,
            batch_inputs.input_ids,
            [job["target_size_hw"] for job in jobs],
            args.box_threshold,
            args.text_threshold,
        )
        if len(results) != len(jobs):
            raise RuntimeError(f"Post-processing returned {len(results)} results for {len(jobs)} jobs.")

        for job, result in zip(jobs, results):
            item = job["item"]
            part = job["part"]
            selected = select_part_boxes(
                result,
                item["crop_w"],
                item["crop_h"],
                part["topk"],
                args.max_area_ratio,
            )
            selected["boxes_xyxy_original"] = boxes_to_original(selected["boxes_xyxy_crop"], item["dog_box"])
            selected["boxes_xyxy_resize"] = boxes_to_resize(
                selected["boxes_xyxy_crop"],
                item["crop_w"],
                item["crop_h"],
                args.resize_size,
            )
            item["parts"][part["name"]] = selected

        for item in items:
            records[item["rel_annotation"]] = {
                "label": item["label"],
                "image_size_wh": item["image_size_wh"],
                "dog_bbox_xyxy_original": torch.tensor(item["dog_box"], dtype=torch.float32),
                "dog_crop_size_wh": [item["crop_w"], item["crop_h"]],
                "parts": item["parts"],
            }

        processed_since_save += len(items)
        if processed_since_save >= args.save_every:
            atomic_save(build_payload(records, args, args.split, len(split_items)), str(output_path))
            processed_since_save = 0

        progress.set_postfix(done=len(records), hours=f"{(time.time() - start_time) / 3600.0:.2f}")

    payload = build_payload(records, args, args.split, len(split_items))
    atomic_save(payload, str(output_path))
    print(f"[Done] {args.split}: saved {len(records)} records to {output_path}")
    print("[Boxes]", payload["meta"]["selected_box_count_by_part"])
    print("[No box]", payload["meta"]["no_box_count_by_part"])


if __name__ == "__main__":
    main()
