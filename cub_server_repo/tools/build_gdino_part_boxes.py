#!/usr/bin/env python3
"""Generate GroundingDINO part boxes for CUB with one prompt per part.

Compatible with the current test1/cub_server_repo pipeline.

Geometry:
    CUB original image
    -> crop by official CUB ground-truth bird bbox
    -> direct warp/resize to --image-size square
    -> GroundingDINO

Prompting:
    Each part is inferred in a separate GroundingDINO forward pass.

Saved layout:
    part_boxes_xyxy_pix: (N, 6, 2, 4)
    part_scores:          (N, 6, 2)

The second box slot is used only by "wing". For all other parts, the second
slot stays [-1, -1, -1, -1] with score -1.

By default the output filenames remain:
    train_part_boxes_gtbbox_warp518.pt
    test_part_boxes_gtbbox_warp518.pt

This naming is intentionally preserved because the current training script in
this repository expects those filenames. The true coordinate size is recorded
in the saved ``img_size`` field, so using --image-size 224 remains correct.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


PARTS: Tuple[str, ...] = (
    "beak",
    "head",
    "wing",
    "body",
    "tail",
    "feet",
)

PROMPTS: Dict[str, str] = {
    "beak": "bird beak.",
    "head": "bird head.",
    "wing": "bird wing.",
    "body": "bird body.",
    "tail": "bird tail.",
    "feet": "bird foot.",
}

# Uniform storage uses two slots. Only wing may fill both slots.
PART_MAX_BOXES: Dict[str, int] = {
    "beak": 1,
    "head": 1,
    "wing": 2,
    "body": 1,
    "tail": 1,
    "feet": 1,
}
MAX_BOX_SLOTS = max(PART_MAX_BOXES.values())  # 2


# -----------------------------------------------------------------------------
# CUB metadata
# -----------------------------------------------------------------------------
def read_kv_text(path: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(maxsplit=1)
            out[int(key)] = value
    return out


def read_kv_int(path: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            key, value = line.split()
            out[int(key)] = int(value)
    return out


def read_bboxes(path: str) -> Dict[int, Tuple[float, float, float, float]]:
    out: Dict[int, Tuple[float, float, float, float]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            values = line.strip().split()
            if not values:
                continue
            out[int(values[0])] = tuple(float(x) for x in values[1:5])
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
    samples: List[Sample] = []
    for image_id in sorted(paths):
        if (split_map[image_id] == 1) == want_train:
            samples.append(Sample(image_id, paths[image_id], bboxes[image_id]))
    return samples


def crop_bbox(
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
) -> Image.Image:
    x, y, width_box, height_box = bbox
    width_image, height_image = image.size

    x1 = max(0, min(width_image - 1, int(np.floor(x))))
    y1 = max(0, min(height_image - 1, int(np.floor(y))))
    x2 = max(x1 + 1, min(width_image, int(np.ceil(x + width_box))))
    y2 = max(y1 + 1, min(height_image, int(np.ceil(y + height_box))))
    return image.crop((x1, y1, x2, y2))


def resolve_processed_path(processed_root: str, relpath: str) -> str:
    """Support both <root>/images/<relpath> and <root>/<relpath>."""
    candidate_with_images = os.path.join(processed_root, "images", relpath)
    if os.path.isfile(candidate_with_images):
        return candidate_with_images

    candidate_direct = os.path.join(processed_root, relpath)
    if os.path.isfile(candidate_direct):
        return candidate_direct

    raise FileNotFoundError(
        "Processed image not found. Checked:\n"
        f"  {candidate_with_images}\n"
        f"  {candidate_direct}"
    )


def load_image(
    cub_root: str,
    sample: Sample,
    image_size: int,
    processed_image_root: Optional[str],
) -> Image.Image:
    if processed_image_root:
        path = resolve_processed_path(processed_image_root, sample.relpath)
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (image_size, image_size):
                image = image.resize(
                    (image_size, image_size),
                    Image.Resampling.BICUBIC,
                )
            return image.copy()

    path = os.path.join(cub_root, "images", sample.relpath)
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = crop_bbox(image, sample.bbox)
        image = image.resize(
            (image_size, image_size),
            Image.Resampling.BICUBIC,
        )
        return image.copy()


# -----------------------------------------------------------------------------
# Detection helpers
# -----------------------------------------------------------------------------
def box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """IoU between one xyxy box and M xyxy boxes, all on CPU."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)

    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter_w = (x2 - x1).clamp_min(0)
    inter_h = (y2 - y1).clamp_min(0)
    intersection = inter_w * inter_h

    area_box = (box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)
    areas = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp_min(0)
    union = area_box + areas - intersection
    return intersection / union.clamp_min(1e-8)


def greedy_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Pure-PyTorch CPU NMS to avoid a torchvision runtime dependency."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    order = torch.argsort(scores, descending=True)
    kept: List[int] = []

    while order.numel() > 0:
        current = int(order[0].item())
        kept.append(current)
        if order.numel() == 1:
            break

        remaining = order[1:]
        ious = box_iou_one_to_many(boxes[current], boxes[remaining])
        order = remaining[ious <= iou_threshold]

    return torch.tensor(kept, dtype=torch.long)


def move_inputs_to_device(inputs, device: str):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def post_process_grounded(
    processor,
    outputs,
    input_ids: torch.Tensor,
    box_threshold: float,
    text_threshold: float,
    target_sizes: Sequence[Tuple[int, int]],
):
    """Compatibility wrapper for Transformers 4.40.x and newer versions."""
    try:
        return processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=list(target_sizes),
        )
    except TypeError as first_error:
        # Some nearby Transformers revisions used ``threshold``.
        try:
            return processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=list(target_sizes),
            )
        except TypeError:
            raise first_error


def detect_one_part(
    processor,
    model,
    images: Sequence[Image.Image],
    part_name: str,
    args: argparse.Namespace,
):
    """Run one independent GroundingDINO forward for one named part."""
    prompt = PROMPTS[part_name]
    text_batch = [prompt] * len(images)

    inputs = processor(
        images=list(images),
        text=text_batch,
        return_tensors="pt",
        padding=True,
    )
    inputs = move_inputs_to_device(inputs, args.device)

    amp_enabled = (not args.no_amp) and args.device.startswith("cuda")
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda" if args.device.startswith("cuda") else "cpu",
            dtype=torch.float16 if args.device.startswith("cuda") else torch.bfloat16,
            enabled=amp_enabled,
        ):
            outputs = model(**inputs)

    return post_process_grounded(
        processor=processor,
        outputs=outputs,
        input_ids=inputs["input_ids"],
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=[(args.image_size, args.image_size)] * len(images),
    )


def select_boxes_for_part(
    result: Dict[str, torch.Tensor],
    part_name: str,
    nms_iou_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    boxes = result["boxes"].detach().float().cpu()
    scores = result["scores"].detach().float().cpu()

    if boxes.numel() == 0:
        return (
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
        )

    valid = (
        torch.isfinite(boxes).all(dim=1)
        & torch.isfinite(scores)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    boxes = boxes[valid]
    scores = scores[valid]
    if boxes.numel() == 0:
        return boxes.reshape(0, 4), scores.reshape(0)

    keep = greedy_nms(boxes, scores, iou_threshold=nms_iou_threshold)
    keep = keep[: PART_MAX_BOXES[part_name]]
    return boxes[keep], scores[keep]


# -----------------------------------------------------------------------------
# Cache I/O
# -----------------------------------------------------------------------------
def output_path(output_dir: str, split: str) -> str:
    # Keep the current trainer-compatible name even when img_size is 224.
    return os.path.join(output_dir, f"{split}_part_boxes_gtbbox_warp518.pt")


def save_state(
    path: str,
    relpaths: Sequence[str],
    boxes: torch.Tensor,
    scores: torch.Tensor,
    processed: int,
    image_size: int,
    model_id: str,
    processed_image_root: Optional[str],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"

    torch.save(
        {
            "relpaths": list(relpaths),
            "parts": list(PARTS),
            "prompts": [PROMPTS[part] for part in PARTS],
            "part_max_boxes": dict(PART_MAX_BOXES),
            "part_boxes_xyxy_pix": boxes,
            "part_scores": scores,
            "processed": int(processed),
            "img_size": int(image_size),
            "model_id": model_id,
            "separate_prompt_per_part": True,
            "processed_image_root": processed_image_root,
            "crop": (
                "pre-exported CUB GT-bbox crop, warped to square"
                if processed_image_root
                else "CUB ground-truth bbox, warped to square"
            ),
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def validate_resume_cache(
    old: Dict,
    relpaths: Sequence[str],
    boxes: torch.Tensor,
    scores: torch.Tensor,
    args: argparse.Namespace,
    path: str,
) -> int:
    expected_boxes_shape = tuple(boxes.shape)
    expected_scores_shape = tuple(scores.shape)

    conditions = [
        old.get("relpaths") == list(relpaths),
        tuple(old.get("part_boxes_xyxy_pix", torch.empty(0)).shape)
        == expected_boxes_shape,
        tuple(old.get("part_scores", torch.empty(0)).shape)
        == expected_scores_shape,
        int(old.get("img_size", -1)) == int(args.image_size),
        bool(old.get("separate_prompt_per_part", False)),
    ]
    if not all(conditions):
        raise RuntimeError(
            f"Existing cache is incompatible: {path}. "
            "Delete it or rerun with --overwrite."
        )

    boxes.copy_(old["part_boxes_xyxy_pix"])
    scores.copy_(old["part_scores"])
    return int(old.get("processed", 0))


# -----------------------------------------------------------------------------
# Main split builder
# -----------------------------------------------------------------------------
def build_split(processor, model, args: argparse.Namespace, split: str) -> None:
    samples = build_samples(args.cub_root, split)
    relpaths = [sample.relpath for sample in samples]
    path = output_path(args.output_dir, split)

    boxes = torch.full(
        (len(samples), len(PARTS), MAX_BOX_SLOTS, 4),
        -1.0,
        dtype=torch.float32,
    )
    scores = torch.full(
        (len(samples), len(PARTS), MAX_BOX_SLOTS),
        -1.0,
        dtype=torch.float32,
    )

    start = 0
    if os.path.isfile(path) and args.resume and not args.overwrite:
        old = torch.load(path, map_location="cpu")
        start = validate_resume_cache(old, relpaths, boxes, scores, args, path)
        print(f"[{split}] resume at {start}/{len(samples)}")
    elif os.path.isfile(path) and not args.overwrite:
        print(f"[{split}] output exists; skipping: {path}")
        print("Use --overwrite to rebuild or --resume to continue.")
        return

    progress = tqdm(
        range(start, len(samples), args.batch_size),
        desc=f"GDINO {split}",
        ncols=140,
    )

    for begin in progress:
        end = min(begin + args.batch_size, len(samples))
        images = [
            load_image(
                cub_root=args.cub_root,
                sample=samples[index],
                image_size=args.image_size,
                processed_image_root=args.processed_image_root,
            )
            for index in range(begin, end)
        ]

        # Six independent forward passes per image batch, one per part.
        for part_index, part_name in enumerate(PARTS):
            results = detect_one_part(
                processor=processor,
                model=model,
                images=images,
                part_name=part_name,
                args=args,
            )

            for local_index, result in enumerate(results):
                global_index = begin + local_index
                selected_boxes, selected_scores = select_boxes_for_part(
                    result=result,
                    part_name=part_name,
                    nms_iou_threshold=args.nms_iou_threshold,
                )

                count = selected_boxes.shape[0]
                if count == 0:
                    continue

                boxes[global_index, part_index, :count] = selected_boxes
                scores[global_index, part_index, :count] = selected_scores

        processed = end
        if (
            processed % args.save_every < args.batch_size
            or processed == len(samples)
        ):
            save_state(
                path=path,
                relpaths=relpaths,
                boxes=boxes,
                scores=scores,
                processed=processed,
                image_size=args.image_size,
                model_id=args.model_id,
                processed_image_root=args.processed_image_root,
            )

        valid_images = (boxes[:processed, ..., 0] >= 0).any(dim=2)
        progress.set_postfix(
            {
                part: int(valid_images[:, part_index].sum().item())
                for part_index, part in enumerate(PARTS)
            }
        )

    save_state(
        path=path,
        relpaths=relpaths,
        boxes=boxes,
        scores=scores,
        processed=len(samples),
        image_size=args.image_size,
        model_id=args.model_id,
        processed_image_root=args.processed_image_root,
    )

    valid_images = (boxes[..., 0] >= 0).any(dim=2)
    valid_box_counts = (boxes[..., 0] >= 0).sum(dim=(0, 2))

    print(f"[{split}] saved: {path}")
    print(f"[{split}] box tensor shape: {tuple(boxes.shape)}")
    print(
        "[detected images] "
        + " | ".join(
            f"{part}:{int(valid_images[:, index].sum())}/{len(samples)}"
            for index, part in enumerate(PARTS)
        )
    )
    print(
        "[total boxes] "
        + " | ".join(
            f"{part}:{int(valid_box_counts[index])}"
            for index, part in enumerate(PARTS)
        )
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CUB GroundingDINO part boxes using one independent "
            "prompt/forward pass per part. Wing keeps two boxes; every other "
            "part keeps one."
        )
    )
    parser.add_argument("--cub-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--processed-image-root",
        default=None,
        help=(
            "Optional root of pre-exported GT-bbox-resized images. Supports "
            "<root>/images/<CUB relpath> or <root>/<CUB relpath>. When omitted, "
            "the script crops the original CUB image by its GT bbox and directly "
            "resizes it to --image-size."
        ),
    )
    parser.add_argument(
        "--model-id",
        default="IDEA-Research/grounding-dino-tiny",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.50)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--split",
        choices=["all", "train", "test"],
        default="all",
    )
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    required = [
        "images.txt",
        "train_test_split.txt",
        "bounding_boxes.txt",
        "images",
    ]
    missing = [
        os.path.join(args.cub_root, name)
        for name in required
        if not os.path.exists(os.path.join(args.cub_root, name))
    ]
    if missing:
        raise FileNotFoundError(
            "Missing CUB files/directories:\n" + "\n".join(missing)
        )

    if args.processed_image_root and not os.path.isdir(args.processed_image_root):
        raise FileNotFoundError(args.processed_image_root)

    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")
    if not 0.0 <= args.nms_iou_threshold <= 1.0:
        raise ValueError("--nms-iou-threshold must be in [0, 1]")


def main() -> None:
    args = parse_args()
    args.cub_root = os.path.abspath(os.path.expanduser(args.cub_root))
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    if args.processed_image_root:
        args.processed_image_root = os.path.abspath(
            os.path.expanduser(args.processed_image_root)
        )

    validate_paths(args)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[model] {args.model_id}")
    print(f"[device] {args.device}; AMP={not args.no_amp}")
    print(f"[image size] {args.image_size}x{args.image_size}")
    print("[prompting] separate forward per part")
    print(f"[part limits] {PART_MAX_BOXES}")
    if args.processed_image_root:
        print(f"[images] pre-exported: {args.processed_image_root}")
    else:
        print("[images] original CUB -> GT bbox crop -> direct square resize")

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id)
    model = model.to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    splits = ("train", "test") if args.split == "all" else (args.split,)
    for split in splits:
        build_split(processor, model, args, split)


if __name__ == "__main__":
    main()
