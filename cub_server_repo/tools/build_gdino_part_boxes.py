#!/usr/bin/env python3
"""Build GroundingDINO part-box caches for CUB-200-2011.

This version is designed for ``cub_server_repo`` on the ``test1`` branch.

Key behavior
------------
1. Read CUB ground-truth part visibility from ``parts/part_locs.txt``.
2. Skip GroundingDINO inference for a coarse part when that part is not visible.
3. Run every coarse part with a separate text prompt; prompts never compete in one
   GroundingDINO call.
4. Save at most two boxes for ``wing`` and one box for every other part.
5. Preserve the cache keys and output filenames expected by
   ``train_cub_semantic_part_additive_proto_server.py``.

Output tensors
--------------
``part_boxes_xyxy_pix`` has shape ``(N, 6, 2, 4)``.
``part_scores`` has shape ``(N, 6, 2)``.

The second slot is always invalid for non-wing parts. Invalid boxes/scores are
filled with -1.
"""

from __future__ import annotations

import argparse
import inspect
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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

PROMPTS: Mapping[str, str] = {
    "beak": "bird beak.",
    "head": "bird head.",
    "wing": "bird wing.",
    "body": "bird body.",
    "tail": "bird tail.",
    "feet": "bird foot.",
}

# The cache uses two slots for every part so that it remains one dense tensor.
# Only wing is allowed to occupy the second slot.
PART_MAX_BOXES: Mapping[str, int] = {
    "beak": 1,
    "head": 1,
    "wing": 2,
    "body": 1,
    "tail": 1,
    "feet": 1,
}
MAX_BOX_SLOTS = 2

# CUB's 15 original keypoint names are read from parts/parts.txt. A coarse part
# is scanned if at least one of its source keypoints is marked visible.
COARSE_VISIBILITY_SOURCES: Mapping[str, Tuple[str, ...]] = {
    "beak": ("beak",),
    "head": (
        "crown",
        "forehead",
        "left eye",
        "right eye",
        "nape",
        "throat",
    ),
    "wing": ("left wing", "right wing"),
    "body": ("back", "belly", "breast"),
    "tail": ("tail",),
    "feet": ("left leg", "right leg"),
}


@dataclass(frozen=True)
class Sample:
    image_id: int
    relpath: str
    bbox: Tuple[float, float, float, float]
    part_visible: Tuple[bool, ...]
    part_visible_count: Tuple[int, ...]


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


def normalize_part_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").split())


def read_cub_part_names(cub_root: str) -> Dict[int, str]:
    path = os.path.join(cub_root, "parts", "parts.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing CUB part names: {path}")

    raw = read_kv_text(path)
    return {part_id: normalize_part_name(name) for part_id, name in raw.items()}


def read_cub_part_visibility(
    cub_root: str,
    part_names: Mapping[int, str],
) -> Dict[int, Dict[str, bool]]:
    path = os.path.join(cub_root, "parts", "part_locs.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing CUB part locations: {path}")

    visibility: Dict[int, Dict[str, bool]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            values = line.strip().split()
            if not values:
                continue
            if len(values) != 5:
                raise ValueError(f"Unexpected part_locs.txt row: {line.rstrip()}")

            image_id = int(values[0])
            part_id = int(values[1])
            visible = int(float(values[4])) > 0
            if part_id not in part_names:
                raise KeyError(f"Unknown CUB part id {part_id} in {path}")

            image_visibility = visibility.setdefault(image_id, {})
            image_visibility[part_names[part_id]] = visible

    return visibility


def coarse_visibility_for_image(
    fine_visibility: Mapping[str, bool],
) -> Tuple[Tuple[bool, ...], Tuple[int, ...]]:
    visible_flags: List[bool] = []
    visible_counts: List[int] = []

    for part in PARTS:
        sources = COARSE_VISIBILITY_SOURCES[part]
        count = sum(bool(fine_visibility.get(source, False)) for source in sources)
        visible_flags.append(count > 0)
        visible_counts.append(count)

    return tuple(visible_flags), tuple(visible_counts)


def build_samples(cub_root: str, split: str) -> List[Sample]:
    paths = read_kv_text(os.path.join(cub_root, "images.txt"))
    split_map = read_kv_int(os.path.join(cub_root, "train_test_split.txt"))
    bboxes = read_bboxes(os.path.join(cub_root, "bounding_boxes.txt"))
    part_names = read_cub_part_names(cub_root)
    visibility = read_cub_part_visibility(cub_root, part_names)

    want_train = split == "train"
    samples: List[Sample] = []

    for image_id in sorted(paths):
        if (split_map[image_id] == 1) != want_train:
            continue
        if image_id not in visibility:
            raise KeyError(f"Missing part visibility for CUB image id {image_id}")

        part_visible, part_visible_count = coarse_visibility_for_image(
            visibility[image_id]
        )
        samples.append(
            Sample(
                image_id=image_id,
                relpath=paths[image_id],
                bbox=bboxes[image_id],
                part_visible=part_visible,
                part_visible_count=part_visible_count,
            )
        )

    return samples


def crop_bbox(
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
) -> Image.Image:
    x, y, width, height = bbox
    image_width, image_height = image.size

    x1 = max(0, min(image_width - 1, int(np.floor(x))))
    y1 = max(0, min(image_height - 1, int(np.floor(y))))
    x2 = max(x1 + 1, min(image_width, int(np.ceil(x + width))))
    y2 = max(y1 + 1, min(image_height, int(np.ceil(y + height))))

    return image.crop((x1, y1, x2, y2))


def resolve_processed_images_dir(processed_image_root: str) -> str:
    root = os.path.abspath(os.path.expanduser(processed_image_root))
    nested = os.path.join(root, "images")
    return nested if os.path.isdir(nested) else root


def load_model_image(
    cub_root: str,
    sample: Sample,
    image_size: int,
    processed_images_dir: Optional[str],
) -> Image.Image:
    if processed_images_dir is not None:
        path = os.path.join(processed_images_dir, sample.relpath)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing processed image: {path}")
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (image_size, image_size):
                image = image.resize(
                    (image_size, image_size),
                    Image.Resampling.BICUBIC,
                )
            return image.copy()

    path = os.path.join(cub_root, "images", sample.relpath)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing CUB image: {path}")

    with Image.open(path) as image:
        image = image.convert("RGB")
        image = crop_bbox(image, sample.bbox)
        image = image.resize(
            (image_size, image_size),
            Image.Resampling.BICUBIC,
        )
        return image.copy()


def output_path(output_dir: str, split: str) -> str:
    # Keep the filename used by the current training script. The real coordinate
    # size is always stored in the checkpoint's ``img_size`` field.
    return os.path.join(
        output_dir,
        f"{split}_part_boxes_gtbbox_warp518.pt",
    )


def box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    intersection = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    area_one = (box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)
    area_many = (
        (boxes[:, 2] - boxes[:, 0]).clamp_min(0)
        * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    )
    union = area_one + area_many - intersection
    return intersection / union.clamp_min(1e-9)


def nms_xyxy(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Pure-PyTorch NMS to avoid a torchvision dependency."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    order = torch.argsort(scores, descending=True)
    keep: List[int] = []

    while order.numel() > 0:
        current = int(order[0].item())
        keep.append(current)
        if order.numel() == 1:
            break

        remaining = order[1:]
        ious = box_iou_one_to_many(boxes[current], boxes[remaining])
        order = remaining[ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)


def post_process_grounded_detection(
    processor,
    outputs,
    input_ids: torch.Tensor,
    target_sizes: Sequence[Tuple[int, int]],
    box_threshold: float,
    text_threshold: float,
):
    """Support both Transformers 4.40.x and newer argument names."""
    method = processor.post_process_grounded_object_detection
    parameters = inspect.signature(method).parameters

    kwargs = {
        "text_threshold": text_threshold,
        "target_sizes": list(target_sizes),
    }
    if "box_threshold" in parameters:
        kwargs["box_threshold"] = box_threshold
    else:
        kwargs["threshold"] = box_threshold

    return method(outputs, input_ids, **kwargs)


def save_state(
    path: str,
    relpaths: Sequence[str],
    boxes: torch.Tensor,
    scores: torch.Tensor,
    visible_gt: torch.Tensor,
    visible_count_gt: torch.Tensor,
    scan_counts: torch.Tensor,
    processed: int,
    image_size: int,
    model_id: str,
    processed_image_root: Optional[str],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = path + ".tmp"

    torch.save(
        {
            "relpaths": list(relpaths),
            "parts": list(PARTS),
            "prompts": [PROMPTS[part] for part in PARTS],
            "part_max_boxes": [PART_MAX_BOXES[part] for part in PARTS],
            "part_boxes_xyxy_pix": boxes,
            "part_scores": scores,
            "part_visible_gt": visible_gt,
            "part_visible_count_gt": visible_count_gt,
            "scan_counts": scan_counts,
            "processed": int(processed),
            "img_size": int(image_size),
            "model_id": model_id,
            "crop": "CUB ground-truth bbox, directly resized to square",
            "processed_image_root": processed_image_root,
            "separate_prompt_per_part": True,
            "visibility_gated_by_cub": True,
            "coarse_visibility_sources": {
                part: list(sources)
                for part, sources in COARSE_VISIBILITY_SOURCES.items()
            },
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def compatible_resume_cache(
    old: Mapping[str, object],
    relpaths: Sequence[str],
    expected_box_shape: Tuple[int, ...],
    expected_score_shape: Tuple[int, ...],
) -> bool:
    old_boxes = old.get("part_boxes_xyxy_pix")
    old_scores = old.get("part_scores")

    return (
        old.get("relpaths") == list(relpaths)
        and isinstance(old_boxes, torch.Tensor)
        and tuple(old_boxes.shape) == expected_box_shape
        and isinstance(old_scores, torch.Tensor)
        and tuple(old_scores.shape) == expected_score_shape
        and bool(old.get("separate_prompt_per_part", False))
        and bool(old.get("visibility_gated_by_cub", False))
    )


def build_split(
    processor,
    model,
    args: argparse.Namespace,
    split: str,
) -> None:
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
    visible_gt = torch.tensor(
        [sample.part_visible for sample in samples],
        dtype=torch.bool,
    )
    visible_count_gt = torch.tensor(
        [sample.part_visible_count for sample in samples],
        dtype=torch.int16,
    )
    scan_counts = torch.zeros(len(PARTS), dtype=torch.int64)

    start = 0
    if os.path.isfile(path):
        if args.overwrite:
            print(f"[{split}] overwriting existing cache: {path}")
        elif args.resume:
            old = torch.load(path, map_location="cpu")
            if not compatible_resume_cache(
                old,
                relpaths,
                tuple(boxes.shape),
                tuple(scores.shape),
            ):
                raise RuntimeError(
                    f"Existing cache is incompatible: {path}\n"
                    "Use --overwrite to create the new visibility-gated cache."
                )

            boxes.copy_(old["part_boxes_xyxy_pix"])
            scores.copy_(old["part_scores"])
            if isinstance(old.get("scan_counts"), torch.Tensor):
                scan_counts.copy_(old["scan_counts"])
            start = int(old.get("processed", 0))
            start = max(0, min(start, len(samples)))
            print(f"[{split}] resume at {start}/{len(samples)}")
        else:
            print(
                f"[{split}] output exists; skipping: {path}\n"
                "Use --overwrite or --resume."
            )
            return

    processed_images_dir: Optional[str] = None
    if args.processed_image_root:
        processed_images_dir = resolve_processed_images_dir(
            args.processed_image_root
        )
        if not os.path.isdir(processed_images_dir):
            raise FileNotFoundError(
                f"Processed image directory does not exist: "
                f"{processed_images_dir}"
            )

    progress = tqdm(
        range(start, len(samples), args.batch_size),
        desc=f"GDINO {split}",
        ncols=120,
    )
    last_saved = start

    for begin in progress:
        end = min(begin + args.batch_size, len(samples))
        batch_samples = samples[begin:end]
        images = [
            load_model_image(
                cub_root=args.cub_root,
                sample=sample,
                image_size=args.image_size,
                processed_images_dir=processed_images_dir,
            )
            for sample in batch_samples
        ]

        # Each coarse part is run in a separate GroundingDINO forward pass.
        for part_index, part in enumerate(PARTS):
            active_local_indices = [
                local_index
                for local_index, sample in enumerate(batch_samples)
                if sample.part_visible[part_index]
            ]
            if not active_local_indices:
                continue

            active_images = [images[index] for index in active_local_indices]
            text_prompts = [PROMPTS[part]] * len(active_images)
            scan_counts[part_index] += len(active_images)

            inputs = processor(
                images=active_images,
                text=text_prompts,
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                key: value.to(args.device)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in inputs.items()
            }

            amp_enabled = (
                not args.no_amp
                and args.device.startswith("cuda")
            )
            device_type = "cuda" if args.device.startswith("cuda") else "cpu"
            autocast_dtype = (
                torch.float16 if device_type == "cuda" else torch.bfloat16
            )

            with torch.inference_mode():
                with torch.autocast(
                    device_type=device_type,
                    dtype=autocast_dtype,
                    enabled=amp_enabled,
                ):
                    outputs = model(**inputs)

            results = post_process_grounded_detection(
                processor=processor,
                outputs=outputs,
                input_ids=inputs["input_ids"],
                target_sizes=[
                    (args.image_size, args.image_size)
                    for _ in active_images
                ],
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )

            for active_index, result in enumerate(results):
                local_index = active_local_indices[active_index]
                global_index = begin + local_index
                sample = batch_samples[local_index]

                detected_boxes = result["boxes"].detach().float().cpu()
                detected_scores = result["scores"].detach().float().cpu()
                if detected_boxes.numel() == 0:
                    continue

                valid_geometry = (
                    (detected_boxes[:, 2] > detected_boxes[:, 0])
                    & (detected_boxes[:, 3] > detected_boxes[:, 1])
                )
                detected_boxes = detected_boxes[valid_geometry]
                detected_scores = detected_scores[valid_geometry]
                if detected_boxes.numel() == 0:
                    continue

                keep = nms_xyxy(
                    detected_boxes,
                    detected_scores,
                    args.nms_iou_threshold,
                )
                detected_boxes = detected_boxes[keep]
                detected_scores = detected_scores[keep]

                order = torch.argsort(detected_scores, descending=True)
                detected_boxes = detected_boxes[order]
                detected_scores = detected_scores[order]

                max_boxes = PART_MAX_BOXES[part]
                if part == "wing":
                    # If only one of the two CUB wing keypoints is visible, keep
                    # only one detector box. If both are visible, allow two.
                    max_boxes = min(
                        max_boxes,
                        max(1, sample.part_visible_count[part_index]),
                    )

                count = min(max_boxes, detected_boxes.shape[0])
                if count > 0:
                    boxes[
                        global_index,
                        part_index,
                        :count,
                    ] = detected_boxes[:count]
                    scores[
                        global_index,
                        part_index,
                        :count,
                    ] = detected_scores[:count]

        if (end - last_saved) >= args.save_every or end == len(samples):
            save_state(
                path=path,
                relpaths=relpaths,
                boxes=boxes,
                scores=scores,
                visible_gt=visible_gt,
                visible_count_gt=visible_count_gt,
                scan_counts=scan_counts,
                processed=end,
                image_size=args.image_size,
                model_id=args.model_id,
                processed_image_root=(
                    os.path.abspath(os.path.expanduser(args.processed_image_root))
                    if args.processed_image_root
                    else None
                ),
            )
            last_saved = end

    save_state(
        path=path,
        relpaths=relpaths,
        boxes=boxes,
        scores=scores,
        visible_gt=visible_gt,
        visible_count_gt=visible_count_gt,
        scan_counts=scan_counts,
        processed=len(samples),
        image_size=args.image_size,
        model_id=args.model_id,
        processed_image_root=(
            os.path.abspath(os.path.expanduser(args.processed_image_root))
            if args.processed_image_root
            else None
        ),
    )

    detected = (boxes[..., 0] >= 0).any(dim=2)
    print(f"[{split}] saved: {path}")
    print(
        "[CUB-visible scans] "
        + " | ".join(
            f"{part}:{int(visible_gt[:, index].sum())}/{len(samples)}"
            for index, part in enumerate(PARTS)
        )
    )
    print(
        "[valid detections] "
        + " | ".join(
            f"{part}:{int(detected[:, index].sum())}/{len(samples)}"
            for index, part in enumerate(PARTS)
        )
    )

    wing_index = PARTS.index("wing")
    two_wing_boxes = (boxes[:, wing_index, :, 0] >= 0).sum(dim=1) >= 2
    print(
        f"[wing] two boxes: {int(two_wing_boxes.sum())}/{len(samples)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run each CUB coarse part as a separate GroundingDINO prompt, "
            "gated by CUB ground-truth part visibility."
        )
    )
    parser.add_argument("--cub-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--processed-image-root",
        default=None,
        help=(
            "Optional root containing GT-bbox-cropped, square-resized images. "
            "The root may either contain images directly or have an images/ "
            "subdirectory. If omitted, this script crops and resizes CUB images."
        ),
    )
    parser.add_argument(
        "--model-id",
        default="IDEA-Research/grounding-dino-tiny",
    )
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.50)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--split",
        choices=("all", "train", "test"),
        default="all",
    )
    # Accepted for compatibility with the previous repository command line.
    # The new output is intentionally fixed to wing=2 and all other parts=1.
    parser.add_argument(
        "--max-boxes-per-part",
        type=int,
        default=2,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.cub_root = os.path.abspath(os.path.expanduser(args.cub_root))
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    if args.processed_image_root:
        args.processed_image_root = os.path.abspath(
            os.path.expanduser(args.processed_image_root)
        )

    if not os.path.isdir(args.cub_root):
        raise FileNotFoundError(f"CUB root does not exist: {args.cub_root}")
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
    validate_args(args)

    print(f"[Device] {args.device}")
    print(f"[Model] {args.model_id}")
    print("[Prompt mode] one GroundingDINO forward per visible coarse part")
    print("[Box policy] wing=2, all other parts=1")
    print("[Visibility] CUB parts/part_locs.txt")
    if args.processed_image_root:
        print(f"[Images] processed root: {args.processed_image_root}")
    else:
        print(
            f"[Images] CUB GT bbox crop -> direct resize "
            f"{args.image_size}x{args.image_size}"
        )

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
