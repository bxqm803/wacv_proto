import os
import json
import math
import inspect
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


PARTS = ["beak", "head", "tail", "body", "feet", "wing"]

# 固定 prompt：沿用你现在这套
PROMPTS = {
    "beak": "beak of a bird.",
    "head": "head of a bird.",
    "tail": "tail feathers.",
    "body": "bird breast.",
    "feet": "bird foot.",
    "wing": "wing of a bird.",
}

# feet / wing 最多两个，其余一个
MAX_KEEP = {
    "beak": 1,
    "head": 1,
    "tail": 1,
    "body": 1,
    "feet": 2,
    "wing": 2,
}

SECOND_BOX_MIN_SCORE = {
    "feet": 0.16,
    "wing": 0.16,
}

SECOND_BOX_REL_SCORE = {
    "feet": 0.55,
    "wing": 0.55,
}

IOU_DUP_THR = 0.80
DROP_AREA_THR = 0.80

MIN_AREA = {
    "beak": 0.0002,
    "head": 0.001,
    "tail": 0.001,
    "body": 0.005,
    "feet": 0.0003,
    "wing": 0.002,
}

# CUB 15 parts -> 6 semantic groups
CUB_GROUPS = {
    "beak": ["beak"],
    "head": ["crown", "forehead", "left eye", "right eye", "nape", "throat"],
    "tail": ["tail"],
    "body": ["back", "belly", "breast"],
    "feet": ["left leg", "right leg"],
    "wing": ["left wing", "right wing"],
}


def normalize_cub_root(cub_root):
    cub_root = Path(cub_root)
    if (cub_root / "images.txt").exists():
        return cub_root
    if (cub_root / "CUB_200_2011" / "images.txt").exists():
        return cub_root / "CUB_200_2011"
    raise FileNotFoundError(f"Cannot find CUB root: {cub_root}")


def read_kv_text(path):
    d = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                k, v = line.split(maxsplit=1)
                d[int(k)] = v
    return d


def read_kv_int(path):
    d = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                k, v = line.split()
                d[int(k)] = int(v)
    return d


def read_bboxes(path):
    d = {}
    with open(path, "r") as f:
        for line in f:
            vals = line.strip().split()
            if vals:
                img_id = int(vals[0])
                x, y, w, h = map(float, vals[1:5])
                d[img_id] = (x, y, w, h)
    return d


def read_part_names(cub_root):
    path = cub_root / "parts" / "parts.txt"
    part_names = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                pid, name = line.split(maxsplit=1)
                part_names[int(pid)] = name
    return part_names


def build_visible_group_flags(cub_root):
    """
    返回:
      visible[img_id][part] = True/False
    """
    part_names = read_part_names(cub_root)
    loc_path = cub_root / "parts" / "part_locs.txt"

    name_to_group = {}
    for group, names in CUB_GROUPS.items():
        for name in names:
            name_to_group[name] = group

    visible = {}

    with open(loc_path, "r") as f:
        for line in f:
            vals = line.strip().split()
            if not vals:
                continue

            img_id = int(vals[0])
            part_id = int(vals[1])
            is_visible = int(vals[4])

            pname = part_names[part_id]
            group = name_to_group.get(pname, None)
            if group is None:
                continue

            if img_id not in visible:
                visible[img_id] = {p: False for p in PARTS}

            if is_visible == 1:
                visible[img_id][group] = True

    return visible


def crop_bbox(img, bbox):
    x, y, w, h = bbox
    W, H = img.size

    x1 = max(0, min(W - 1, int(math.floor(x))))
    y1 = max(0, min(H - 1, int(math.floor(y))))
    x2 = max(x1 + 1, min(W, int(math.ceil(x + w))))
    y2 = max(y1 + 1, min(H, int(math.ceil(y + h))))

    return img.crop((x1, y1, x2, y2))


def resize_only(img, size):
    if hasattr(Image, "Resampling"):
        return img.resize((size, size), Image.Resampling.BICUBIC)
    return img.resize((size, size), Image.BICUBIC)


def clamp_box(box, image_size):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(image_size), float(x1)))
    y1 = max(0.0, min(float(image_size), float(y1)))
    x2 = max(0.0, min(float(image_size), float(x2)))
    y2 = max(0.0, min(float(image_size), float(y2)))
    return [x1, y1, x2, y2]


def box_area(box, image_size):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(image_size * image_size)


def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def post_process(processor, outputs, input_ids, image_size, device, box_thr, text_thr):
    target_sizes = torch.tensor([[image_size, image_size]], device=device)

    fn = processor.post_process_grounded_object_detection
    sig = inspect.signature(fn)

    kwargs = {
        "outputs": outputs,
        "target_sizes": target_sizes,
    }

    if "input_ids" in sig.parameters:
        kwargs["input_ids"] = input_ids

    if "box_threshold" in sig.parameters:
        kwargs["box_threshold"] = box_thr
    else:
        kwargs["threshold"] = box_thr

    if "text_threshold" in sig.parameters:
        kwargs["text_threshold"] = text_thr

    try:
        return fn(**kwargs)[0]
    except TypeError:
        return fn(
            outputs,
            input_ids,
            box_threshold=box_thr,
            text_threshold=text_thr,
            target_sizes=target_sizes,
        )[0]


@torch.no_grad()
def detect_part(img, part, processor, model, device, image_size, box_thr, text_thr):
    prompt = PROMPTS[part]

    inputs = processor(
        images=img,
        text=prompt,
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs)

    result = post_process(
        processor=processor,
        outputs=outputs,
        input_ids=inputs.input_ids,
        image_size=image_size,
        device=device,
        box_thr=box_thr,
        text_thr=text_thr,
    )

    boxes = result["boxes"].detach().cpu()
    scores = result["scores"].detach().cpu()

    if len(boxes) == 0:
        return [], []

    order = torch.argsort(scores, descending=True).tolist()

    candidates = []
    for idx in order:
        box = clamp_box(boxes[idx].tolist(), image_size)
        area = box_area(box, image_size)
        candidates.append({
            "box": box,
            "score": float(scores[idx]),
            "area": float(area),
            "prompt": prompt,
            "drop_area": bool(area > DROP_AREA_THR),
            "too_small": bool(area < MIN_AREA[part]),
        })

    valid = [
        c for c in candidates
        if (not c["drop_area"]) and (not c["too_small"])
    ]

    if len(valid) == 0:
        return [], candidates

    max_keep = MAX_KEEP[part]

    if max_keep == 1:
        return [valid[0]], candidates

    keep = [valid[0]]
    top1_score = valid[0]["score"]

    for cand in valid[1:]:
        if len(keep) >= max_keep:
            break

        cond_abs = cand["score"] >= SECOND_BOX_MIN_SCORE.get(part, 0.0)
        cond_rel = cand["score"] >= top1_score * SECOND_BOX_REL_SCORE.get(part, 0.0)
        cond_iou = all(box_iou_xyxy(cand["box"], k["box"]) < IOU_DUP_THR for k in keep)

        if cond_abs and cond_rel and cond_iou:
            keep.append(cand)

    return keep, candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub-root", type=str, default="./data/CUB_200_2011")
    parser.add_argument("--output-dir", type=str, default="./artifacts/gdino_part_boxes_visible_resize224")
    parser.add_argument("--model-id", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--box-thr", type=float, default=0.22)
    parser.add_argument("--text-thr", type=float, default=0.18)
    parser.add_argument("--split", type=str, default="both", choices=["train", "test", "both"])
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    cub_root = normalize_cub_root(args.cub_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("cub_root:", cub_root)
    print("output_dir:", output_dir)

    paths = read_kv_text(cub_root / "images.txt")
    split_map = read_kv_int(cub_root / "train_test_split.txt")
    bboxes = read_bboxes(cub_root / "bounding_boxes.txt")
    visible = build_visible_group_flags(cub_root)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device).eval()

    split_specs = []
    if args.split in ["train", "both"]:
        split_specs.append(("train", 1))
    if args.split in ["test", "both"]:
        split_specs.append(("test", 0))

    for split_name, split_flag in split_specs:
        img_ids = [i for i in paths if split_map[i] == split_flag]
        img_ids = sorted(img_ids)

        if args.max_images > 0:
            img_ids = img_ids[:args.max_images]

        results = {
            "split": split_name,
            "image_size": args.image_size,
            "box_thr": args.box_thr,
            "text_thr": args.text_thr,
            "drop_area_thr": DROP_AREA_THR,
            "prompts": PROMPTS,
            "img_ids": [],
            "relpaths": [],
            "visible_parts": [],
            "selected_boxes_xyxy": [],
            "selected_scores": [],
            "selected_areas": [],
            "raw_candidate_counts": [],
        }

        num_missing_image = 0
        num_run_part = 0
        num_skip_part = 0

        for idx, img_id in enumerate(img_ids, 1):
            rel = paths[img_id]
            img_path = cub_root / "images" / rel

            if img_id not in visible:
                vis_flags = {p: False for p in PARTS}
            else:
                vis_flags = visible[img_id]

            if not os.path.exists(img_path):
                num_missing_image += 1
                continue

            img = Image.open(img_path).convert("RGB")
            img = crop_bbox(img, bboxes[img_id])
            img = resize_only(img, args.image_size)

            image_boxes = []
            image_scores = []
            image_areas = []
            image_cand_counts = []
            image_vis = []

            for part in PARTS:
                is_vis = bool(vis_flags.get(part, False))
                image_vis.append(is_vis)

                if not is_vis:
                    num_skip_part += 1
                    image_boxes.append([])
                    image_scores.append([])
                    image_areas.append([])
                    image_cand_counts.append(0)
                    continue

                num_run_part += 1
                selected, candidates = detect_part(
                    img=img,
                    part=part,
                    processor=processor,
                    model=model,
                    device=device,
                    image_size=args.image_size,
                    box_thr=args.box_thr,
                    text_thr=args.text_thr,
                )

                image_boxes.append([det["box"] for det in selected])
                image_scores.append([det["score"] for det in selected])
                image_areas.append([det["area"] for det in selected])
                image_cand_counts.append(len(candidates))

            results["img_ids"].append(img_id)
            results["relpaths"].append(rel)
            results["visible_parts"].append(image_vis)          # [6]
            results["selected_boxes_xyxy"].append(image_boxes)  # [6][K][4]
            results["selected_scores"].append(image_scores)     # [6][K]
            results["selected_areas"].append(image_areas)       # [6][K]
            results["raw_candidate_counts"].append(image_cand_counts)

            if idx % args.log_every == 0 or idx == len(img_ids):
                print(
                    f"[{split_name}] {idx}/{len(img_ids)} "
                    f"run_parts={num_run_part} skip_parts={num_skip_part}"
                )

        out_pt = output_dir / f"{split_name}_part_boxes_visible_resize224.pt"
        torch.save(results, out_pt)
        print(f"saved: {out_pt}")

        if args.save_json:
            out_json = output_dir / f"{split_name}_part_boxes_visible_resize224.json"
            with open(out_json, "w") as f:
                json.dump(results, f)
            print(f"saved: {out_json}")

        print(
            f"[{split_name}] done. images={len(results['img_ids'])}, "
            f"missing_image={num_missing_image}, run_parts={num_run_part}, skip_parts={num_skip_part}"
        )


if __name__ == "__main__":
    main()
