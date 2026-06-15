import os
import json
import math
import random
import inspect
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


PARTS = ["beak", "head", "tail", "body", "feet", "wing"]

# 这里不要写成 bird beak.，否则有时会更容易框到整只鸟。
PROMPTS = {
    "beak": "beak.",
    "head": "head.",
    "tail": "tail.",
    "body": "body.",
    "feet": "feet.",
    "wing": "wing.",
}

TOPK = {
    "beak": 1,
    "head": 1,
    "tail": 1,
    "body": 1,
    "feet": 1,
    "wing": 2,
}

# 防止 GroundingDINO 把整只鸟当成某个部位。
# body 可以大一些，其他部位限制更严。
MAX_AREA = {
    "beak": 0.20,
    "head": 0.45,
    "tail": 0.45,
    "body": 0.85,
    "feet": 0.35,
    "wing": 0.60,
}

MIN_AREA = {
    "beak": 0.0005,
    "head": 0.002,
    "tail": 0.002,
    "body": 0.01,
    "feet": 0.001,
    "wing": 0.005,
}

COLORS = {
    "beak": (255, 0, 0),
    "head": (255, 128, 0),
    "tail": (180, 0, 255),
    "body": (0, 180, 0),
    "feet": (0, 128, 255),
    "wing": (255, 0, 180),
}

CUB_PART_GROUPS = {
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
    raise FileNotFoundError(f"Cannot find CUB images.txt under: {cub_root}")


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
    if not path.exists():
        return None

    part_names = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                pid, name = line.split(maxsplit=1)
                part_names[int(pid)] = name
    return part_names


def read_visible_groups(cub_root):
    part_names = read_part_names(cub_root)
    loc_path = cub_root / "parts" / "part_locs.txt"

    if part_names is None or not loc_path.exists():
        return None

    name_to_group = {}
    for group, names in CUB_PART_GROUPS.items():
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

            part_name = part_names.get(part_id)
            group = name_to_group.get(part_name)

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


def pil_resize(img, size):
    if hasattr(Image, "Resampling"):
        return img.resize((size, size), Image.Resampling.BICUBIC)
    return img.resize((size, size), Image.BICUBIC)


def post_process_grounded(processor, outputs, input_ids, target_sizes, box_thr, text_thr):
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
    elif "threshold" in sig.parameters:
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
def detect_part(img, part, processor, model, device, image_size, box_thr, text_thr, fallback_box_thr, fallback_text_thr):
    prompt = PROMPTS[part]

    inputs = processor(
        images=img,
        text=prompt,
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs)

    target_sizes = torch.tensor([[image_size, image_size]], device=device)

    result = post_process_grounded(
        processor=processor,
        outputs=outputs,
        input_ids=inputs.input_ids,
        target_sizes=target_sizes,
        box_thr=box_thr,
        text_thr=text_thr,
    )

    used_fallback = False

    if len(result["boxes"]) == 0 and fallback_box_thr is not None:
        result = post_process_grounded(
            processor=processor,
            outputs=outputs,
            input_ids=inputs.input_ids,
            target_sizes=target_sizes,
            box_thr=fallback_box_thr,
            text_thr=fallback_text_thr,
        )
        used_fallback = True

    boxes = result["boxes"].detach().cpu()
    scores = result["scores"].detach().cpu()

    if len(boxes) == 0:
        return []

    candidates = []

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [float(x) for x in box.tolist()]

        x1 = max(0.0, min(float(image_size), x1))
        y1 = max(0.0, min(float(image_size), y1))
        x2 = max(0.0, min(float(image_size), x2))
        y2 = max(0.0, min(float(image_size), y2))

        if x2 <= x1 or y2 <= y1:
            continue

        area = ((x2 - x1) * (y2 - y1)) / float(image_size * image_size)

        candidates.append({
            "box": [x1, y1, x2, y2],
            "score": float(score),
            "area": float(area),
            "prompt": prompt,
            "used_fallback": used_fallback,
        })

    if len(candidates) == 0:
        return []

    filtered = [
        c for c in candidates
        if MIN_AREA[part] <= c["area"] <= MAX_AREA[part]
    ]

    # 如果面积过滤后没有框，就退回最高分框，但标记 area_filter_failed。
    if len(filtered) == 0:
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        keep = candidates[:TOPK[part]]
        for c in keep:
            c["area_filter_failed"] = True
        return keep

    filtered = sorted(filtered, key=lambda x: x["score"], reverse=True)
    keep = filtered[:TOPK[part]]
    for c in keep:
        c["area_filter_failed"] = False
    return keep


def draw_label(draw, xy, text, color, font):
    x, y = xy
    x = int(max(0, x))
    y = int(max(0, y))

    try:
        box = draw.textbbox((x, y), text, font=font)
        pad = 2
        bg = (255, 255, 255)
        draw.rectangle(
            (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
            fill=bg,
        )
    except Exception:
        pass

    draw.text((x, y), text, fill=color, font=font)


def draw_detections(img, detections, only_part=None):
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    font = ImageFont.load_default()

    parts = [only_part] if only_part is not None else PARTS

    for part in parts:
        if part is None:
            continue

        color = COLORS[part]
        for det in detections.get(part, []):
            x1, y1, x2, y2 = det["box"]
            box_int = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

            draw.rectangle(box_int, outline=color, width=3)

            tag = f"{part}:{det['score']:.2f},a={det['area']:.2f}"
            if det.get("used_fallback", False):
                tag += ",fb"
            if det.get("area_filter_failed", False):
                tag += ",large"

            draw_label(
                draw,
                (box_int[0], max(0, box_int[1] - 12)),
                tag,
                color,
                font,
            )

    return vis


def make_contact_sheet(img, detections, title, image_size):
    panels = []
    panels.append(("original", img.copy()))
    panels.append(("all parts", draw_detections(img, detections, only_part=None)))

    for part in PARTS:
        panels.append((part, draw_detections(img, detections, only_part=part)))

    cols = 4
    rows = 2
    header_h = 24
    label_h = 18

    W = cols * image_size
    H = rows * (image_size + label_h) + header_h

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((4, 4), title, fill=(0, 0, 0), font=font)

    for idx, (name, panel) in enumerate(panels):
        r = idx // cols
        c = idx % cols

        x = c * image_size
        y = header_h + r * (image_size + label_h)

        canvas.paste(panel, (x, y + label_h))
        draw.text((x + 4, y + 2), name, fill=(0, 0, 0), font=font)

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub-root", type=str, default="./data/CUB_200_2011")
    parser.add_argument("--out", type=str, default="./runs/gdino_5img_probe")
    parser.add_argument("--model-id", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"])
    parser.add_argument("--box-thr", type=float, default=0.25)
    parser.add_argument("--text-thr", type=float, default=0.20)

    # 如果高阈值没有框，用 fallback 阈值只为了可视化。
    # 如果不想 fallback，传 --no-fallback。
    parser.add_argument("--fallback-box-thr", type=float, default=0.12)
    parser.add_argument("--fallback-text-thr", type=float, default=0.10)
    parser.add_argument("--no-fallback", action="store_true")

    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--require-visible-all", action="store_true")
    args = parser.parse_args()

    cub_root = normalize_cub_root(args.cub_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = read_kv_text(cub_root / "images.txt")
    split = read_kv_int(cub_root / "train_test_split.txt")
    bboxes = read_bboxes(cub_root / "bounding_boxes.txt")

    if args.split == "train":
        ids = [i for i in paths if split[i] == 1]
    elif args.split == "test":
        ids = [i for i in paths if split[i] == 0]
    else:
        ids = list(paths.keys())

    if args.require_visible_all:
        visible = read_visible_groups(cub_root)
        if visible is not None:
            ids_vis = []
            for img_id in ids:
                v = visible.get(img_id)
                if v is not None and all(v.get(p, False) for p in PARTS):
                    ids_vis.append(img_id)

            if len(ids_vis) >= args.num_images:
                ids = ids_vis
                print(f"Using visible-all filter: {len(ids)} eligible images")
            else:
                print(f"Warning: visible-all eligible images too few: {len(ids_vis)}. Fallback to original split.")
        else:
            print("Warning: parts/part_locs.txt not found. Skip visible-all filter.")

    random.seed(args.seed)
    ids = random.sample(ids, min(args.num_images, len(ids)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("CUB_ROOT:", cub_root)
    print("OUT:", out_dir)
    print("device:", device)
    print("sample ids:", ids)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device).eval()

    fallback_box_thr = None if args.no_fallback else args.fallback_box_thr
    fallback_text_thr = None if args.no_fallback else args.fallback_text_thr

    all_results = []

    for rank, img_id in enumerate(ids, 1):
        rel = paths[img_id]
        img_path = cub_root / "images" / rel

        img = Image.open(img_path).convert("RGB")

        if not args.no_crop:
            img = crop_bbox(img, bboxes[img_id])

        img = pil_resize(img, args.image_size)

        detections = {}

        print(f"\n[{rank}] img_id={img_id} {rel}")

        for part in PARTS:
            dets = detect_part(
                img=img,
                part=part,
                processor=processor,
                model=model,
                device=device,
                image_size=args.image_size,
                box_thr=args.box_thr,
                text_thr=args.text_thr,
                fallback_box_thr=fallback_box_thr,
                fallback_text_thr=fallback_text_thr,
            )

            detections[part] = dets

            if len(dets) == 0:
                print(f"  {part:>5s}: missing")
            else:
                for k, d in enumerate(dets):
                    box = [round(x, 1) for x in d["box"]]
                    mark = ""
                    if d.get("used_fallback", False):
                        mark += " fallback"
                    if d.get("area_filter_failed", False):
                        mark += " area_filter_failed"

                    print(
                        f"  {part:>5s}[{k}]: "
                        f"score={d['score']:.3f} "
                        f"area={d['area']:.3f} "
                        f"box={box}"
                        f"{mark}"
                    )

        safe_rel = rel.replace("/", "__")
        overlay = draw_detections(img, detections)
        sheet = make_contact_sheet(
            img=img,
            detections=detections,
            title=f"{rank:03d} img_id={img_id} {rel}",
            image_size=args.image_size,
        )

        overlay_path = out_dir / f"{rank:03d}_{safe_rel}_overlay.jpg"
        sheet_path = out_dir / f"{rank:03d}_{safe_rel}_sheet.jpg"

        overlay.save(overlay_path)
        sheet.save(sheet_path)

        print("  saved overlay:", overlay_path)
        print("  saved sheet:  ", sheet_path)

        all_results.append({
            "rank": rank,
            "img_id": img_id,
            "relpath": rel,
            "overlay_path": str(overlay_path),
            "sheet_path": str(sheet_path),
            "detections": detections,
        })

    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nDone:", out_dir)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
