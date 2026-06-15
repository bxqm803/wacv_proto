import os, random, inspect, json, torch, numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

CUB_ROOT = "./data/CUB_200_2011"
OUT = "./runs/gdino_10img_probe_fixedprompt"
IMAGE_SIZE = 224

BOX_THR = 0.22
TEXT_THR = 0.18

SEED = 0
NUM_IMAGES = 10
SPLIT = "train"   # train / test / all

USE_GT_BBOX_CROP = True

PARTS = ["beak", "head", "tail", "body", "feet", "wing"]

# 每个部位固定一个 prompt
PROMPTS = {
    "beak": "beak of a bird.",
    "head": "head of a bird.",
    "tail": "tail feathers.",
    "body": "bird breast.",
    "feet": "bird foot.",
    "wing": "wing of a bird.",
}

# feet / wing 最多保留2个，其余1个
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

# 统一规则：area > 0.8 直接 drop
DROP_AREA_THR = 0.80

# 防止特别离谱的小框
MIN_AREA = {
    "beak": 0.0002,
    "head": 0.001,
    "tail": 0.001,
    "body": 0.005,
    "feet": 0.0003,
    "wing": 0.002,
}

COLORS = {
    "beak": (255, 0, 0),
    "head": (255, 128, 0),
    "tail": (180, 0, 255),
    "body": (0, 180, 0),
    "feet": (0, 128, 255),
    "wing": (255, 0, 180),
}

os.makedirs(OUT, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def read_kv_text(path):
    d = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                k, v = line.strip().split(maxsplit=1)
                d[int(k)] = v
    return d


def read_kv_int(path):
    d = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                k, v = line.strip().split()
                d[int(k)] = int(v)
    return d


def read_bboxes(path):
    d = {}
    with open(path) as f:
        for line in f:
            vals = line.strip().split()
            if vals:
                d[int(vals[0])] = tuple(float(x) for x in vals[1:5])
    return d


def crop_bbox(img, bbox):
    x, y, w, h = bbox
    W, H = img.size

    x1 = max(0, min(W - 1, int(np.floor(x))))
    y1 = max(0, min(H - 1, int(np.floor(y))))
    x2 = max(x1 + 1, min(W, int(np.ceil(x + w))))
    y2 = max(y1 + 1, min(H, int(np.ceil(y + h))))

    return img.crop((x1, y1, x2, y2))


def resize_only(img, size):
    if hasattr(Image, "Resampling"):
        return img.resize((size, size), Image.Resampling.BICUBIC)
    return img.resize((size, size), Image.BICUBIC)


def post_process(processor, outputs, input_ids, image_size):
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
        kwargs["box_threshold"] = BOX_THR
    else:
        kwargs["threshold"] = BOX_THR

    if "text_threshold" in sig.parameters:
        kwargs["text_threshold"] = TEXT_THR

    try:
        return fn(**kwargs)[0]
    except TypeError:
        return fn(
            outputs,
            input_ids,
            box_threshold=BOX_THR,
            text_threshold=TEXT_THR,
            target_sizes=target_sizes,
        )[0]


def clamp_box(box):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(IMAGE_SIZE), float(x1)))
    y1 = max(0.0, min(float(IMAGE_SIZE), float(y1)))
    x2 = max(0.0, min(float(IMAGE_SIZE), float(x2)))
    y2 = max(0.0, min(float(IMAGE_SIZE), float(y2)))
    return [x1, y1, x2, y2]


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(IMAGE_SIZE * IMAGE_SIZE)


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


@torch.no_grad()
def detect_part(img, part, processor, model):
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
        image_size=IMAGE_SIZE,
    )

    boxes = result["boxes"].detach().cpu()
    scores = result["scores"].detach().cpu()

    candidates = []

    if len(boxes) == 0:
        return [], []

    order = torch.argsort(scores, descending=True).tolist()

    for idx in order:
        box = clamp_box(boxes[idx].tolist())
        area = box_area(box)

        candidates.append({
            "box": box,
            "score": float(scores[idx]),
            "area": float(area),
            "prompt": prompt,
            "drop_area": bool(area > DROP_AREA_THR),
        })

    valid = [
        c for c in candidates
        if (not c["drop_area"]) and (c["area"] >= MIN_AREA[part])
    ]

    valid = sorted(valid, key=lambda x: x["score"], reverse=True)

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


def draw_selected_overlay(img, selected_by_part):
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    font = ImageFont.load_default()

    for part in PARTS:
        color = COLORS[part]
        for det in selected_by_part[part]:
            x1, y1, x2, y2 = det["box"]
            box = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

            draw.rectangle(box, outline=color, width=3)
            draw.text(
                (box[0], max(0, box[1] - 12)),
                f"{part}:{det['score']:.2f},a={det['area']:.2f}",
                fill=color,
                font=font,
            )

    return vis


def draw_part(img, selected_by_part, part):
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    font = ImageFont.load_default()
    color = COLORS[part]

    for det in selected_by_part[part]:
        x1, y1, x2, y2 = det["box"]
        box = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

        draw.rectangle(box, outline=color, width=3)
        draw.text(
            (box[0], max(0, box[1] - 12)),
            f"{part}:{det['score']:.2f},a={det['area']:.2f}",
            fill=color,
            font=font,
        )

    return vis


def make_selected_sheet(img, selected_by_part, title):
    font = ImageFont.load_default()

    panels = [
        ("original", img.copy()),
        ("selected_all", draw_selected_overlay(img, selected_by_part)),
    ]

    for part in PARTS:
        panels.append((part, draw_part(img, selected_by_part, part)))

    cols = 4
    rows = 2
    label_h = 18
    header_h = 24

    canvas = Image.new(
        "RGB",
        (cols * IMAGE_SIZE, rows * (IMAGE_SIZE + label_h) + header_h),
        (255, 255, 255),
    )

    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title, fill=(0, 0, 0), font=font)

    for i, (name, panel) in enumerate(panels):
        r = i // cols
        c = i % cols

        x = c * IMAGE_SIZE
        y = header_h + r * (IMAGE_SIZE + label_h)

        draw.text((x + 4, y + 2), name, fill=(0, 0, 0), font=font)
        canvas.paste(panel, (x, y + label_h))

    return canvas


print("device:", device)
print("CUB_ROOT:", CUB_ROOT)
print("OUT:", OUT)
print("preprocess: GT bbox crop -> resize 224 only -> GroundingDINO")
print("NUM_IMAGES:", NUM_IMAGES)
print("DROP_AREA_THR:", DROP_AREA_THR)

paths = read_kv_text(os.path.join(CUB_ROOT, "images.txt"))
split = read_kv_int(os.path.join(CUB_ROOT, "train_test_split.txt"))
bboxes = read_bboxes(os.path.join(CUB_ROOT, "bounding_boxes.txt"))

if SPLIT == "train":
    candidate_ids = [i for i in paths if split[i] == 1]
elif SPLIT == "test":
    candidate_ids = [i for i in paths if split[i] == 0]
else:
    candidate_ids = list(paths.keys())

random.seed(SEED)
ids = random.sample(candidate_ids, NUM_IMAGES)

processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-tiny"
).to(device).eval()

all_results = []
stats = {
    part: {
        "selected": 0,
        "missing": 0,
        "dropped_area": 0,
    }
    for part in PARTS
}

for rank, img_id in enumerate(ids, 1):
    rel = paths[img_id]
    img_path = os.path.join(CUB_ROOT, "images", rel)

    img = Image.open(img_path).convert("RGB")

    if USE_GT_BBOX_CROP:
        img = crop_bbox(img, bboxes[img_id])

    img = resize_only(img, IMAGE_SIZE)

    print(f"\n[{rank}] img_id={img_id} {rel}")

    selected_by_part = {}
    all_candidates_by_part = {}

    for part in PARTS:
        selected, candidates = detect_part(img, part, processor, model)

        selected_by_part[part] = selected
        all_candidates_by_part[part] = candidates

        dropped_area = sum(1 for c in candidates if c["drop_area"])
        stats[part]["dropped_area"] += dropped_area

        print(f"  {part}:")
        for k, c in enumerate(candidates[:5]):
            box = [round(x, 1) for x in c["box"]]
            flag = " DROP_AREA" if c["drop_area"] else ""
            print(
                f"    cand[{k}] score={c['score']:.3f} "
                f"area={c['area']:.3f} box={box}{flag}"
            )

        if len(selected) == 0:
            stats[part]["missing"] += 1
            print("    SELECTED: missing")
        else:
            stats[part]["selected"] += len(selected)
            for k, det in enumerate(selected):
                box = [round(x, 1) for x in det["box"]]
                print(
                    f"    SELECTED[{k}]: "
                    f"score={det['score']:.3f} "
                    f"area={det['area']:.3f} "
                    f"box={box} "
                    f"prompt={det['prompt']}"
                )

    safe_rel = rel.replace("/", "__")

    overlay = draw_selected_overlay(img, selected_by_part)
    sheet = make_selected_sheet(img, selected_by_part, f"{rank:03d} img_id={img_id} {rel}")

    overlay_path = os.path.join(OUT, f"{rank:03d}_{safe_rel}_overlay.jpg")
    sheet_path = os.path.join(OUT, f"{rank:03d}_{safe_rel}_sheet.jpg")

    overlay.save(overlay_path)
    sheet.save(sheet_path)

    print("  saved overlay:", overlay_path)
    print("  saved sheet:", sheet_path)

    all_results.append({
        "rank": rank,
        "img_id": img_id,
        "relpath": rel,
        "overlay_path": overlay_path,
        "sheet_path": sheet_path,
        "selected": selected_by_part,
        "all_candidates": all_candidates_by_part,
    })

json_path = os.path.join(OUT, "results.json")
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)

stats_path = os.path.join(OUT, "stats.json")
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

print("\n========== SUMMARY ==========")
for part in PARTS:
    print(f"\n{part}:")
    print(f"  selected boxes: {stats[part]['selected']}")
    print(f"  missing images: {stats[part]['missing']}")
    print(f"  dropped_area: {stats[part]['dropped_area']}")

print("\nDone:", OUT)
print("JSON:", json_path)
print("STATS:", stats_path)
