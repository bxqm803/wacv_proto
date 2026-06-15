from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os, random, inspect, json, torch, numpy as np
CUB_ROOT = "./data/CUB_200_2011"
OUT = "./runs/gdino_5img_probe_prompt_ensemble"
IMAGE_SIZE = 224

BOX_THR = 0.22
TEXT_THR = 0.18

SEED = 0
NUM_IMAGES = 5
SPLIT = "train"  # train / test / all

USE_GT_BBOX_CROP = True

PARTS = ["beak", "head", "tail", "body", "feet", "wing"]

PROMPTS = {
    "beak": [
        "bird beak.",
        "bird bill.",
        "beak of a bird.",
    ],
    "head": [
        "bird head.",
        "head of a bird.",
    ],
    "tail": [
        "bird tail.",
        "tail feathers.",
        "bird tail feathers.",
    ],
    "body": [
        "bird breast.",
        "bird belly.",
        "bird back.",
    ],
    "feet": [
        "bird feet.",
        "bird foot.",
        "bird legs.",
        "bird claws.",
    ],
    "wing": [
        "bird wing.",
        "bird wings.",
        "left bird wing.",
        "right bird wing.",
    ],
}

MAX_KEEP = {
    "beak": 1,
    "head": 1,
    "tail": 1,
    "body": 1,
    "feet": 2,
    "wing": 2,
}

# 第二个框是否保留：只对 feet / wing 生效
SECOND_BOX_MIN_SCORE = {
    "feet": 0.16,
    "wing": 0.16,
}
SECOND_BOX_REL_SCORE = {
    "feet": 0.55,
    "wing": 0.55,
}

# 太像的框去重
IOU_DUP_THR = 0.80

# 防止 body/wing/head 等直接框整只鸟
MIN_AREA = {
    "beak": 0.0003,
    "head": 0.002,
    "tail": 0.002,
    "body": 0.010,
    "feet": 0.0005,
    "wing": 0.004,
}

MAX_AREA = {
    "beak": 0.25,
    "head": 0.45,
    "tail": 0.55,
    "body": 0.65,
    "feet": 0.35,
    "wing": 0.65,
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


def select_candidates(part, candidates):
    if len(candidates) == 0:
        return []

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    # 先过滤明显过大/过小的框
    filtered = [
        c for c in candidates
        if MIN_AREA[part] <= c["area"] <= MAX_AREA[part]
    ]

    # 如果过滤后全没了，回退到原始最高分
    pool = filtered if len(filtered) > 0 else candidates

    max_keep = MAX_KEEP[part]

    # 普通部位只保留最高分
    if max_keep == 1:
        return [pool[0]]

    # feet / wing：最多保留两个，但第二个要满足分数和去重条件
    keep = [pool[0]]
    top1_score = pool[0]["score"]

    for cand in pool[1:]:
        if len(keep) >= max_keep:
            break

        cond_score_abs = cand["score"] >= SECOND_BOX_MIN_SCORE.get(part, 0.0)
        cond_score_rel = cand["score"] >= top1_score * SECOND_BOX_REL_SCORE.get(part, 0.0)
        cond_iou = all(box_iou_xyxy(cand["box"], k["box"]) < IOU_DUP_THR for k in keep)

        if cond_score_abs and cond_score_rel and cond_iou:
            keep.append(cand)

    return keep


@torch.no_grad()
def detect_part_prompt_ensemble(img, part, processor, model):
    candidates = []

    for prompt in PROMPTS[part]:
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

        for box, score in zip(boxes, scores):
            box = clamp_box(box.tolist())
            area = box_area(box)

            if area <= 0:
                continue

            candidates.append({
                "box": box,
                "score": float(score),
                "area": float(area),
                "prompt": prompt,
            })

    return select_candidates(part, candidates)


def draw_dets(img, detections):
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    font = ImageFont.load_default()

    for part in PARTS:
        color = COLORS[part]

        for det in detections[part]:
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


def draw_single_part(img, detections, part):
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    font = ImageFont.load_default()
    color = COLORS[part]

    for det in detections[part]:
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


def make_sheet(img, detections, title):
    font = ImageFont.load_default()
    panels = []

    panels.append(("original", img.copy()))
    panels.append(("all", draw_dets(img, detections)))

    for part in PARTS:
        panels.append((part, draw_single_part(img, detections, part)))

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
print("preprocess:", "GT bbox crop -> resize 224 only -> GroundingDINO")
print("BOX_THR:", BOX_THR, "TEXT_THR:", TEXT_THR)

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

for rank, img_id in enumerate(ids, 1):
    rel = paths[img_id]
    img_path = os.path.join(CUB_ROOT, "images", rel)

    img = Image.open(img_path).convert("RGB")

    if USE_GT_BBOX_CROP:
        img = crop_bbox(img, bboxes[img_id])

    # 关键：只 resize，不 center crop
    img = resize_only(img, IMAGE_SIZE)

    detections = {}

    print(f"\n[{rank}] img_id={img_id} {rel}")

    for part in PARTS:
        dets = detect_part_prompt_ensemble(img, part, processor, model)
        detections[part] = dets

        if len(dets) == 0:
            print(f"  {part}: missing")
        else:
            for k, det in enumerate(dets):
                box = [round(x, 1) for x in det["box"]]
                print(
                    f"  {part}[{k}]: "
                    f"score={det['score']:.3f} "
                    f"area={det['area']:.3f} "
                    f"box={box} "
                    f"prompt={det['prompt']}"
                )

    safe_rel = rel.replace("/", "__")

    overlay = draw_dets(img, detections)
    sheet = make_sheet(img, detections, f"{rank:03d} img_id={img_id} {rel}")

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
        "detections": detections,
    })

json_path = os.path.join(OUT, "results.json")
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)

print("\nDone:", OUT)
print("JSON:", json_path)
