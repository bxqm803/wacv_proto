import os, random, inspect, json, torch, numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

CUB_ROOT = "./data/CUB_200_2011"
OUT = "./runs/gdino_5img_probe_cropdetect_resize224"
FINAL_IMAGE_SIZE = 224

BOX_THR = 0.25
TEXT_THR = 0.20

SEED = 0
NUM_IMAGES = 5
SPLIT = "train"   # train / test / all

PARTS = ["beak", "head", "tail", "body", "feet", "wing"]

PROMPTS = {
    "beak": "beak.",
    "head": "head.",
    "tail": "tail.",
    "body": "torso.",
    "feet": "feet.",
    "wing": "wing.",
}

MAX_KEEP = {
    "beak": 1,
    "head": 1,
    "tail": 1,
    "body": 1,
    "feet": 2,
    "wing": 2,
}

SECOND_BOX_MIN_SCORE = {
    "feet": 0.18,
    "wing": 0.18,
}
SECOND_BOX_REL_SCORE = {
    "feet": 0.60,
    "wing": 0.60,
}
IOU_DUP_THR = 0.85

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


def post_process(processor, outputs, input_ids, target_h, target_w):
    target_sizes = torch.tensor([[target_h, target_w]], device=device)

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


def select_boxes_by_score(part, boxes, scores, crop_w, crop_h):
    if len(boxes) == 0:
        return []

    order = torch.argsort(scores, descending=True).tolist()
    candidates = []

    for idx in order:
        box = boxes[idx].tolist()
        score = float(scores[idx])

        x1, y1, x2, y2 = box
        x1 = max(0.0, min(float(crop_w), x1))
        y1 = max(0.0, min(float(crop_h), y1))
        x2 = max(0.0, min(float(crop_w), x2))
        y2 = max(0.0, min(float(crop_h), y2))

        area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(crop_w * crop_h)

        candidates.append({
            "box_crop": [x1, y1, x2, y2],
            "score": score,
            "area_crop": area,
            "prompt": PROMPTS[part],
        })

    max_keep = MAX_KEEP[part]

    if max_keep == 1:
        return candidates[:1]

    keep = []
    top1 = candidates[0]
    keep.append(top1)

    if len(candidates) == 1:
        return keep

    top1_score = top1["score"]
    min_score = SECOND_BOX_MIN_SCORE.get(part, 0.0)
    rel_score = SECOND_BOX_REL_SCORE.get(part, 0.0)

    for cand in candidates[1:]:
        if len(keep) >= max_keep:
            break

        cond_score_abs = cand["score"] >= min_score
        cond_score_rel = cand["score"] >= top1_score * rel_score
        cond_iou = box_iou_xyxy(cand["box_crop"], keep[0]["box_crop"]) < IOU_DUP_THR

        if cond_score_abs and cond_score_rel and cond_iou:
            keep.append(cand)

    return keep


def scale_detections_to_final(detections, crop_w, crop_h, final_size):
    sx = final_size / float(crop_w)
    sy = final_size / float(crop_h)

    out = {}
    for part, dets in detections.items():
        scaled = []
        for det in dets:
            x1, y1, x2, y2 = det["box_crop"]
            nx1 = x1 * sx
            ny1 = y1 * sy
            nx2 = x2 * sx
            ny2 = y2 * sy

            area = max(0.0, nx2 - nx1) * max(0.0, ny2 - ny1) / float(final_size * final_size)

            scaled.append({
                "box": [nx1, ny1, nx2, ny2],
                "score": det["score"],
                "area": area,
                "prompt": det["prompt"],
                "box_crop": det["box_crop"],
                "area_crop": det["area_crop"],
            })
        out[part] = scaled
    return out


@torch.no_grad()
def detect_part_on_crop(crop_img, part, processor, model):
    inputs = processor(
        images=crop_img,
        text=PROMPTS[part],
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs)

    result = post_process(
        processor=processor,
        outputs=outputs,
        input_ids=inputs.input_ids,
        target_h=crop_img.height,
        target_w=crop_img.width,
    )

    boxes = result["boxes"].detach().cpu()
    scores = result["scores"].detach().cpu()

    return select_boxes_by_score(part, boxes, scores, crop_img.width, crop_img.height)


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


def make_sheet(img, detections, title):
    font = ImageFont.load_default()
    panels = []

    panels.append(("resized", img.copy()))
    panels.append(("all", draw_dets(img, detections)))

    for part in PARTS:
        tmp = img.copy()
        draw = ImageDraw.Draw(tmp)
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

        panels.append((part, tmp))

    cols = 4
    rows = 2
    label_h = 18
    header_h = 24

    canvas = Image.new("RGB", (cols * FINAL_IMAGE_SIZE, rows * (FINAL_IMAGE_SIZE + label_h) + header_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title, fill=(0, 0, 0), font=font)

    for i, (name, panel) in enumerate(panels):
        r = i // cols
        c = i % cols
        x = c * FINAL_IMAGE_SIZE
        y = header_h + r * (FINAL_IMAGE_SIZE + label_h)
        draw.text((x + 4, y + 2), name, fill=(0, 0, 0), font=font)
        canvas.paste(panel, (x, y + label_h))

    return canvas


print("device:", device)
print("CUB_ROOT:", CUB_ROOT)
print("OUT:", OUT)
print("preprocess: gt_bbox_crop -> detect_on_crop -> resize_to_224_and_scale_boxes")

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
model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device).eval()

all_results = []

for rank, img_id in enumerate(ids, 1):
    rel = paths[img_id]
    img_path = os.path.join(CUB_ROOT, "images", rel)

    raw_img = Image.open(img_path).convert("RGB")
    crop_img = crop_bbox(raw_img, bboxes[img_id])

    crop_w, crop_h = crop_img.size

    detections_crop = {}

    print(f"\n[{rank}] img_id={img_id} {rel} crop_size=({crop_w},{crop_h})")

    for part in PARTS:
        dets = detect_part_on_crop(crop_img, part, processor, model)
        detections_crop[part] = dets

        if len(dets) == 0:
            print(f"  {part}: missing")
        else:
            for k, det in enumerate(dets):
                box = [round(x, 1) for x in det["box_crop"]]
                print(
                    f"  {part}[{k}]: "
                    f"score={det['score']:.3f} "
                    f"area_crop={det['area_crop']:.3f} "
                    f"box_crop={box}"
                )

    final_img = resize_only(crop_img, FINAL_IMAGE_SIZE)
    detections_final = scale_detections_to_final(detections_crop, crop_w, crop_h, FINAL_IMAGE_SIZE)

    safe_rel = rel.replace("/", "__")

    overlay = draw_dets(final_img, detections_final)
    sheet = make_sheet(final_img, detections_final, f"{rank:03d} img_id={img_id} {rel}")

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
        "crop_size": [crop_w, crop_h],
        "final_size": [FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE],
        "overlay_path": overlay_path,
        "sheet_path": sheet_path,
        "detections_crop": detections_crop,
        "detections_final": detections_final,
    })

json_path = os.path.join(OUT, "results.json")
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)

print("\nDone:", OUT)
print("JSON:", json_path)
