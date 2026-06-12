#!/usr/bin/env python3
"""Build DINOv2 ViT-B/14 CLS and patch-token memmaps for CUB-200-2011.

The image is first cropped to the CUB ground-truth bird bounding box and then
warped to 224x224, matching the training pipeline's expected coordinate system.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
            if not vals:
                continue
            image_id = int(vals[0])
            out[image_id] = tuple(float(x) for x in vals[1:5])
    return out


@dataclass(frozen=True)
class Sample:
    image_id: int
    relpath: str
    label: int
    bbox: Tuple[float, float, float, float]


def build_samples(cub_root: str, split: str) -> List[Sample]:
    paths = read_kv_text(os.path.join(cub_root, "images.txt"))
    labels = read_kv_int(os.path.join(cub_root, "image_class_labels.txt"))
    split_map = read_kv_int(os.path.join(cub_root, "train_test_split.txt"))
    bboxes = read_bboxes(os.path.join(cub_root, "bounding_boxes.txt"))
    want_train = split == "train"
    samples: List[Sample] = []
    for image_id in sorted(paths):
        if (split_map[image_id] == 1) != want_train:
            continue
        samples.append(
            Sample(
                image_id=image_id,
                relpath=paths[image_id],
                label=labels[image_id] - 1,
                bbox=bboxes[image_id],
            )
        )
    return samples


def crop_bbox(image: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
    x, y, w, h = bbox
    width, height = image.size
    x1 = max(0, min(width - 1, int(np.floor(x))))
    y1 = max(0, min(height - 1, int(np.floor(y))))
    x2 = max(x1 + 1, min(width, int(np.ceil(x + w))))
    y2 = max(y1 + 1, min(height, int(np.ceil(y + h))))
    return image.crop((x1, y1, x2, y2))


class CUBBBoxDataset(Dataset):
    def __init__(self, cub_root: str, split: str, image_size: int):
        self.cub_root = cub_root
        self.samples = build_samples(cub_root, split)
        self.image_size = int(image_size)
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = os.path.join(self.cub_root, "images", sample.relpath)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = crop_bbox(image, sample.bbox)
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
            tensor = (tensor - self.mean) / self.std
        return tensor, sample.label, sample.relpath


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cub-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="dinov2_vitb14")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--float32", action="store_true", help="Store float32 instead of float16.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def cache_paths(output_dir: str, split: str) -> Dict[str, str]:
    feat_dir = os.path.join(output_dir, "features")
    return {
        "meta": os.path.join(feat_dir, f"{split}_meta.json"),
        "cls": os.path.join(feat_dir, f"{split}_cls.dat"),
        "patch": os.path.join(feat_dir, f"{split}_patch.dat"),
        "labels": os.path.join(feat_dir, f"{split}_labels.npy"),
        "relpaths": os.path.join(feat_dir, f"{split}_relpaths.json"),
    }


def build_split(model, args: argparse.Namespace, split: str) -> None:
    paths = cache_paths(args.output_dir, split)
    os.makedirs(os.path.dirname(paths["meta"]), exist_ok=True)
    if not args.overwrite and all(os.path.isfile(x) for x in paths.values()):
        print(f"[{split}] cache exists; skipping. Use --overwrite to rebuild.")
        return

    ds = CUBBBoxDataset(args.cub_root, split, args.image_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )

    storage_dtype = np.float32 if args.float32 else np.float16
    cls_mm = patch_mm = None
    labels = np.empty((len(ds),), dtype=np.int64)
    relpaths: List[str] = []
    offset = 0

    with torch.inference_mode():
        for images, y, batch_relpaths in tqdm(loader, desc=f"DINO {split}", ncols=120):
            images = images.to(args.device, non_blocking=True)
            amp_enabled = (not args.no_amp) and args.device.startswith("cuda")
            with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.float16 if args.device.startswith("cuda") else torch.bfloat16, enabled=amp_enabled):
                features = model.forward_features(images)
                cls = features["x_norm_clstoken"].float().cpu().numpy()
                patch = features["x_norm_patchtokens"].float().cpu().numpy()

            if cls_mm is None:
                n, d = len(ds), cls.shape[-1]
                p = patch.shape[1]
                cls_mm = np.memmap(paths["cls"], mode="w+", dtype=storage_dtype, shape=(n, d))
                patch_mm = np.memmap(paths["patch"], mode="w+", dtype=storage_dtype, shape=(n, p, d))
                meta = {
                    "N": n,
                    "D": d,
                    "P": p,
                    "dtype": "float32" if args.float32 else "float16",
                    "model": args.model,
                    "image_size": args.image_size,
                    "crop": "CUB ground-truth bbox, warped to square",
                    "split": split,
                }

            bs = cls.shape[0]
            cls_mm[offset : offset + bs] = cls.astype(storage_dtype, copy=False)
            patch_mm[offset : offset + bs] = patch.astype(storage_dtype, copy=False)
            labels[offset : offset + bs] = y.numpy()
            relpaths.extend(list(batch_relpaths))
            offset += bs

    assert offset == len(ds)
    cls_mm.flush()
    patch_mm.flush()
    np.save(paths["labels"], labels)
    with open(paths["relpaths"], "w", encoding="utf-8") as f:
        json.dump(relpaths, f, ensure_ascii=False)
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[{split}] wrote {len(ds)} samples to {os.path.dirname(paths['meta'])}")


def main() -> None:
    args = parse_args()
    args.cub_root = os.path.abspath(os.path.expanduser(args.cub_root))
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    if not os.path.isdir(args.cub_root):
        raise FileNotFoundError(args.cub_root)
    if args.image_size % 14 != 0:
        raise ValueError("DINOv2 ViT/14 requires --image-size divisible by 14.")

    print(f"Loading {args.model} from the official facebookresearch/dinov2 PyTorch Hub repository")
    model = torch.hub.load("facebookresearch/dinov2", args.model)
    model = model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    build_split(model, args, "train")
    build_split(model, args, "test")


if __name__ == "__main__":
    main()
