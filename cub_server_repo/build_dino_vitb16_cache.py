#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build CUB DINO ViT-B/16 feature cache compatible with the existing training scripts.

Outputs under <output_dir>/features:
  train_meta.json, train_cls.dat, train_patch.dat, train_labels.npy, train_relpaths.json
  test_meta.json,  test_cls.dat,  test_patch.dat,  test_labels.npy,  test_relpaths.json

Default model:
  torch.hub facebookresearch/dino:main dino_vitb16

For 224x224 images with patch size 16:
  patch grid = 14 x 14
  P = 196
  D = 768
"""

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

try:
    from torchvision import transforms
except Exception as e:
    raise RuntimeError("torchvision is required for preprocessing") from e


@dataclass
class CUBSample:
    image_id: int
    relpath: str
    label: int
    split: str
    bbox_xywh: Tuple[float, float, float, float]


def read_kv_txt(path: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, v = line.split(maxsplit=1)
            out[int(k)] = v
    return out


def read_kv_int_txt(path: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, v = line.split()
            out[int(k)] = int(v)
    return out


def read_bbox_txt(path: str) -> Dict[int, Tuple[float, float, float, float]]:
    out: Dict[int, Tuple[float, float, float, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            image_id = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            out[image_id] = (x, y, w, h)
    return out


def load_cub_samples(cub_root: str, split: str) -> List[CUBSample]:
    if split not in {"train", "test"}:
        raise ValueError(split)

    id2path = read_kv_txt(os.path.join(cub_root, "images.txt"))
    id2label = read_kv_int_txt(os.path.join(cub_root, "image_class_labels.txt"))
    id2train = read_kv_int_txt(os.path.join(cub_root, "train_test_split.txt"))
    id2bbox = read_bbox_txt(os.path.join(cub_root, "bounding_boxes.txt"))

    want_train = split == "train"
    samples: List[CUBSample] = []
    for image_id in sorted(id2path):
        is_train = id2train[image_id] == 1
        if is_train != want_train:
            continue
        samples.append(
            CUBSample(
                image_id=image_id,
                relpath=id2path[image_id],
                label=id2label[image_id] - 1,
                split=split,
                bbox_xywh=id2bbox[image_id],
            )
        )
    return samples


def crop_cub_bbox(image: Image.Image, bbox_xywh: Tuple[float, float, float, float]) -> Image.Image:
    x, y, w, h = bbox_xywh
    W, H = image.size
    x1 = int(math.floor(x))
    y1 = int(math.floor(y))
    x2 = int(math.ceil(x + w))
    y2 = int(math.ceil(y + h))
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(x1 + 1, min(W, x2))
    y2 = max(y1 + 1, min(H, y2))
    return image.crop((x1, y1, x2, y2))


class CUBImageDataset(Dataset):
    def __init__(
        self,
        cub_root: str,
        split: str,
        image_size: int,
        processed_image_root: str = "",
    ):
        self.cub_root = cub_root
        self.split = split
        self.image_size = int(image_size)
        self.processed_image_root = processed_image_root
        self.samples = load_cub_samples(cub_root, split)

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _load_processed(self, relpath: str) -> Optional[Image.Image]:
        if not self.processed_image_root:
            return None
        path = os.path.join(self.processed_image_root, "images", relpath)
        if not os.path.isfile(path):
            return None
        img = Image.open(path).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        return img

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = self._load_processed(s.relpath)
        if img is None:
            path = os.path.join(self.cub_root, "images", s.relpath)
            img = Image.open(path).convert("RGB")
            img = crop_cub_bbox(img, s.bbox_xywh)
            img = img.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        x = self.tf(img)
        return x, int(s.label), s.relpath


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    relpaths = [b[2] for b in batch]
    return imgs, labels, relpaths


@torch.inference_mode()
def infer_tokens(model: torch.nn.Module, images: torch.Tensor, amp: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    # DINO torch.hub ViT exposes get_intermediate_layers; returned tensor includes CLS + patch tokens.
    with torch.cuda.amp.autocast(enabled=(amp and images.is_cuda)):
        tokens = model.get_intermediate_layers(images, n=1)[0]
    tokens = tokens.float()
    cls = tokens[:, 0]
    patch = tokens[:, 1:]
    return cls, patch


def build_split(args, model: torch.nn.Module, split: str, device: str) -> None:
    ds = CUBImageDataset(
        cub_root=args.cub_root,
        split=split,
        image_size=args.image_size,
        processed_image_root=args.processed_image_root,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
        collate_fn=collate_fn,
    )

    feat_dir = os.path.join(args.output_dir, "features")
    os.makedirs(feat_dir, exist_ok=True)

    n = len(ds)
    d = 768
    p_expected = (args.image_size // args.patch_size) ** 2

    dtype_np = np.float16 if args.dtype == "float16" else np.float32
    cls_path = os.path.join(feat_dir, f"{split}_cls.dat")
    patch_path = os.path.join(feat_dir, f"{split}_patch.dat")
    labels_path = os.path.join(feat_dir, f"{split}_labels.npy")
    meta_path = os.path.join(feat_dir, f"{split}_meta.json")
    relpaths_path = os.path.join(feat_dir, f"{split}_relpaths.json")

    cls_mm = np.memmap(cls_path, mode="w+", dtype=dtype_np, shape=(n, d))
    patch_mm = np.memmap(patch_path, mode="w+", dtype=dtype_np, shape=(n, p_expected, d))
    labels_np = np.empty((n,), dtype=np.int64)
    relpaths_all: List[str] = []

    offset = 0
    pbar = tqdm(dl, desc=f"DINO ViT-B/16 {split}", ncols=120)
    for images, labels, relpaths in pbar:
        images = images.to(device, non_blocking=True)
        cls, patch = infer_tokens(model, images, amp=args.amp)
        if patch.shape[1] != p_expected:
            raise RuntimeError(
                f"Unexpected patch token count: got {patch.shape[1]}, expected {p_expected}. "
                f"image_size={args.image_size}, patch_size={args.patch_size}"
            )
        if cls.shape[1] != d:
            raise RuntimeError(f"Unexpected feature dim: got {cls.shape[1]}, expected {d}")

        b = images.shape[0]
        cls_mm[offset:offset+b] = cls.cpu().numpy().astype(dtype_np, copy=False)
        patch_mm[offset:offset+b] = patch.cpu().numpy().astype(dtype_np, copy=False)
        labels_np[offset:offset+b] = labels.numpy()
        relpaths_all.extend(relpaths)
        offset += b
        pbar.set_postfix({"written": offset})

    cls_mm.flush()
    patch_mm.flush()
    np.save(labels_path, labels_np)

    meta = {
        "N": int(n),
        "D": int(d),
        "P": int(p_expected),
        "grid": [int(args.image_size // args.patch_size), int(args.image_size // args.patch_size)],
        "dtype": str(args.dtype),
        "image_size": int(args.image_size),
        "patch_size": int(args.patch_size),
        "model_name": str(args.model_name),
        "model_repo": str(args.model_repo),
        "crop": "pre-exported GT-bbox warp" if args.processed_image_root else "CUB ground-truth bbox, warped to square",
        "processed_image_root": args.processed_image_root or None,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(relpaths_path, "w", encoding="utf-8") as f:
        json.dump(relpaths_all, f, ensure_ascii=False, indent=2)

    print(f"[{split}] saved:")
    print(f"  {cls_path}")
    print(f"  {patch_path}")
    print(f"  {labels_path}")
    print(f"  {meta_path}")
    print(f"  {relpaths_path}")


def parse_args():
    p = argparse.ArgumentParser("Build CUB DINO ViT-B/16 feature cache")
    p.add_argument("--cub-root", default="./data/CUB_200_2011")
    p.add_argument("--processed-image-root", default="", help="Optional root containing images/<CUB relpath>, e.g. CUB_200_2011_gtbbox_resize224")
    p.add_argument("--output-dir", default="./artifacts/dino_vitb16_gtbbox_warp224")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", help="Use CUDA autocast during feature extraction")
    p.add_argument("--model-repo", default="facebookresearch/dino:main")
    p.add_argument("--model-name", default="dino_vitb16")
    p.add_argument("--torch-hub-dir", default="", help="Optional torch hub cache dir")
    p.add_argument("--split", choices=["train", "test", "all"], default="all")
    return p.parse_args()


def main():
    args = parse_args()
    args.cub_root = os.path.abspath(os.path.expanduser(args.cub_root))
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    args.processed_image_root = os.path.abspath(os.path.expanduser(args.processed_image_root)) if args.processed_image_root else ""

    if args.torch_hub_dir:
        torch.hub.set_dir(os.path.abspath(os.path.expanduser(args.torch_hub_dir)))

    if args.image_size % args.patch_size != 0:
        raise ValueError(f"image_size must be divisible by patch_size: {args.image_size} vs {args.patch_size}")

    print(f"[Device] {args.device}; AMP={args.amp}")
    print(f"[Model] torch.hub.load({args.model_repo!r}, {args.model_name!r})")
    print(f"[Output] {args.output_dir}")

    model = torch.hub.load(args.model_repo, args.model_name)
    model.eval().to(args.device)

    splits = ["train", "test"] if args.split == "all" else [args.split]
    for split in splits:
        build_split(args, model, split, args.device)


if __name__ == "__main__":
    main()
