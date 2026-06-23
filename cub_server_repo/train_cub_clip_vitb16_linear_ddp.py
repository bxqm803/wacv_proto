#!/usr/bin/env python3
"""Frozen CLIP ViT-B/16 linear probing on CUB-200-2011.

Default protocol (chosen to match the current part-prototype experiments):
    raw image -> CUB GT bird bbox crop -> direct Resize(224,224) -> CLIP norm
    frozen OpenAI CLIP ViT-B/16 vision tower
    final CLS / pooler feature -> Linear(768, 200)

No data augmentation is used.  --feature patch_mean / patch_max are included
for controlled pooling ablations; all three settings remain frozen-backbone,
single-linear-head probes.

Launch, e.g.:
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train_cub_clip_vitb16_linear_ddp.py --cub-root ./data/CUB_200_2011 \
  --save-dir ./runs/clip_vitb16_linear_cls_birdcrop224 --feature cls \
  --image-size 224 --epochs 100 --batch-size 64 --num-workers 4
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm


CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
NUM_CLASSES = 200

RANK = 0
WORLD_SIZE = 1
LOCAL_RANK = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    cub_root: str = "./data/CUB_200_2011"
    save_dir: str = "./runs/clip_vitb16_linear_cls_birdcrop224"
    image_size: int = 224
    feature: str = "cls"                 # cls | patch_mean | patch_max
    use_bird_bbox: bool = True
    l2_normalize: bool = False
    epochs: int = 100
    batch_size: int = 64                 # per GPU under DDP
    num_workers: int = 4
    lr: float = 3e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    amp: bool = True
    seed: int = 42
    resume_from: str = "none"            # none | last | best | path


cfg = Config()


def parse_args() -> Config:
    p = argparse.ArgumentParser("Frozen CLIP ViT-B/16 linear probing on CUB-200-2011")
    p.add_argument("--cub-root", default=cfg.cub_root)
    p.add_argument("--save-dir", default=cfg.save_dir)
    p.add_argument("--image-size", type=int, default=cfg.image_size)
    p.add_argument("--feature", choices=["cls", "patch_mean", "patch_max"], default=cfg.feature)
    p.add_argument("--use-bird-bbox", action=argparse.BooleanOptionalAction, default=cfg.use_bird_bbox)
    p.add_argument("--l2-normalize", action=argparse.BooleanOptionalAction, default=cfg.l2_normalize)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    p.add_argument("--label-smoothing", type=float, default=cfg.label_smoothing)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=cfg.amp)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--resume-from", default=cfg.resume_from,
                   help="none | last | best | absolute/relative checkpoint path")
    ns = p.parse_args()
    return Config(**vars(ns))


def setup_distributed() -> None:
    global RANK, WORLD_SIZE, LOCAL_RANK, DEVICE
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
    if WORLD_SIZE > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested but CUDA is unavailable.")
        RANK = int(os.environ["RANK"])
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(LOCAL_RANK)
        DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
        dist.init_process_group(backend="nccl")
    else:
        RANK, LOCAL_RANK = 0, 0
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def distributed() -> bool:
    return WORLD_SIZE > 1 and dist.is_available() and dist.is_initialized()


def main_process() -> bool:
    return RANK == 0


def rank0_print(*args, **kwargs) -> None:
    if main_process():
        print(*args, **kwargs)


def barrier() -> None:
    if distributed():
        dist.barrier()


def cleanup() -> None:
    if distributed():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    seed = int(seed) + RANK
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_kv_text(path: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                key, value = line.split(maxsplit=1)
                out[int(key)] = value
    return out


def read_kv_int(path: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                key, value = line.split()
                out[int(key)] = int(value)
    return out


def read_bboxes(path: str) -> Dict[int, Tuple[float, float, float, float]]:
    out: Dict[int, Tuple[float, float, float, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.strip().split()
            if vals:
                out[int(vals[0])] = tuple(float(x) for x in vals[1:5])
    return out


def crop_bbox(img: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
    x, y, w, h = bbox
    width, height = img.size
    x1 = max(0, min(width - 1, int(np.floor(x))))
    y1 = max(0, min(height - 1, int(np.floor(y))))
    x2 = max(x1 + 1, min(width, int(np.ceil(x + w))))
    y2 = max(y1 + 1, min(height, int(np.ceil(y + h))))
    return img.crop((x1, y1, x2, y2))


def build_cub_samples(cub_root: str, split: str) -> List[Dict[str, object]]:
    image_paths = read_kv_text(os.path.join(cub_root, "images.txt"))
    labels = read_kv_int(os.path.join(cub_root, "image_class_labels.txt"))
    split_map = read_kv_int(os.path.join(cub_root, "train_test_split.txt"))
    bboxes = read_bboxes(os.path.join(cub_root, "bounding_boxes.txt"))
    want_train = split == "train"

    samples: List[Dict[str, object]] = []
    for image_id in sorted(image_paths):
        if (split_map[image_id] == 1) != want_train:
            continue
        samples.append({
            "path": os.path.join(cub_root, "images", image_paths[image_id]),
            "label": labels[image_id] - 1,
            "bbox": bboxes[image_id],
        })
    return samples


class CUBCropDataset(Dataset):
    def __init__(self, samples: Sequence[Dict[str, object]], transform, use_bird_bbox: bool):
        self.samples = list(samples)
        self.transform = transform
        self.use_bird_bbox = bool(use_bird_bbox)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        with Image.open(str(sample["path"])) as im:
            image = im.convert("RGB")
        if self.use_bird_bbox:
            image = crop_bbox(image, sample["bbox"])  # type: ignore[arg-type]
        return self.transform(image), int(sample["label"])


class DistributedEvalSampler(Sampler[int]):
    """Non-padding distributed sampler so test metrics are exact."""
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __iter__(self) -> Iterator[int]:
        return iter(range(RANK, len(self.dataset), WORLD_SIZE))

    def __len__(self) -> int:
        return (len(self.dataset) + WORLD_SIZE - 1 - RANK) // WORLD_SIZE


def build_loaders() -> Tuple[DataLoader, DataLoader, DistributedSampler]:
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    train_ds = CUBCropDataset(build_cub_samples(cfg.cub_root, "train"), transform, cfg.use_bird_bbox)
    test_ds = CUBCropDataset(build_cub_samples(cfg.cub_root, "test"), transform, cfg.use_bird_bbox)

    train_sampler = DistributedSampler(train_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, drop_last=False)
    test_sampler: Sampler[int] = DistributedEvalSampler(test_ds)
    pin = DEVICE.type == "cuda"
    common = dict(num_workers=cfg.num_workers, pin_memory=pin,
                  persistent_workers=(cfg.num_workers > 0))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=train_sampler, drop_last=False, **common)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, sampler=test_sampler, drop_last=False, **common)
    if main_process():
        print(f"[Data] train={len(train_ds)} test={len(test_ds)} bbox_crop={cfg.use_bird_bbox} direct_resize={cfg.image_size}")
    return train_loader, test_loader, train_sampler


def load_frozen_clip() -> nn.Module:
    try:
        from transformers import CLIPVisionModel
    except ImportError as e:
        raise ImportError("This script requires transformers: pip install transformers") from e
    encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    return encoder


def extract_features(encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=(cfg.amp and DEVICE.type == "cuda")):
            out = encoder(pixel_values=images, return_dict=True)
            if cfg.feature == "cls":
                # CLIPVisionModel pooler_output = final CLS after post_layernorm.
                feats = out.pooler_output
            else:
                patches = out.last_hidden_state[:, 1:, :]
                feats = patches.mean(dim=1) if cfg.feature == "patch_mean" else patches.amax(dim=1)
    feats = feats.float()
    if cfg.l2_normalize:
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return feats


def reduce_pair(correct: float, total: float) -> Tuple[float, float]:
    x = torch.tensor([correct, total], device=DEVICE, dtype=torch.float64)
    if distributed():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return float(x[0].item()), float(x[1].item())


def evaluate(encoder: nn.Module, head: nn.Module, loader: DataLoader) -> float:
    head.eval()
    encoder.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            feats = extract_features(encoder, images)
            logits = head(feats)
            correct += float((logits.argmax(dim=1) == labels).sum().item())
            total += float(labels.numel())
    correct, total = reduce_pair(correct, total)
    return correct / max(1.0, total)


def resolve_resume_path() -> Optional[str]:
    value = str(cfg.resume_from)
    if value.lower() in {"", "none", "no"}:
        return None
    if value == "last":
        return os.path.join(cfg.save_dir, "last.pth")
    if value == "best":
        return os.path.join(cfg.save_dir, "best.pth")
    return value


def save_checkpoint(path: str, epoch: int, best_acc: float, head: nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler) -> None:
    if not main_process():
        return
    module = head.module if isinstance(head, DDP) else head
    torch.save({
        "epoch": epoch,
        "best_acc": best_acc,
        "head": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": asdict(cfg),
    }, path)


def main() -> None:
    global cfg
    cfg = parse_args()
    setup_distributed()
    set_seed(cfg.seed)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if main_process():
        os.makedirs(cfg.save_dir, exist_ok=True)
    barrier()

    rank0_print("[Probe] frozen CLIP ViT-B/16 + single linear head")
    rank0_print(f"[Probe] feature={cfg.feature}; l2_normalize={cfg.l2_normalize}; global_batch={cfg.batch_size * WORLD_SIZE}")
    rank0_print("[Input] bird bbox crop -> direct Resize -> CLIP normalize; no augmentation")

    train_loader, test_loader, train_sampler = build_loaders()
    encoder = load_frozen_clip()
    hidden_dim = int(encoder.config.hidden_size)
    head: nn.Module = nn.Linear(hidden_dim, NUM_CLASSES).to(DEVICE)
    if distributed():
        head = DDP(head, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, broadcast_buffers=False)

    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs), eta_min=cfg.lr * 0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    start_epoch, best_acc = 1, -1.0
    resume_path = resolve_resume_path()
    if resume_path is not None:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")
        module = head.module if isinstance(head, DDP) else head
        module.load_state_dict(checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_acc = float(checkpoint.get("best_acc", -1.0))
        rank0_print(f"[Resume] {resume_path}; next_epoch={start_epoch}; best={best_acc:.4f}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_sampler.set_epoch(epoch)
        encoder.eval()
        head.train()
        running_loss = 0.0
        running_correct = 0.0
        running_total = 0.0
        iterator = tqdm(train_loader, desc=f"Train {epoch}/{cfg.epochs}", disable=not main_process(), dynamic_ncols=True)
        for images, labels in iterator:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            feats = extract_features(encoder, images)
            logits = head(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_n = labels.numel()
            running_loss += float(loss.detach().item()) * batch_n
            running_correct += float((logits.detach().argmax(dim=1) == labels).sum().item())
            running_total += float(batch_n)
            if main_process():
                iterator.set_postfix(loss=f"{running_loss / max(1.0, running_total):.4f}",
                                     acc=f"{running_correct / max(1.0, running_total):.4f}")

        scheduler.step()
        train_correct, train_total = reduce_pair(running_correct, running_total)
        train_loss_sum, _ = reduce_pair(running_loss, running_total)
        train_acc = train_correct / max(1.0, train_total)
        train_loss = train_loss_sum / max(1.0, train_total)
        test_acc = evaluate(encoder, head, test_loader)

        if main_process():
            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"test_acc={test_acc:.4f} lr={optimizer.param_groups[0]['lr']:.3e}")
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(os.path.join(cfg.save_dir, "best.pth"), epoch, best_acc, head, optimizer, scheduler)
            rank0_print(f"[Best] epoch={epoch} test_acc={best_acc:.4f}")
        save_checkpoint(os.path.join(cfg.save_dir, "last.pth"), epoch, best_acc, head, optimizer, scheduler)
        barrier()

    rank0_print(f"[Done] best_test_acc={best_acc:.4f}")
    cleanup()


if __name__ == "__main__":
    main()
