#!/usr/bin/env python3
"""End-to-end CLIP ViT-B/16 fine-tuning on CUB-200-2011.

PRISM-comparable protocol defaults:
    full CUB image -> 224x224 CLIP input -> CLIP ViT-B/16 vision encoder
    all vision parameters are trainable (lr=1e-5)
    768-d pooled CLS feature -> Linear(768, 200) (lr=1e-3)
    AdamW + 5-epoch 1e-6 warm-up + cosine decay, 300 epochs

This is a clean visual baseline, not an implementation of PRISM. PRISM does
not disclose its baseline classifier head or augmentation details, so the
result should be reported as a reproduced CLIP ViT-B/16 fine-tuning baseline.

Example (two GPUs):
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train_cub_clip_vitb16_finetune_ddp.py \
  --cub-root ./data/CUB_200_2011 \
  --save-dir ./runs/clip_vitb16_fullimg_finetune_300e \
  --epochs 300 --batch-size 128 --num-workers 8
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
import torch.nn.functional as F
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
    save_dir: str = "./runs/clip_vitb16_fullimg_finetune_300e"
    model_name: str = "openai/clip-vit-base-patch16"
    image_size: int = 224
    eval_resize: int = 256
    feature: str = "cls"                 # cls | patch_mean | patch_max
    use_bird_bbox: bool = False           # PRISM uses full, uncropped images.
    l2_normalize: bool = False
    train_aug: str = "rrc"               # rrc | none
    rrc_scale_min: float = 0.70
    epochs: int = 300
    warmup_epochs: int = 5
    warmup_lr: float = 1e-6
    min_lr_scale: float = 0.01
    batch_size: int = 128                 # per GPU under DDP
    num_workers: int = 8
    lr_encoder: float = 1e-5
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    grad_clip_norm: float = 0.0
    amp: bool = True
    seed: int = 42
    resume_from: str = "none"            # none | last | best | path


cfg = Config()


def parse_args() -> Config:
    p = argparse.ArgumentParser("End-to-end CLIP ViT-B/16 fine-tuning on CUB-200-2011")
    p.add_argument("--cub-root", default=cfg.cub_root)
    p.add_argument("--save-dir", default=cfg.save_dir)
    p.add_argument("--model-name", default=cfg.model_name)
    p.add_argument("--image-size", type=int, default=cfg.image_size)
    p.add_argument("--eval-resize", type=int, default=cfg.eval_resize)
    p.add_argument("--feature", choices=["cls", "patch_mean", "patch_max"], default=cfg.feature)
    p.add_argument("--use-bird-bbox", action=argparse.BooleanOptionalAction, default=cfg.use_bird_bbox)
    p.add_argument("--l2-normalize", action=argparse.BooleanOptionalAction, default=cfg.l2_normalize)
    p.add_argument("--train-aug", choices=["rrc", "none"], default=cfg.train_aug)
    p.add_argument("--rrc-scale-min", type=float, default=cfg.rrc_scale_min)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--warmup-epochs", type=int, default=cfg.warmup_epochs)
    p.add_argument("--warmup-lr", type=float, default=cfg.warmup_lr)
    p.add_argument("--min-lr-scale", type=float, default=cfg.min_lr_scale)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)
    p.add_argument("--lr-encoder", type=float, default=cfg.lr_encoder)
    p.add_argument("--lr-head", type=float, default=cfg.lr_head)
    p.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    p.add_argument("--label-smoothing", type=float, default=cfg.label_smoothing)
    p.add_argument("--grad-clip-norm", type=float, default=cfg.grad_clip_norm)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=cfg.amp)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--resume-from", default=cfg.resume_from,
                   help="none | last | best | absolute/relative checkpoint path")
    ns = p.parse_args()
    out = Config(**vars(ns))
    if not (0.0 < out.rrc_scale_min <= 1.0):
        raise ValueError("--rrc-scale-min must be in (0, 1].")
    if out.epochs < 1:
        raise ValueError("--epochs must be positive.")
    if out.warmup_epochs < 0 or out.warmup_epochs >= out.epochs:
        raise ValueError("--warmup-epochs must be >= 0 and < --epochs.")
    return out


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
    # Separate process RNG streams, while DistributedSampler owns global shuffling.
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


def build_transforms():
    norm = transforms.Normalize(CLIP_MEAN, CLIP_STD)
    if cfg.train_aug == "rrc":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                cfg.image_size,
                scale=(cfg.rrc_scale_min, 1.0),
                ratio=(0.75, 4.0 / 3.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            norm,
        ])

    eval_transform = transforms.Compose([
        transforms.Resize(cfg.eval_resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        norm,
    ])
    return train_transform, eval_transform


def build_loaders() -> Tuple[DataLoader, DataLoader, DistributedSampler]:
    train_transform, eval_transform = build_transforms()
    train_ds = CUBCropDataset(build_cub_samples(cfg.cub_root, "train"), train_transform, cfg.use_bird_bbox)
    test_ds = CUBCropDataset(build_cub_samples(cfg.cub_root, "test"), eval_transform, cfg.use_bird_bbox)

    train_sampler = DistributedSampler(train_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, drop_last=False)
    test_sampler: Sampler[int] = DistributedEvalSampler(test_ds)
    pin = DEVICE.type == "cuda"
    common = dict(num_workers=cfg.num_workers, pin_memory=pin,
                  persistent_workers=(cfg.num_workers > 0))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=train_sampler, drop_last=False, **common)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, sampler=test_sampler, drop_last=False, **common)
    rank0_print(
        f"[Data] train={len(train_ds)} test={len(test_ds)} bbox_crop={cfg.use_bird_bbox}; "
        f"train_aug={cfg.train_aug}; eval=Resize({cfg.eval_resize})->CenterCrop({cfg.image_size})"
    )
    return train_loader, test_loader, train_sampler


class CLIPVisionLinearClassifier(nn.Module):
    def __init__(self, model_name: str, feature: str, l2_normalize: bool):
        super().__init__()
        try:
            from transformers import CLIPVisionModel
        except ImportError as e:
            raise ImportError("This script requires transformers: pip install transformers") from e

        self.encoder = CLIPVisionModel.from_pretrained(model_name)
        self.feature = feature
        self.l2_normalize = bool(l2_normalize)
        hidden_dim = int(self.encoder.config.hidden_size)
        self.head = nn.Linear(hidden_dim, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out = self.encoder(pixel_values=images, return_dict=True)
        if self.feature == "cls":
            feats = out.pooler_output
        else:
            patches = out.last_hidden_state[:, 1:, :]
            feats = patches.mean(dim=1) if self.feature == "patch_mean" else patches.amax(dim=1)
        if self.l2_normalize:
            feats = F.normalize(feats, dim=-1, eps=1e-8)
        return self.head(feats)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    base = unwrap(model)
    return torch.optim.AdamW(
        [
            {"params": base.encoder.parameters(), "lr": cfg.lr_encoder, "name": "encoder"},
            {"params": base.head.parameters(), "lr": cfg.lr_head, "name": "head"},
        ],
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )


def cosine_lr(base_lr: float, epoch: int) -> float:
    """Epoch-indexed LR. First warmup_epochs use exactly warmup_lr."""
    if epoch <= cfg.warmup_epochs:
        return cfg.warmup_lr
    # First post-warmup epoch starts at the requested base learning rate.
    total_decay_epochs = max(1, cfg.epochs - cfg.warmup_epochs - 1)
    progress = min(1.0, (epoch - cfg.warmup_epochs - 1) / total_decay_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (cfg.min_lr_scale + (1.0 - cfg.min_lr_scale) * cosine)


def set_epoch_lrs(optimizer: torch.optim.Optimizer, epoch: int) -> Tuple[float, float]:
    enc_lr = cosine_lr(cfg.lr_encoder, epoch)
    head_lr = cosine_lr(cfg.lr_head, epoch)
    optimizer.param_groups[0]["lr"] = enc_lr
    optimizer.param_groups[1]["lr"] = head_lr
    return enc_lr, head_lr


def reduce_values(*values: float) -> List[float]:
    x = torch.tensor(values, device=DEVICE, dtype=torch.float64)
    if distributed():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return [float(v) for v in x.tolist()]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0.0
    total = 0.0
    autocast_enabled = cfg.amp and DEVICE.type == "cuda"
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=autocast_enabled):
            logits = model(images)
        correct += float((logits.argmax(dim=1) == labels).sum().item())
        total += float(labels.numel())
    correct, total = reduce_values(correct, total)
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


def scaler_state_dict(scaler):
    return scaler.state_dict() if scaler.is_enabled() else None


def save_checkpoint(path: str, epoch: int, best_acc: float, model: nn.Module,
                    optimizer: torch.optim.Optimizer, scaler) -> None:
    if not main_process():
        return
    payload = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler_state_dict(scaler),
        "config": asdict(cfg),
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def build_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def main() -> None:
    global cfg
    cfg = parse_args()
    setup_distributed()
    set_seed(cfg.seed)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if main_process():
        os.makedirs(cfg.save_dir, exist_ok=True)
    barrier()

    rank0_print("[Run] end-to-end CLIP ViT-B/16 fine-tuning + linear head")
    rank0_print(
        f"[Run] model={cfg.model_name}; feature={cfg.feature}; l2_normalize={cfg.l2_normalize}; "
        f"global_batch={cfg.batch_size * WORLD_SIZE}"
    )
    rank0_print(
        f"[Run] lr_encoder={cfg.lr_encoder:.2e}; lr_head={cfg.lr_head:.2e}; "
        f"warmup={cfg.warmup_epochs}@{cfg.warmup_lr:.2e}; epochs={cfg.epochs}"
    )

    train_loader, test_loader, train_sampler = build_loaders()
    model: nn.Module = CLIPVisionLinearClassifier(cfg.model_name, cfg.feature, cfg.l2_normalize).to(DEVICE)
    if distributed():
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, broadcast_buffers=False)

    optimizer = build_optimizer(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    autocast_enabled = cfg.amp and DEVICE.type == "cuda"
    scaler = build_scaler(autocast_enabled)

    start_epoch, best_acc = 1, -1.0
    resume_path = resolve_resume_path()
    if resume_path is not None:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")
        unwrap(model).load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_acc = float(checkpoint.get("best_acc", -1.0))
        rank0_print(f"[Resume] {resume_path}; next_epoch={start_epoch}; best={best_acc:.4f}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_sampler.set_epoch(epoch)
        enc_lr, head_lr = set_epoch_lrs(optimizer, epoch)
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        running_total = 0.0

        iterator = tqdm(train_loader, desc=f"Train {epoch}/{cfg.epochs}", disable=not main_process(), dynamic_ncols=True)
        for images, labels in iterator:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=autocast_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            if cfg.grad_clip_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            batch_n = labels.numel()
            running_loss += float(loss.detach().item()) * batch_n
            running_correct += float((logits.detach().argmax(dim=1) == labels).sum().item())
            running_total += float(batch_n)
            if main_process():
                iterator.set_postfix(
                    loss=f"{running_loss / max(1.0, running_total):.4f}",
                    acc=f"{running_correct / max(1.0, running_total):.4f}",
                    enc_lr=f"{enc_lr:.1e}",
                    head_lr=f"{head_lr:.1e}",
                )

        train_loss_sum, train_correct, train_total = reduce_values(running_loss, running_correct, running_total)
        train_loss = train_loss_sum / max(1.0, train_total)
        train_acc = train_correct / max(1.0, train_total)
        test_acc = evaluate(model, test_loader)

        if main_process():
            print(
                f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"test_acc={test_acc:.4f} enc_lr={enc_lr:.3e} head_lr={head_lr:.3e}"
            )
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(os.path.join(cfg.save_dir, "best.pth"), epoch, best_acc, model, optimizer, scaler)
            rank0_print(f"[Best] epoch={epoch} test_acc={best_acc:.4f}")
        save_checkpoint(os.path.join(cfg.save_dir, "last.pth"), epoch, best_acc, model, optimizer, scaler)
        barrier()

    rank0_print(f"[Done] best_test_acc={best_acc:.4f}")
    cleanup()


if __name__ == "__main__":
    main()
