"""Full fine-tuning of OpenAI CLIP RN50x4 with a 200-way linear head on CUB, without augmentation.

Training:
    saved GT-bird crops only (no offline or online augmentation) -> direct Resize(288)
Evaluation:
    raw CUB test image -> official GT bird bbox crop -> direct Resize(288)

The complete RN50x4 visual tower, including its BatchNorm affine parameters and
running statistics, is fine-tuned together with a new 200-way linear classifier.
The text branch is intentionally not loaded into the trainable module. Under DDP,
SyncBatchNorm is used by default so running statistics are synchronized across GPUs.

Dependency (once, in the active environment):
    python -m pip install git+https://github.com/openai/CLIP.git
"""

from __future__ import annotations

import argparse
import json
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
    save_dir: str = "./runs/clip_rn50x4_fullft_linear_npplaug_all_resize288_8e"
    image_size: int = 288
    l2_normalize: bool = False
    clip_cache_dir: str = ""

    # Saved original GT-bird crops. This matches the original-image pixels used
    # by the previous NPPL-all linear baseline, but excludes every augmentation.
    train_crop_root: str = "./data/cub200_cropped_nppl_bboxsync/train_cropped"

    epochs: int = 8
    warmup_source_epochs: float = 5.0
    warmup_lr: float = 1e-6
    min_lr_scale: float = 0.01
    batch_size: int = 32  # per GPU under DDP; conservative for full RN50x4 fine-tuning
    num_workers: int = 8
    lr_encoder: float = 1e-5
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    sync_bn: bool = True
    grad_clip_norm: float = 1.0
    amp: bool = True
    seed: int = 42
    resume_from: str = "none"  # none | last | best | path


cfg = Config()


def parse_args() -> Config:
    p = argparse.ArgumentParser("Full fine-tuning of OpenAI CLIP RN50x4 on CUB without augmentation")
    p.add_argument("--cub-root", default=cfg.cub_root)
    p.add_argument("--save-dir", default=cfg.save_dir)
    p.add_argument("--image-size", type=int, default=cfg.image_size)
    p.add_argument("--l2-normalize", action=argparse.BooleanOptionalAction, default=cfg.l2_normalize)
    p.add_argument(
        "--clip-cache-dir",
        default=cfg.clip_cache_dir,
        help="Optional OpenAI CLIP checkpoint cache; leave empty to use the package default.",
    )

    p.add_argument(
        "--train-crop-root",
        default=cfg.train_crop_root,
        help="Directory of saved original GT-bird crops, preserving CUB relpaths.",
    )

    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--warmup-source-epochs", type=float, default=cfg.warmup_source_epochs)
    p.add_argument("--warmup-lr", type=float, default=cfg.warmup_lr)
    p.add_argument("--min-lr-scale", type=float, default=cfg.min_lr_scale)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)
    p.add_argument("--lr-encoder", type=float, default=cfg.lr_encoder)
    p.add_argument("--lr-head", type=float, default=cfg.lr_head)
    p.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    p.add_argument("--label-smoothing", type=float, default=cfg.label_smoothing)
    p.add_argument("--sync-bn", action=argparse.BooleanOptionalAction, default=cfg.sync_bn)
    p.add_argument("--grad-clip-norm", type=float, default=cfg.grad_clip_norm)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=cfg.amp)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--resume-from", default=cfg.resume_from, help="none | last | best | path")

    ns = p.parse_args()
    out = Config(**vars(ns))
    if out.epochs < 1:
        raise ValueError("--epochs must be positive.")
    if out.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if out.warmup_source_epochs < 0:
        raise ValueError("--warmup-source-epochs must be non-negative.")
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
        samples.append(
            {
                "path": os.path.join(cub_root, "images", image_paths[image_id]),
                "relpath": image_paths[image_id],
                "label": labels[image_id] - 1,
                "bbox": bboxes[image_id],
            }
        )
    return samples


class CUBBirdCropDataset(Dataset):
    """Raw CUB images cropped once by the official GT bird bbox, then resized."""

    def __init__(self, samples: Sequence[Dict[str, object]], transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        with Image.open(str(sample["path"])) as im:
            image = im.convert("RGB")
        image = crop_bbox(image, sample["bbox"])  # type: ignore[arg-type]
        return self.transform(image), int(sample["label"])


class SavedCUBBirdCropDataset(Dataset):
    """Saved original GT-bird crops only; no augmentation of any kind."""

    def __init__(self, samples: Sequence[Dict[str, object]], crop_root: str, transform):
        self.samples = list(samples)
        self.crop_root = os.path.abspath(crop_root)
        self.transform = transform
        self.num_sources = len(self.samples)

        if not os.path.isdir(self.crop_root):
            raise FileNotFoundError(f"Saved training crop directory not found: {self.crop_root}")

        missing = []
        for sample in self.samples:
            path = os.path.join(self.crop_root, str(sample["relpath"]))
            if not os.path.isfile(path):
                missing.append(path)
                if len(missing) >= 10:
                    break
        if missing:
            preview = "\n".join(f"  - {p}" for p in missing)
            raise FileNotFoundError(
                "Saved training crops are missing or do not preserve CUB relpaths:\n" + preview
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        path = os.path.join(self.crop_root, str(sample["relpath"]))
        with Image.open(path) as im:
            image = im.convert("RGB")
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
    transform = transforms.Compose(
        [
            transforms.Resize(
                (cfg.image_size, cfg.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )
    return transform, transform


def build_loaders() -> Tuple[DataLoader, DataLoader, DistributedSampler, SavedCUBBirdCropDataset]:
    train_transform, eval_transform = build_transforms()
    train_samples = build_cub_samples(cfg.cub_root, "train")
    test_samples = build_cub_samples(cfg.cub_root, "test")

    train_ds = SavedCUBBirdCropDataset(train_samples, cfg.train_crop_root, train_transform)
    test_ds = CUBBirdCropDataset(test_samples, eval_transform)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, drop_last=False
    )
    test_sampler: Sampler[int] = DistributedEvalSampler(test_ds)
    pin = DEVICE.type == "cuda"

    common = dict(
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=(cfg.num_workers > 0),
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler, drop_last=False, **common
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, sampler=test_sampler, drop_last=False, **common
    )

    rank0_print(
        f"[Data] train_original_only={len(train_ds):,}; test={len(test_ds)}; "
        f"train=saved GT-bird crops only (no augmentation), test=raw image->GT-bird crop; "
        f"train/eval=direct Resize({cfg.image_size}, {cfg.image_size})"
    )
    return train_loader, test_loader, train_sampler, train_ds


def _import_openai_clip():
    try:
        import clip  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenAI CLIP is required for RN50x4. Install it in the active environment with:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        ) from exc
    if not hasattr(clip, "load") or not hasattr(clip, "available_models"):
        raise ImportError(
            "The installed `clip` package is not OpenAI CLIP. Install the official package with:\n"
            "  pip uninstall -y clip && pip install git+https://github.com/openai/CLIP.git"
        )
    return clip


def prime_clip_cache() -> None:
    """Avoid concurrent first-time RN50x4 downloads from two DDP ranks."""
    if not distributed() or not main_process():
        return
    clip = _import_openai_clip()
    kwargs = {"name": "RN50x4", "device": "cpu", "jit": False}
    if cfg.clip_cache_dir:
        kwargs["download_root"] = cfg.clip_cache_dir
    model, _ = clip.load(**kwargs)
    del model


class FullCLIPRN50x4LinearClassifier(nn.Module):
    """Trainable OpenAI CLIP RN50x4 visual tower followed by a 200-way linear head."""

    def __init__(self, l2_normalize: bool):
        super().__init__()
        clip = _import_openai_clip()
        # Load on CPU first. OpenAI CLIP otherwise converts its GPU weights to fp16;
        # we keep a fp32 master model and use AMP for the forward/backward pass.
        kwargs = {"name": "RN50x4", "device": "cpu", "jit": False}
        if cfg.clip_cache_dir:
            kwargs["download_root"] = cfg.clip_cache_dir
        clip_model, _ = clip.load(**kwargs)
        clip_model = clip_model.float()

        if not hasattr(clip_model, "visual") or not hasattr(clip_model.visual, "output_dim"):
            raise RuntimeError("Loaded CLIP model does not expose the RN50x4 visual output dimension.")
        self.visual = clip_model.visual
        del clip_model  # exclude the unused text branch from DDP and the optimizer

        self.expected_image_size = int(self.visual.input_resolution)
        feature_dim = int(self.visual.output_dim)
        self.l2_normalize = bool(l2_normalize)
        self.head = nn.Linear(feature_dim, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.visual(images)
        features = features.float()
        if self.l2_normalize:
            features = F.normalize(features, dim=-1, eps=1e-8)
        return self.head(features)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    base = unwrap(model)
    return torch.optim.AdamW(
        [
            {"params": base.visual.parameters(), "lr": cfg.lr_encoder, "name": "encoder"},
            {"params": base.head.parameters(), "lr": cfg.lr_head, "name": "head"},
        ],
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )


def cosine_lr(base_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return cfg.warmup_lr
    decay_steps = max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (cfg.min_lr_scale + (1.0 - cfg.min_lr_scale) * cosine)


def set_step_lrs(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> Tuple[float, float]:
    enc_lr = cosine_lr(cfg.lr_encoder, step, total_steps, warmup_steps)
    head_lr = cosine_lr(cfg.lr_head, step, total_steps, warmup_steps)
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


def save_checkpoint(
    path: str,
    epoch: int,
    best_acc: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
) -> None:
    if not main_process():
        return
    payload = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler_state_dict(scaler),
        "config": asdict(cfg),
        "encoder": "OpenAI CLIP RN50x4 visual tower (full fine-tuning; no augmentation)",
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

    rank0_print("[Run] full fine-tuning OpenAI CLIP RN50x4 visual tower + 200-way linear head (no augmentation)")
    rank0_print(
        f"[Run] image_size={cfg.image_size}; l2_normalize={cfg.l2_normalize}; "
        f"global_batch={cfg.batch_size * WORLD_SIZE}; encoder=full-finetune; sync_bn={cfg.sync_bn}"
    )
    rank0_print(
        f"[Run] lr_encoder={cfg.lr_encoder:.2e}; lr_head={cfg.lr_head:.2e}; "
        f"warmup_source_epochs={cfg.warmup_source_epochs:g}@{cfg.warmup_lr:.2e}; epochs={cfg.epochs}"
    )

    # Rank 0 downloads/caches the model once before all ranks instantiate it.
    prime_clip_cache()
    barrier()

    train_loader, test_loader, train_sampler, train_dataset = build_loaders()
    model: nn.Module = FullCLIPRN50x4LinearClassifier(cfg.l2_normalize).to(DEVICE)
    if cfg.image_size != unwrap(model).expected_image_size:
        raise ValueError(
            f"RN50x4 expects image_size={unwrap(model).expected_image_size}, but got {cfg.image_size}. "
            "Use the model-native resolution for this baseline."
        )

    feature_dim = unwrap(model).head.in_features
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(
        f"[Model] RN50x4 visual embedding_dim={feature_dim}; trainable_params={trainable:,}; "
        f"BatchNorm={'SyncBatchNorm' if (distributed() and cfg.sync_bn) else 'local BatchNorm'}"
    )

    if distributed() and cfg.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(DEVICE)
    if distributed():
        model = DDP(
            model,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK,
            broadcast_buffers=False,
        )

    optimizer = build_optimizer(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    autocast_enabled = cfg.amp and DEVICE.type == "cuda"
    scaler = build_scaler(autocast_enabled)

    steps_per_epoch = len(train_loader)
    reference_source_steps = math.ceil(train_dataset.num_sources / max(1, cfg.batch_size * WORLD_SIZE))
    warmup_steps = int(math.ceil(cfg.warmup_source_epochs * reference_source_steps))
    total_steps = cfg.epochs * steps_per_epoch
    if warmup_steps >= total_steps:
        raise ValueError(
            f"Warm-up is too long: warmup_steps={warmup_steps} but total_steps={total_steps}. "
            "Increase --epochs or reduce --warmup-source-epochs."
        )
    rank0_print(
        f"[Schedule] steps_per_original_only_epoch={steps_per_epoch}; total_steps={total_steps}; "
        f"warmup_steps={warmup_steps} (={cfg.warmup_source_epochs:g} source-image epochs)"
    )

    start_epoch, best_acc = 1, -1.0
    resume_path = resolve_resume_path()
    if resume_path is not None:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        except TypeError:
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
        model.train()

        running_loss = 0.0
        running_correct = 0.0
        running_total = 0.0
        iterator = tqdm(
            train_loader,
            desc=f"Train {epoch}/{cfg.epochs}",
            disable=not main_process(),
            dynamic_ncols=True,
        )

        for batch_idx, (images, labels) in enumerate(iterator):
            global_step = (epoch - 1) * steps_per_epoch + batch_idx
            enc_lr, head_lr = set_step_lrs(optimizer, global_step, total_steps, warmup_steps)
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

        train_loss_sum, train_correct, train_total = reduce_values(
            running_loss, running_correct, running_total
        )
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
            save_checkpoint(
                os.path.join(cfg.save_dir, "best.pth"),
                epoch, best_acc, model, optimizer, scaler,
            )
            rank0_print(f"[Best] epoch={epoch} test_acc={best_acc:.4f}")

        save_checkpoint(
            os.path.join(cfg.save_dir, "last.pth"),
            epoch, best_acc, model, optimizer, scaler,
        )
        barrier()

    rank0_print(f"[Done] best_test_acc={best_acc:.4f}")
    cleanup()


if __name__ == "__main__":
    main()
