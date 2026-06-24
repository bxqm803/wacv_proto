#!/usr/bin/env python3
"""DINOv2 ViT-B/14 last-two-block fine-tuning with a single linear head on Stanford Dogs.

Expected official Stanford Dogs layout:
  <dogs-root>/
    Images/
    Annotation/
    train_list.mat
    test_list.mat

Default protocol:
  official dog bbox crop -> direct Resize(224,224) -> DINOv2 ViT-B/14
  with only the final two transformer blocks trainable -> 120-way single linear classifier

No data augmentation is applied by default. Use --crop-mode full for full images
instead of official bbox crops.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision import transforms
from tqdm import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class CFG:
    dogs_root: str = "./data/StanfordDogs"
    save_dir: str = "./runs/dinov2_vitb14_stanforddogs_bbox_last2_linear"
    dino_model: str = "dinov2_vitb14"
    image_size: int = 224
    crop_mode: str = "bbox"  # bbox | full

    num_classes: int = 120
    epochs: int = 50
    batch_size: int = 64
    num_workers: int = 8
    seed: int = 42
    amp: bool = True

    lr_backbone: float = 1e-5
    unfreeze_last_blocks: int = 2
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    min_lr_scale: float = 0.01
    label_smoothing: float = 0.0
    grad_clip_norm: float = 1.0

    resume: bool = True
    resume_from: str = "last"  # last | best | none | path
    reset_optimizer_on_resume: bool = False

    eval_every: int = 1
    save_every: int = 1


cfg = CFG()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANK = 0
WORLD_SIZE = 1
LOCAL_RANK = 0


# -----------------------------------------------------------------------------
# Distributed / utility
# -----------------------------------------------------------------------------
def setup_distributed() -> None:
    global DEVICE, RANK, WORLD_SIZE, LOCAL_RANK
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
        RANK = 0
        LOCAL_RANK = 0
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_distributed() -> bool:
    return WORLD_SIZE > 1 and dist.is_available() and dist.is_initialized()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    return RANK == 0


def rank0_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    full_seed = int(seed) + RANK
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(full_seed)


def safe_torch_load(path: str, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def crop_bbox(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    xmin, ymin, xmax, ymax = bbox
    width, height = image.size
    xmin = max(0, min(width - 1, int(xmin)))
    ymin = max(0, min(height - 1, int(ymin)))
    xmax = max(xmin + 1, min(width, int(xmax)))
    ymax = max(ymin + 1, min(height, int(ymax)))
    return image.crop((xmin, ymin, xmax, ymax))


# -----------------------------------------------------------------------------
# Stanford Dogs metadata
# -----------------------------------------------------------------------------
def matlab_string(value: Any) -> str:
    """Robustly unwrap MATLAB string cells loaded through scipy.io.loadmat."""
    current = value
    while isinstance(current, np.ndarray):
        if current.size != 1:
            current = current.flat[0]
        else:
            current = current.item()
    if isinstance(current, bytes):
        return current.decode("utf-8")
    return str(current)


def parse_split_mat(path: str) -> List[Tuple[str, int]]:
    data = loadmat(path)
    if "annotation_list" not in data or "labels" not in data:
        raise KeyError(f"{path} must contain annotation_list and labels.")

    annotations = data["annotation_list"].squeeze()
    labels = np.asarray(data["labels"]).squeeze()

    if annotations.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Split file mismatch in {path}: {annotations.shape[0]} annotations vs {labels.shape[0]} labels."
        )

    samples: List[Tuple[str, int]] = []
    for ann, label in zip(annotations, labels):
        rel_annotation = matlab_string(ann).replace("\\", "/")
        label0 = int(np.asarray(label).item()) - 1
        samples.append((rel_annotation, label0))
    return samples


def parse_annotation_union_box(path: str) -> Tuple[int, int, int, int]:
    """Use the union of all object boxes, avoiding arbitrary object selection."""
    root = ET.parse(path).getroot()
    boxes: List[Tuple[int, int, int, int]] = []

    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = int(float(box.findtext("xmin", "0")))
        ymin = int(float(box.findtext("ymin", "0")))
        xmax = int(float(box.findtext("xmax", "0")))
        ymax = int(float(box.findtext("ymax", "0")))
        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))

    if not boxes:
        raise RuntimeError(f"No valid bounding box found in annotation: {path}")

    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


class StanfordDogsDataset(Dataset):
    def __init__(self, root: str, split: str, crop_mode: str, transform):
        if split not in {"train", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if crop_mode not in {"bbox", "full"}:
            raise ValueError(f"crop_mode must be bbox or full, got {crop_mode!r}")

        self.root = os.path.abspath(root)
        self.images_root = os.path.join(self.root, "Images")
        self.annotations_root = os.path.join(self.root, "Annotation")
        mat_path = os.path.join(self.root, f"{split}_list.mat")

        if not os.path.isdir(self.images_root):
            raise FileNotFoundError(f"Missing Images directory: {self.images_root}")
        if crop_mode == "bbox" and not os.path.isdir(self.annotations_root):
            raise FileNotFoundError(f"Missing Annotation directory: {self.annotations_root}")
        if not os.path.isfile(mat_path):
            raise FileNotFoundError(f"Missing split file: {mat_path}")

        split_items = parse_split_mat(mat_path)
        self.samples: List[Dict[str, Any]] = []

        for rel_annotation, label in split_items:
            image_path = os.path.join(self.images_root, rel_annotation + ".jpg")
            annotation_path = os.path.join(self.annotations_root, rel_annotation)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image referenced by {mat_path} is missing: {image_path}")

            bbox = None
            if crop_mode == "bbox":
                if not os.path.isfile(annotation_path):
                    raise FileNotFoundError(
                        f"Annotation referenced by {mat_path} is missing: {annotation_path}"
                    )
                bbox = parse_annotation_union_box(annotation_path)

            self.samples.append(
                {
                    "image_path": image_path,
                    "annotation_path": annotation_path,
                    "label": label,
                    "bbox": bbox,
                    "rel_annotation": rel_annotation,
                }
            )

        self.transform = transform
        self.crop_mode = crop_mode

        labels = [sample["label"] for sample in self.samples]
        if min(labels) < 0 or max(labels) >= cfg.num_classes:
            raise RuntimeError(
                f"Labels out of expected 0..{cfg.num_classes - 1} range: "
                f"min={min(labels)}, max={max(labels)}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample["image_path"]) as image:
            image = image.convert("RGB")

        if self.crop_mode == "bbox":
            image = crop_bbox(image, sample["bbox"])

        return self.transform(image), int(sample["label"])


class DistributedEvalSampler(Sampler[int]):
    """Distributed test sampler with no padding and no duplicate samples."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __iter__(self) -> Iterator[int]:
        return iter(range(RANK, len(self.dataset), WORLD_SIZE))

    def __len__(self) -> int:
        return (len(self.dataset) + WORLD_SIZE - 1 - RANK) // WORLD_SIZE


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class DINOv2LinearClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        if not isinstance(features, dict) or "x_norm_clstoken" not in features:
            raise RuntimeError("DINOv2 forward_features did not return x_norm_clstoken.")
        cls = features["x_norm_clstoken"].float()
        return self.head(cls)


def load_dinov2_model() -> nn.Module:
    backbone = torch.hub.load("facebookresearch/dinov2", cfg.dino_model)
    if not hasattr(backbone, "embed_dim"):
        raise RuntimeError(f"Could not infer embed_dim from {cfg.dino_model}.")
    return backbone


# -----------------------------------------------------------------------------
# Optimization
# -----------------------------------------------------------------------------
def configure_last_blocks_trainable(backbone: nn.Module, num_last_blocks: int) -> List[str]:
    """Freeze the whole DINOv2 encoder, then unfreeze only its final N transformer blocks."""
    if not hasattr(backbone, "blocks"):
        raise AttributeError("DINOv2 backbone does not expose a .blocks ModuleList.")

    all_blocks = list(backbone.blocks)
    if num_last_blocks < 1 or num_last_blocks > len(all_blocks):
        raise ValueError(
            f"--unfreeze-last-blocks must be in [1, {len(all_blocks)}], got {num_last_blocks}."
        )

    for parameter in backbone.parameters():
        parameter.requires_grad = False

    selected = all_blocks[-num_last_blocks:]
    for block in selected:
        for parameter in block.parameters():
            parameter.requires_grad = True

    trainable_names = [
        name for name, parameter in backbone.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names:
        raise RuntimeError("No DINOv2 backbone parameters were unfrozen.")

    return trainable_names


def build_optimizer(model: DINOv2LinearClassifier) -> torch.optim.Optimizer:
    trainable_backbone = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    if not trainable_backbone:
        raise RuntimeError("Optimizer received no trainable backbone parameters.")

    return torch.optim.AdamW(
        [
            {
                "params": trainable_backbone,
                "lr": cfg.lr_backbone,
                "name": "backbone",
            },
            {
                "params": list(model.head.parameters()),
                "lr": cfg.lr_head,
                "name": "head",
            },
        ],
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )


def cosine_warmup_lambda(epoch_fraction: float) -> float:
    warmup = max(0.0, float(cfg.warmup_epochs))
    total = max(1.0, float(cfg.epochs))

    if warmup > 0 and epoch_fraction < warmup:
        return max(1e-8, epoch_fraction / warmup)

    progress = (epoch_fraction - warmup) / max(1e-8, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return cfg.min_lr_scale + (1.0 - cfg.min_lr_scale) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def build_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    autocast_enabled = cfg.amp and DEVICE.type == "cuda"

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=autocast_enabled,
        ):
            logits = model(images)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())

    if is_distributed():
        stats = torch.tensor([correct, total], device=DEVICE, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        correct, total = int(stats[0].item()), int(stats[1].item())

    return correct / max(1, total)


def resolve_resume_path() -> Optional[str]:
    value = str(cfg.resume_from).strip()
    if not cfg.resume or value.lower() in {"", "none", "no"}:
        return None
    if value == "last":
        return os.path.join(cfg.save_dir, "last.pth")
    if value == "best":
        return os.path.join(cfg.save_dir, "best.pth")
    return os.path.abspath(os.path.expanduser(value))


def save_checkpoint(
    path: str,
    epoch: int,
    best_acc: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
) -> None:
    if not is_main_process():
        return

    core = unwrap_model(model)
    payload = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model": core.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "config": asdict(cfg),
    }

    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


# -----------------------------------------------------------------------------
# Args and main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "DINOv2 ViT-B/14 last-two-block fine-tuning with a single linear head on Stanford Dogs"
    )

    parser.add_argument("--dogs-root", default=cfg.dogs_root)
    parser.add_argument("--save-dir", default=cfg.save_dir)
    parser.add_argument("--dino-model", default=cfg.dino_model)
    parser.add_argument("--image-size", type=int, default=cfg.image_size)
    parser.add_argument("--crop-mode", choices=["bbox", "full"], default=cfg.crop_mode)

    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=cfg.amp)

    parser.add_argument("--lr-backbone", type=float, default=cfg.lr_backbone)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=cfg.unfreeze_last_blocks)
    parser.add_argument("--lr-head", type=float, default=cfg.lr_head)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--warmup-epochs", type=int, default=cfg.warmup_epochs)
    parser.add_argument("--min-lr-scale", type=float, default=cfg.min_lr_scale)
    parser.add_argument("--label-smoothing", type=float, default=cfg.label_smoothing)
    parser.add_argument("--grad-clip-norm", type=float, default=cfg.grad_clip_norm)

    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=cfg.resume)
    parser.add_argument("--resume-from", default=cfg.resume_from)
    parser.add_argument("--reset-optimizer-on-resume", action="store_true")

    parser.add_argument("--eval-every", type=int, default=cfg.eval_every)
    parser.add_argument("--save-every", type=int, default=cfg.save_every)

    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    global cfg
    for key, value in vars(args).items():
        setattr(cfg, key, value)

    cfg.dogs_root = os.path.abspath(os.path.expanduser(cfg.dogs_root))
    cfg.save_dir = os.path.abspath(os.path.expanduser(cfg.save_dir))


def preflight() -> None:
    expected = [
        os.path.join(cfg.dogs_root, "Images"),
        os.path.join(cfg.dogs_root, "train_list.mat"),
        os.path.join(cfg.dogs_root, "test_list.mat"),
    ]
    if cfg.crop_mode == "bbox":
        expected.append(os.path.join(cfg.dogs_root, "Annotation"))

    missing = [path for path in expected if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Stanford Dogs files are missing:\n" + "\n".join(f"  - {path}" for path in missing)
        )


def main() -> None:
    args = parse_args()
    apply_args(args)
    setup_distributed()
    preflight()
    set_seed(cfg.seed)

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if is_main_process():
        ensure_dir(cfg.save_dir)
    barrier()

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (cfg.image_size, cfg.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                (cfg.image_size, cfg.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_set = StanfordDogsDataset(
        cfg.dogs_root,
        "train",
        cfg.crop_mode,
        train_transform,
    )
    test_set = StanfordDogsDataset(
        cfg.dogs_root,
        "test",
        cfg.crop_mode,
        eval_transform,
    )

    train_sampler = DistributedSampler(
        train_set,
        num_replicas=WORLD_SIZE,
        rank=RANK,
        shuffle=True,
        drop_last=False,
    )
    test_sampler: Sampler[int] = DistributedEvalSampler(test_set)

    loader_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        sampler=test_sampler,
        drop_last=False,
        **loader_kwargs,
    )

    rank0_print(
        f"[Run] DINOv2 {cfg.dino_model} last-{cfg.unfreeze_last_blocks}-block fine-tuning "
        f"+ single linear head on Stanford Dogs | crop_mode={cfg.crop_mode} | "
        f"image_size={cfg.image_size}"
    )
    rank0_print(
        f"[Data] train={len(train_set)}, test={len(test_set)}, classes={cfg.num_classes}; "
        f"world_size={WORLD_SIZE}, per_gpu_batch={cfg.batch_size}, "
        f"global_batch={cfg.batch_size * WORLD_SIZE}"
    )
    rank0_print(
        f"[Opt] lr_last_blocks={cfg.lr_backbone:.2e}, lr_head={cfg.lr_head:.2e}, "
        f"warmup_epochs={cfg.warmup_epochs}, min_lr_scale={cfg.min_lr_scale}, "
        f"label_smoothing={cfg.label_smoothing}"
    )

    backbone = load_dinov2_model()
    trainable_backbone_names = configure_last_blocks_trainable(
        backbone,
        cfg.unfreeze_last_blocks,
    )
    dim = int(backbone.embed_dim)
    model = DINOv2LinearClassifier(backbone, dim, cfg.num_classes).to(DEVICE)

    num_trainable_backbone = sum(
        parameter.numel() for parameter in model.backbone.parameters() if parameter.requires_grad
    )
    num_trainable_head = sum(parameter.numel() for parameter in model.head.parameters())
    rank0_print(
        f"[Trainable] final_blocks={cfg.unfreeze_last_blocks}; "
        f"backbone_params={num_trainable_backbone:,}; "
        f"head_params={num_trainable_head:,}; "
        f"first={trainable_backbone_names[0]}; last={trainable_backbone_names[-1]}"
    )

    if is_distributed():
        model = DDP(
            model,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    optimizer = build_optimizer(unwrap_model(model))
    scaler = build_scaler(cfg.amp and DEVICE.type == "cuda")

    best_acc = -1.0
    start_epoch = 1

    resume_path = resolve_resume_path()
    if resume_path is not None:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Requested resume checkpoint not found: {resume_path}")

        checkpoint = safe_torch_load(resume_path, map_location="cpu")
        unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
        best_acc = float(checkpoint.get("best_acc", -1.0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

        if not cfg.reset_optimizer_on_resume:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint.get("scaler") is not None and scaler.is_enabled():
                scaler.load_state_dict(checkpoint["scaler"])

        rank0_print(
            f"[Resume] {resume_path} | start_epoch={start_epoch} | best_acc={best_acc:.4f}"
        )

    if start_epoch > cfg.epochs:
        rank0_print("[Done] requested epoch count is already completed.")
        cleanup_distributed()
        return

    autocast_enabled = cfg.amp and DEVICE.type == "cuda"

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        progress = tqdm(
            train_loader,
            desc=f"Train {epoch:03d}/{cfg.epochs}",
            dynamic_ncols=True,
            disable=not is_main_process(),
        )

        for step, (images, labels) in enumerate(progress):
            global_progress = (epoch - 1) + step / max(1, len(train_loader))
            lr_factor = cosine_warmup_lambda(global_progress)
            for group in optimizer.param_groups:
                base_lr = cfg.lr_backbone if group["name"] == "backbone" else cfg.lr_head
                group["lr"] = base_lr * lr_factor

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                logits = model(images)
                loss = F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=cfg.label_smoothing,
                )

            scaler.scale(loss).backward()

            if cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    unwrap_model(model).parameters(),
                    max_norm=cfg.grad_clip_norm,
                )

            scaler.step(optimizer)
            scaler.update()

            batch_size = int(labels.numel())
            running_loss += float(loss.detach().item()) * batch_size
            running_correct += int((logits.argmax(dim=1) == labels).sum().item())
            running_total += batch_size

            if is_main_process():
                progress.set_postfix(
                    loss=f"{running_loss / max(1, running_total):.4f}",
                    acc=f"{running_correct / max(1, running_total):.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        if is_distributed():
            stats = torch.tensor(
                [running_loss, running_correct, running_total],
                device=DEVICE,
                dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            running_loss = float(stats[0].item())
            running_correct = int(stats[1].item())
            running_total = int(stats[2].item())

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)

        if epoch % cfg.eval_every == 0:
            test_acc = evaluate(model, test_loader)
        else:
            test_acc = float("nan")

        rank0_print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f}"
        )

        if is_main_process() and epoch % cfg.save_every == 0:
            save_checkpoint(
                os.path.join(cfg.save_dir, "last.pth"),
                epoch,
                best_acc,
                model,
                optimizer,
                scaler,
            )

        if math.isfinite(test_acc) and test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(
                os.path.join(cfg.save_dir, "best.pth"),
                epoch,
                best_acc,
                model,
                optimizer,
                scaler,
            )
            rank0_print(f"[Best] epoch={epoch} acc={best_acc:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
