#!/usr/bin/env python3
"""Train shared part-prototype concepts on CUB with light DINOv2 finetuning.

This is the "scheme A" version discussed in chat:

  shared part-level prototype memory M[p,k]
  + bounded trainable residual R[p,k]
  + class-specific non-negative readout weights w[c,p,k]

The prototypes are shared across classes.  Class contrast is produced by
class-specific readout weights over the same shared part-concept dictionary:

  logit_c(x) = bias_c + sum_p sum_k w[c,p,k] * e[p,k](x)

The memory M is updated with the original paper-style mechanism:
responsibility-weighted EMA + semantic injection from part boxes.  The part
boxes are only used for route/EMA supervision; by default prototype scoring is
not hard-masked by boxes.

Expected geometry:
  raw CUB image -> CUB GT bird bbox crop -> resize/warp to 224x224.
  GroundingDINO boxes should be generated on the same crop geometry, possibly
  at another resolution; this script rescales them to 224.
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_PARTS = ("beak", "head", "wing", "body", "tail", "feet")


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    cub_root: str = "./data/CUB_200_2011"
    gdino_box_dir: str = "./artifacts/gdino_part_boxes_gtbbox_warp518"
    gdino_train_file: str = "train_part_boxes_gtbbox_warp518.pt"
    gdino_test_file: str = "test_part_boxes_gtbbox_warp518.pt"
    save_dir: str = "./runs/shared_part_proto_finetune"

    # DINOv2 torch.hub name.  Common options: dinov2_vits14, dinov2_vitb14.
    dino_model: str = "dinov2_vitb14"
    image_size: int = 224
    num_classes: int = 200
    parts: Tuple[str, ...] = DEFAULT_PARTS
    k_per_part: int = 50

    # Light finetuning.
    unfreeze_last_blocks: int = 2
    unfreeze_norm: bool = True
    freeze_backbone_epochs: int = 0

    # Prototype scoring.
    # resp_sum: original responsibility-weighted evidence.
    # scan_max: full-image prototype scan max, no part-box mask.
    # scan_topk: average top-k full-image prototype responses.
    score_mode: str = "resp_sum"
    scan_topk: int = 3
    tau_part: float = 0.20
    tau_proto: float = 0.05
    null_logit_init: float = 0.0
    residual_scale: float = 0.20

    # EMA memory update.
    ema_rho: float = 0.95
    ema_sem_mix: float = 0.50
    ema_min_mass: float = 1e-3
    ema_start_epoch: int = 1
    ema_every_steps: int = 1

    # Part-box target.
    box_target_gaussian: bool = True
    box_gaussian_sigma_scale: float = 0.50

    # Loss weights.
    label_smoothing: float = 0.0
    lambda_ce: float = 1.0
    lambda_route: float = 1.0
    route_final_ratio: float = 0.25
    route_decay_epochs: int = 60
    lambda_vis: float = 0.05
    lambda_proto_lb: float = 0.02
    lambda_proto_div: float = 0.02
    proto_div_margin: float = 0.30
    lambda_cls_sparse: float = 1e-5

    # Optim.
    lr_backbone: float = 1e-5
    lr_router: float = 3e-5
    lr_proto: float = 3e-5
    lr_classifier: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0

    # Train.
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 0
    amp: bool = True
    resume: bool = True
    resume_from: str = "last"  # last | best | none

    # Semantic bootstrap.
    bootstrap_memory: bool = True
    bootstrap_batches: int = 16
    bootstrap_max_tokens_per_part: int = 20000
    bootstrap_kmeans_iters: int = 20

    # Eval / logging.
    eval_every: int = 1
    save_every: int = 1
    compute_scan_purity: bool = True

    eps: float = 1e-9


cfg = CFG()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2n(x: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


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
            if vals:
                out[int(vals[0])] = tuple(float(x) for x in vals[1:5])
    return out


def crop_bbox(image: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
    x, y, w, h = bbox
    width, height = image.size
    x1 = max(0, min(width - 1, int(np.floor(x))))
    y1 = max(0, min(height - 1, int(np.floor(y))))
    x2 = max(x1 + 1, min(width, int(np.ceil(x + w))))
    y2 = max(y1 + 1, min(height, int(np.ceil(y + h))))
    return image.crop((x1, y1, x2, y2))


def route_lambda(epoch: int) -> float:
    if cfg.route_decay_epochs <= 0:
        return cfg.lambda_route
    t = min(max(epoch - 1, 0) / float(cfg.route_decay_epochs), 1.0)
    ratio = 1.0 + t * (cfg.route_final_ratio - 1.0)
    return cfg.lambda_route * ratio


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# CUB metadata and GDINO boxes
# -----------------------------
def build_cub_samples(cub_root: str, split: str) -> List[Dict[str, Any]]:
    paths = read_kv_text(os.path.join(cub_root, "images.txt"))
    labels = read_kv_int(os.path.join(cub_root, "image_class_labels.txt"))
    split_map = read_kv_int(os.path.join(cub_root, "train_test_split.txt"))
    bboxes = read_bboxes(os.path.join(cub_root, "bounding_boxes.txt"))
    want_train = split == "train"
    samples: List[Dict[str, Any]] = []
    for image_id in sorted(paths):
        if (split_map[image_id] == 1) != want_train:
            continue
        samples.append({
            "image_id": image_id,
            "relpath": paths[image_id],
            "label": labels[image_id] - 1,
            "bbox": bboxes[image_id],
        })
    return samples


def _parse_one_sample_part_boxes(elem: Any, parts_in_file: List[str]) -> np.ndarray:
    out = np.full((len(parts_in_file), 4), -1.0, dtype=np.float32)
    if elem is None:
        return out
    if isinstance(elem, dict):
        for j, name in enumerate(parts_in_file):
            val = elem.get(name)
            if val is None:
                continue
            arr = np.asarray(val, dtype=np.float32)
            if arr.ndim == 1 and arr.size == 4:
                out[j] = arr
            elif arr.ndim == 2 and arr.shape[-1] == 4 and arr.shape[0] > 0:
                valid = arr[(arr[:, 0] >= 0) & (arr[:, 2] > arr[:, 0]) & (arr[:, 3] > arr[:, 1])]
                if len(valid):
                    out[j] = np.array([valid[:,0].min(), valid[:,1].min(), valid[:,2].max(), valid[:,3].max()], dtype=np.float32)
        return out
    arr = np.asarray(elem, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[-1] == 4:
        m = min(len(parts_in_file), arr.shape[0])
        out[:m] = arr[:m]
    return out


def _load_gdino_part_boxes_tensor(gd: Dict[str, Any], parts_in_file: List[str]) -> torch.Tensor:
    pb = gd["part_boxes_xyxy_pix"]
    p_need = len(parts_in_file)
    if isinstance(pb, torch.Tensor):
        pb = pb.float()
        if pb.ndim == 3 and pb.shape[-1] == 4:
            out = torch.full((pb.shape[0], p_need, 4), -1.0, dtype=torch.float32)
            out[:, : min(p_need, pb.shape[1])] = pb[:, : min(p_need, pb.shape[1])]
            return out
        if pb.ndim == 4 and pb.shape[-1] == 4:
            b = pb.reshape(pb.shape[0], pb.shape[1], -1, 4)
            valid = (b[..., 0] >= 0) & (b[..., 2] > b[..., 0]) & (b[..., 3] > b[..., 1])
            big = torch.tensor(1e9, dtype=torch.float32)
            neg = torch.tensor(-1e9, dtype=torch.float32)
            x1 = torch.where(valid, b[..., 0], big).min(dim=2).values
            y1 = torch.where(valid, b[..., 1], big).min(dim=2).values
            x2 = torch.where(valid, b[..., 2], neg).max(dim=2).values
            y2 = torch.where(valid, b[..., 3], neg).max(dim=2).values
            any_valid = valid.any(dim=2)
            merged = torch.stack([x1, y1, x2, y2], dim=-1)
            merged[~any_valid] = -1.0
            out = torch.full((pb.shape[0], p_need, 4), -1.0, dtype=torch.float32)
            out[:, : min(p_need, merged.shape[1])] = merged[:, : min(p_need, merged.shape[1])]
            return out
        raise ValueError(f"Unexpected part box tensor shape: {tuple(pb.shape)}")
    if isinstance(pb, (list, tuple)):
        out = torch.full((len(pb), p_need, 4), -1.0, dtype=torch.float32)
        for i, elem in enumerate(pb):
            out[i] = torch.from_numpy(_parse_one_sample_part_boxes(elem, parts_in_file))
        return out
    raise TypeError(f"Unsupported part_boxes_xyxy_pix container: {type(pb)}")


def load_aligned_part_boxes(samples: List[Dict[str, Any]], gdino_path: str, model_image_size: int) -> torch.Tensor:
    gd = torch.load(gdino_path, map_location="cpu")
    if "relpaths" not in gd or "part_boxes_xyxy_pix" not in gd:
        raise KeyError(f"Bad GDINO cache: {gdino_path}")
    src_size = int(gd.get("img_size", model_image_size))
    scale = float(model_image_size) / float(src_size)
    parts_in_file = [str(x).lower() for x in gd.get("parts", list(DEFAULT_PARTS))]
    boxes_pf = _load_gdino_part_boxes_tensor(gd, parts_in_file)
    relpaths = [str(x) for x in gd["relpaths"]]
    gd_index = {rp: i for i, rp in enumerate(relpaths)}

    def match_indices(target: str) -> List[int]:
        t = target.lower()
        if t == "wing":
            return [i for i, n in enumerate(parts_in_file) if "wing" in n]
        if t == "beak":
            return [i for i, n in enumerate(parts_in_file) if "beak" in n or "bill" in n]
        if t == "feet":
            return [i for i, n in enumerate(parts_in_file) if "feet" in n or "foot" in n or "leg" in n]
        return [i for i, n in enumerate(parts_in_file) if n == t]

    merged_src = torch.full((boxes_pf.shape[0], len(cfg.parts), 4), -1.0, dtype=torch.float32)
    for p, part_name in enumerate(cfg.parts):
        idxs = match_indices(part_name)
        if not idxs:
            continue
        bx = boxes_pf[:, idxs, :]
        valid = (bx[..., 0] >= 0) & (bx[..., 2] > bx[..., 0]) & (bx[..., 3] > bx[..., 1])
        big = torch.tensor(1e9, dtype=torch.float32)
        neg = torch.tensor(-1e9, dtype=torch.float32)
        x1 = torch.where(valid, bx[..., 0], big).min(dim=1).values
        y1 = torch.where(valid, bx[..., 1], big).min(dim=1).values
        x2 = torch.where(valid, bx[..., 2], neg).max(dim=1).values
        y2 = torch.where(valid, bx[..., 3], neg).max(dim=1).values
        any_valid = valid.any(dim=1)
        merged = torch.stack([x1, y1, x2, y2], dim=1)
        merged[~any_valid] = -1.0
        merged_src[:, p] = merged

    out = torch.full((len(samples), len(cfg.parts), 4), -1.0, dtype=torch.float32)
    missing = 0
    for i, s in enumerate(samples):
        j = gd_index.get(str(s["relpath"]))
        if j is None:
            missing += 1
        else:
            out[i] = merged_src[j] * scale
    print(f"[GDINO] {os.path.basename(gdino_path)} aligned={len(samples)-missing}/{len(samples)} missing={missing}")
    return out


class CUBImageWithPartBoxes(Dataset):
    def __init__(self, split: str, gdino_path: str):
        if split not in {"train", "test"}:
            raise ValueError(split)
        self.split = split
        self.samples = build_cub_samples(cfg.cub_root, split)
        self.boxes = load_aligned_part_boxes(self.samples, gdino_path, cfg.image_size)
        self.transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        path = os.path.join(cfg.cub_root, "images", s["relpath"])
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = crop_bbox(img, s["bbox"])
            x = self.transform(img)
        return x, self.boxes[idx], int(s["label"]), str(s["relpath"])


def make_loader(ds: Dataset, train: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        pin_memory=(DEVICE == "cuda"),
        drop_last=train,
    )


# -----------------------------
# Part boxes -> token distributions
# -----------------------------
def boxes_to_soft_q(boxes: torch.Tensor, h: int, w: int, image_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, parts, _ = boxes.shape
    device = boxes.device
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * (float(image_size) / h)
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * (float(image_size) / w)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    q = torch.zeros((bsz, parts, h, w), device=device, dtype=torch.float32)
    valid_bp = torch.zeros((bsz, parts), device=device, dtype=torch.bool)
    for b in range(bsz):
        for p in range(parts):
            x1, y1, x2, y2 = boxes[b, p]
            if x1 < 0 or x2 <= x1 or y2 <= y1:
                continue
            x1 = x1.clamp(0, image_size)
            y1 = y1.clamp(0, image_size)
            x2 = x2.clamp(0, image_size)
            y2 = y2.clamp(0, image_size)
            mask = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
            if not mask.any():
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                flat = int(((xx - cx).pow(2) + (yy - cy).pow(2)).argmin().item())
                mask = torch.zeros_like(mask)
                mask.view(-1)[flat] = True
            if cfg.box_target_gaussian:
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                sx = ((x2 - x1) * cfg.box_gaussian_sigma_scale).clamp_min(float(image_size) / w)
                sy = ((y2 - y1) * cfg.box_gaussian_sigma_scale).clamp_min(float(image_size) / h)
                weight = torch.exp(-0.5 * (((xx - cx) / sx).pow(2) + ((yy - cy) / sy).pow(2))) * mask.float()
            else:
                weight = mask.float()
            if weight.sum() > 0:
                q[b, p] = weight / weight.sum().clamp_min(cfg.eps)
                valid_bp[b, p] = True
    return q.flatten(2), valid_bp


# -----------------------------
# DINO backbone
# -----------------------------
def load_dinov2(model_name: str) -> nn.Module:
    # Assumes internet or cached torch hub repo on server.
    return torch.hub.load("facebookresearch/dinov2", model_name)


def set_dino_trainability(backbone: nn.Module, last_blocks: int, unfreeze_norm: bool) -> int:
    for p in backbone.parameters():
        p.requires_grad = False
    if last_blocks > 0 and hasattr(backbone, "blocks"):
        blocks = list(backbone.blocks)
        for blk in blocks[-int(last_blocks):]:
            for p in blk.parameters():
                p.requires_grad = True
    if unfreeze_norm and hasattr(backbone, "norm"):
        for p in backbone.norm.parameters():
            p.requires_grad = True
    return sum(p.numel() for p in backbone.parameters() if p.requires_grad)


# -----------------------------
# Model
# -----------------------------
class SharedPartPrototypeDINO(nn.Module):
    def __init__(self, backbone: nn.Module, dim: int, parts: int, k: int, classes: int):
        super().__init__()
        self.backbone = backbone
        self.dim = int(dim)
        self.parts = int(parts)
        self.k = int(k)
        self.classes = int(classes)
        self.part_queries = nn.Parameter(torch.randn(parts, dim) * 0.02)
        self.null_logits = nn.Parameter(torch.full((parts,), float(cfg.null_logit_init)))
        memory = l2n(torch.randn(parts, k, dim), dim=-1)
        self.register_buffer("memory", memory)
        self.proto_residual = nn.Parameter(torch.zeros(parts, k, dim))
        self.class_theta = nn.Parameter(torch.full((classes, parts, k), -4.0))
        self.class_bias = nn.Parameter(torch.zeros(classes))

    def extract_tokens(self, images: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        feats = self.backbone.forward_features(images)
        if isinstance(feats, dict):
            x = feats.get("x_norm_patchtokens", None)
            if x is None:
                x = feats.get("x_prenorm", None)
                if x is not None:
                    # Drop CLS if present.
                    x = x[:, 1:]
        else:
            x = feats
            if x.ndim == 3 and x.shape[1] == 197:
                x = x[:, 1:]
        if x is None or x.ndim != 3:
            raise RuntimeError("Could not extract patch tokens from DINO output.")
        n = x.shape[1]
        g = int(round(math.sqrt(n)))
        if g * g != n:
            raise RuntimeError(f"Patch tokens are not a square grid: N={n}")
        return x.float(), g, g

    def effective_prototypes(self) -> torch.Tensor:
        delta = float(cfg.residual_scale) * torch.tanh(self.proto_residual.float())
        return l2n(self.memory.float() + delta, dim=-1, eps=cfg.eps)

    def class_weights(self) -> torch.Tensor:
        return F.softplus(self.class_theta.float())

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        x, h, w = self.extract_tokens(images)
        xn = l2n(x, dim=-1, eps=cfg.eps)
        # One distribution over local tokens plus a null slot per named part.
        part_logits = torch.einsum("bnc,pc->bpn", xn, l2n(self.part_queries.float(), dim=-1, eps=cfg.eps))
        part_logits = part_logits / max(cfg.tau_part, 1e-6)
        null = self.null_logits.float().view(1, self.parts, 1).expand(xn.shape[0], -1, -1)
        part_all = torch.cat([part_logits, null], dim=-1)
        part_prob = F.softmax(part_all, dim=-1)
        part_map = part_prob[..., :-1]
        visibility = 1.0 - part_prob[..., -1]

        prototypes = self.effective_prototypes()
        sim = torch.einsum("bnc,pkc->bpnk", xn, prototypes)
        proto_assign = F.softmax(sim / max(cfg.tau_proto, 1e-6), dim=-1)
        responsibility = part_map.unsqueeze(-1) * proto_assign
        relu_sim = F.relu(sim)

        if cfg.score_mode == "resp_sum":
            proto_score = (responsibility * relu_sim).sum(dim=2)  # B,P,K
        elif cfg.score_mode == "scan_max":
            proto_score = relu_sim.max(dim=2).values
        elif cfg.score_mode == "scan_topk":
            kk = min(max(1, cfg.scan_topk), relu_sim.shape[2])
            proto_score = relu_sim.topk(kk, dim=2).values.mean(dim=2)
        else:
            raise ValueError(f"Unknown score_mode={cfg.score_mode}")

        utilization = responsibility.sum(dim=2)
        class_weight = self.class_weights()
        contributions = proto_score[:, None] * class_weight[None]  # B,C,P,K
        part_evidence = contributions.sum(dim=-1)                  # B,C,P
        logits = self.class_bias.float().view(1, -1) + part_evidence.sum(dim=-1)

        return {
            "logits": logits,
            "Xn": xn,
            "grid_h": torch.tensor(h, device=images.device),
            "grid_w": torch.tensor(w, device=images.device),
            "part_map": part_map,
            "visibility": visibility,
            "prototypes": prototypes,
            "sim": sim,
            "proto_assign": proto_assign,
            "responsibility": responsibility,
            "utilization": utilization,
            "proto_score": proto_score,
            "class_weight": class_weight,
            "contributions": contributions,
            "part_evidence": part_evidence,
        }

    @torch.no_grad()
    def ema_update_memory(
        self,
        xn: torch.Tensor,
        part_map: torch.Tensor,
        proto_assign: torch.Tensor,
        q_sem: torch.Tensor,
        valid_bp: torch.Tensor,
    ) -> None:
        # Original-style self-routed target.
        self_resp = part_map.unsqueeze(-1) * proto_assign
        self_den = self_resp.sum(dim=(0, 2))
        self_num = torch.einsum("bpnk,bnc->pkc", self_resp, xn.float())
        self_target = l2n(self_num / self_den.unsqueeze(-1).clamp_min(cfg.eps), dim=-1, eps=cfg.eps)

        # Semantic target: part box first, then proto_assign partitions that region.
        sem_resp = q_sem.unsqueeze(-1) * proto_assign
        sem_den = sem_resp.sum(dim=(0, 2))
        sem_num = torch.einsum("bpnk,bnc->pkc", sem_resp, xn.float())
        sem_target = l2n(sem_num / sem_den.unsqueeze(-1).clamp_min(cfg.eps), dim=-1, eps=cfg.eps)

        valid_part = valid_bp.any(dim=0).float().view(self.parts, 1, 1)
        sem_valid_pk = (sem_den > cfg.ema_min_mass).float().unsqueeze(-1)
        alpha = float(cfg.ema_sem_mix) * valid_part * sem_valid_pk
        target = l2n((1.0 - alpha) * self_target + alpha * sem_target, dim=-1, eps=cfg.eps)

        update_mask = ((self_den > cfg.ema_min_mass) | (sem_den > cfg.ema_min_mass)).unsqueeze(-1)
        candidate = l2n(float(cfg.ema_rho) * self.memory.float() + (1.0 - float(cfg.ema_rho)) * target,
                        dim=-1, eps=cfg.eps)
        self.memory.copy_(torch.where(update_mask, candidate, self.memory.float()))


# -----------------------------
# Losses and metrics
# -----------------------------
def semantic_route_loss(part_map: torch.Tensor, q_sem: torch.Tensor, valid_bp: torch.Tensor) -> torch.Tensor:
    pred = part_map / part_map.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)
    ce = -(q_sem * pred.clamp_min(cfg.eps).log()).sum(dim=-1)
    valid = valid_bp.float()
    return (ce * valid).sum() / valid.sum().clamp_min(1.0)


def visible_part_loss(visibility: torch.Tensor, valid_bp: torch.Tensor) -> torch.Tensor:
    valid = valid_bp.float()
    loss = -visibility.clamp_min(cfg.eps).log()
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def semantic_proto_load_balance_loss(proto_assign: torch.Tensor, q_sem: torch.Tensor, valid_bp: torch.Tensor) -> torch.Tensor:
    usage = (q_sem.unsqueeze(-1) * proto_assign).sum(dim=(0, 2))
    part_valid = valid_bp.any(dim=0)
    if not part_valid.any():
        return usage.new_tensor(0.0)
    prob = usage / usage.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)
    entropy = -(prob.clamp_min(cfg.eps) * prob.clamp_min(cfg.eps).log()).sum(dim=-1)
    return (math.log(max(2, usage.shape[-1])) - entropy[part_valid]).mean()


def within_part_prototype_diversity_loss(prototypes: torch.Tensor) -> torch.Tensor:
    q = l2n(prototypes.float(), dim=-1, eps=cfg.eps)
    sim = torch.matmul(q, q.transpose(1, 2))
    k = q.shape[1]
    eye = torch.eye(k, device=q.device, dtype=torch.bool).unsqueeze(0)
    penalty = F.relu(sim - float(cfg.proto_div_margin)).pow(2).masked_fill(eye, 0.0)
    return penalty.sum() / max(1, q.shape[0] * k * (k - 1))


def classifier_sparsity_loss(class_weight: torch.Tensor) -> torch.Tensor:
    return class_weight.mean()


@torch.no_grad()
def compute_eval(model: SharedPartPrototypeDINO, loader: DataLoader, epoch: int) -> Dict[str, Any]:
    model.eval()
    totals: Dict[str, float] = {"count": 0, "correct": 0, "ce": 0.0, "route": 0.0}
    # full-image scan top activation lies inside its named part box.
    purity_num = 0.0
    purity_den = 0.0
    active_sum = torch.zeros((model.parts, model.k), device=DEVICE)
    part_margin_abs = torch.zeros((model.parts,), device=DEVICE)
    part_margin_count = 0

    for images, boxes, y, _ in tqdm(loader, desc=f"Eval {epoch}", ncols=120):
        images = images.to(DEVICE, non_blocking=True).float()
        boxes = boxes.to(DEVICE, non_blocking=True).float()
        y = y.to(DEVICE, non_blocking=True)
        out = model(images)
        h, w = int(out["grid_h"].item()), int(out["grid_w"].item())
        q_sem, valid_bp = boxes_to_soft_q(boxes, h, w, cfg.image_size)
        logits = out["logits"]
        bs = y.shape[0]
        totals["count"] += bs
        totals["correct"] += int((logits.argmax(1) == y).sum().item())
        totals["ce"] += float(F.cross_entropy(logits, y).item()) * bs
        totals["route"] += float(semantic_route_loss(out["part_map"], q_sem, valid_bp).item()) * bs
        active_sum += out["proto_score"].float().sum(dim=0)

        # part-wise top1-vs-top2 margin contribution magnitude.
        top2 = logits.topk(2, dim=1).indices
        pred, conf = top2[:, 0], top2[:, 1]
        pe = out["part_evidence"]
        delta = pe[torch.arange(bs, device=DEVICE), pred] - pe[torch.arange(bs, device=DEVICE), conf]
        part_margin_abs += delta.abs().sum(dim=0)
        part_margin_count += bs

        if cfg.compute_scan_purity:
            sim = out["sim"].float()  # B,P,N,K, unmasked scan
            topn = sim.argmax(dim=2)  # B,P,K
            # q_sem > 0 means inside the original hard box support because q outside box is zero.
            q_flat = q_sem > 0
            for p in range(model.parts):
                valid_b = valid_bp[:, p]
                if not valid_b.any():
                    continue
                ids = topn[valid_b, p, :]  # Bv,K
                q_p = q_flat[valid_b, p, :]  # Bv,N
                hit = q_p.gather(1, ids.reshape(ids.shape[0], -1)).float()
                purity_num += float(hit.sum().item())
                purity_den += float(hit.numel())

    n = max(1, int(totals["count"]))
    metrics: Dict[str, Any] = {
        "acc": totals["correct"] / n,
        "ce": totals["ce"] / n,
        "route": totals["route"] / n,
        "scan_part_purity": (purity_num / max(1.0, purity_den)) if cfg.compute_scan_purity else None,
        "active_proto_mean": (active_sum / n).detach().cpu().numpy().tolist(),
        "mean_abs_part_margin": (part_margin_abs / max(1, part_margin_count)).detach().cpu().numpy().tolist(),
    }
    return metrics


# -----------------------------
# Bootstrap memory
# -----------------------------
@torch.no_grad()
def torch_kmeans_cosine(x: torch.Tensor, k: int, iters: int) -> torch.Tensor:
    x = l2n(x.float(), dim=-1, eps=cfg.eps)
    n = x.shape[0]
    if n == 0:
        return l2n(torch.randn(k, x.shape[-1], device=x.device), dim=-1)
    if n < k:
        reps = int(math.ceil(k / n))
        init = x.repeat(reps, 1)[:k].clone()
    else:
        idx = torch.randperm(n, device=x.device)[:k]
        init = x[idx].clone()
    cent = l2n(init, dim=-1, eps=cfg.eps)
    for _ in range(max(1, iters)):
        assign = (x @ cent.t()).argmax(dim=1)
        new = []
        for j in range(k):
            m = assign == j
            if m.any():
                new.append(x[m].mean(dim=0))
            else:
                new.append(x[torch.randint(0, n, (1,), device=x.device)[0]])
        cent = l2n(torch.stack(new, dim=0), dim=-1, eps=cfg.eps)
    return cent


@torch.no_grad()
def bootstrap_memory_from_part_boxes(model: SharedPartPrototypeDINO, loader: DataLoader) -> None:
    if cfg.bootstrap_batches <= 0:
        return
    model.eval()
    buckets: List[List[torch.Tensor]] = [[] for _ in range(model.parts)]
    max_per_part = int(cfg.bootstrap_max_tokens_per_part)
    seen_batches = 0
    for images, boxes, _, _ in tqdm(loader, desc="Bootstrap memory", ncols=120):
        images = images.to(DEVICE, non_blocking=True).float()
        boxes = boxes.to(DEVICE, non_blocking=True).float()
        x, h, w = model.extract_tokens(images)
        xn = l2n(x.float(), dim=-1, eps=cfg.eps)
        q_sem, valid_bp = boxes_to_soft_q(boxes, h, w, cfg.image_size)
        hard = q_sem > 0
        for p in range(model.parts):
            mask = hard[:, p, :]
            if mask.any() and sum(t.shape[0] for t in buckets[p]) < max_per_part:
                vals = xn[mask]
                if vals.shape[0] > 2048:
                    vals = vals[torch.randperm(vals.shape[0], device=vals.device)[:2048]]
                buckets[p].append(vals.detach())
        seen_batches += 1
        if seen_batches >= cfg.bootstrap_batches:
            break
    new_mem = model.memory.detach().clone().to(DEVICE)
    for p in range(model.parts):
        if buckets[p]:
            x = torch.cat(buckets[p], dim=0)
            if x.shape[0] > max_per_part:
                x = x[torch.randperm(x.shape[0], device=x.device)[:max_per_part]]
            new_mem[p] = torch_kmeans_cosine(x, model.k, cfg.bootstrap_kmeans_iters)
            print(f"[Bootstrap] {cfg.parts[p]} tokens={x.shape[0]}")
        else:
            print(f"[Bootstrap] warning: no tokens for {cfg.parts[p]}")
    model.memory.copy_(l2n(new_mem, dim=-1, eps=cfg.eps).cpu() if model.memory.device.type == "cpu" else l2n(new_mem, dim=-1, eps=cfg.eps))
    model.train()


# -----------------------------
# Checkpointing
# -----------------------------
def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, epoch: int, best_acc: float) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save({
        "epoch": epoch,
        "best_acc": best_acc,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "cfg": asdict(cfg),
    }, path)


def pick_resume_path(save_dir: str) -> Optional[str]:
    if cfg.resume_from == "none" or not cfg.resume:
        return None
    name = "best.pth" if cfg.resume_from == "best" else "last.pth"
    path = os.path.join(save_dir, name)
    return path if os.path.isfile(path) else None


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Shared part-prototype DINOv2 finetuning on CUB")
    p.add_argument("--cub-root", default=os.environ.get("CUB_ROOT", cfg.cub_root))
    p.add_argument("--gdino-box-dir", default=os.environ.get("GDINO_BOX_DIR", cfg.gdino_box_dir))
    p.add_argument("--gdino-train-file", default=cfg.gdino_train_file)
    p.add_argument("--gdino-test-file", default=cfg.gdino_test_file)
    p.add_argument("--save-dir", default=os.environ.get("SAVE_DIR", cfg.save_dir))
    p.add_argument("--dino-model", default=cfg.dino_model)
    p.add_argument("--image-size", type=int, default=cfg.image_size)
    p.add_argument("--parts", default=",".join(cfg.parts), help="Comma-separated parts.")
    p.add_argument("--k-per-part", type=int, default=cfg.k_per_part)
    p.add_argument("--score-mode", choices=["resp_sum", "scan_max", "scan_topk"], default=cfg.score_mode)
    p.add_argument("--scan-topk", type=int, default=cfg.scan_topk)
    p.add_argument("--unfreeze-last-blocks", type=int, default=cfg.unfreeze_last_blocks)
    p.add_argument("--freeze-backbone-epochs", type=int, default=cfg.freeze_backbone_epochs)
    p.add_argument("--no-unfreeze-norm", action="store_true")
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--lr-backbone", type=float, default=cfg.lr_backbone)
    p.add_argument("--lr-router", type=float, default=cfg.lr_router)
    p.add_argument("--lr-proto", type=float, default=cfg.lr_proto)
    p.add_argument("--lr-classifier", type=float, default=cfg.lr_classifier)
    p.add_argument("--ema-rho", type=float, default=cfg.ema_rho)
    p.add_argument("--ema-sem-mix", type=float, default=cfg.ema_sem_mix)
    p.add_argument("--ema-start-epoch", type=int, default=cfg.ema_start_epoch)
    p.add_argument("--residual-scale", type=float, default=cfg.residual_scale)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--resume-from", choices=["last", "best", "none"], default=cfg.resume_from)
    p.add_argument("--skip-bootstrap", action="store_true")
    p.add_argument("--bootstrap-batches", type=int, default=cfg.bootstrap_batches)
    p.add_argument("--no-scan-purity", action="store_true")
    return p.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    cfg.cub_root = os.path.abspath(os.path.expanduser(args.cub_root))
    cfg.gdino_box_dir = os.path.abspath(os.path.expanduser(args.gdino_box_dir))
    cfg.gdino_train_file = args.gdino_train_file
    cfg.gdino_test_file = args.gdino_test_file
    cfg.save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
    cfg.dino_model = args.dino_model
    cfg.image_size = args.image_size
    cfg.parts = tuple([x.strip().lower() for x in args.parts.split(",") if x.strip()])
    cfg.k_per_part = args.k_per_part
    cfg.score_mode = args.score_mode
    cfg.scan_topk = args.scan_topk
    cfg.unfreeze_last_blocks = args.unfreeze_last_blocks
    cfg.freeze_backbone_epochs = args.freeze_backbone_epochs
    cfg.unfreeze_norm = not args.no_unfreeze_norm
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.seed = args.seed
    cfg.lr_backbone = args.lr_backbone
    cfg.lr_router = args.lr_router
    cfg.lr_proto = args.lr_proto
    cfg.lr_classifier = args.lr_classifier
    cfg.ema_rho = args.ema_rho
    cfg.ema_sem_mix = args.ema_sem_mix
    cfg.ema_start_epoch = args.ema_start_epoch
    cfg.residual_scale = args.residual_scale
    cfg.amp = not args.no_amp
    cfg.resume = not args.no_resume
    cfg.resume_from = args.resume_from
    cfg.bootstrap_memory = not args.skip_bootstrap
    cfg.bootstrap_batches = args.bootstrap_batches
    cfg.compute_scan_purity = not args.no_scan_purity


def preflight() -> None:
    required = [
        os.path.join(cfg.cub_root, "images.txt"),
        os.path.join(cfg.cub_root, "image_class_labels.txt"),
        os.path.join(cfg.cub_root, "train_test_split.txt"),
        os.path.join(cfg.cub_root, "bounding_boxes.txt"),
        os.path.join(cfg.cub_root, "images"),
        os.path.join(cfg.gdino_box_dir, cfg.gdino_train_file),
        os.path.join(cfg.gdino_box_dir, cfg.gdino_test_file),
    ]
    missing = [x for x in required if not os.path.exists(x)]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(f"  - {x}" for x in missing))


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    apply_args(args)
    preflight()
    set_seed(cfg.seed)
    ensure_dir(cfg.save_dir)
    with open(os.path.join(cfg.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_gdino = os.path.join(cfg.gdino_box_dir, cfg.gdino_train_file)
    test_gdino = os.path.join(cfg.gdino_box_dir, cfg.gdino_test_file)
    ds_train = CUBImageWithPartBoxes("train", train_gdino)
    ds_test = CUBImageWithPartBoxes("test", test_gdino)
    dl_train = make_loader(ds_train, train=True)
    dl_test = make_loader(ds_test, train=False)

    print(f"[DINO] loading {cfg.dino_model}")
    backbone = load_dinov2(cfg.dino_model).to(DEVICE)
    n_trainable = set_dino_trainability(backbone, cfg.unfreeze_last_blocks, cfg.unfreeze_norm)
    print(f"[DINO] trainable params after unfreeze={n_trainable:,}")

    # Infer dimension with one small forward.
    with torch.no_grad():
        x0, _, _, _ = next(iter(DataLoader(ds_train, batch_size=1, shuffle=False, num_workers=0)))
        x0 = x0.to(DEVICE)
        feats = backbone.forward_features(x0)
        if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
            dim = int(feats["x_norm_patchtokens"].shape[-1])
        else:
            raise RuntimeError("DINOv2 output missing x_norm_patchtokens")

    model = SharedPartPrototypeDINO(backbone, dim=dim, parts=len(cfg.parts), k=cfg.k_per_part, classes=cfg.num_classes).to(DEVICE)

    if cfg.freeze_backbone_epochs > 0:
        set_dino_trainability(model.backbone, 0, False)
        print(f"[DINO] backbone frozen for first {cfg.freeze_backbone_epochs} epochs")

    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay},
        {"params": [model.part_queries, model.null_logits], "lr": cfg.lr_router, "weight_decay": cfg.weight_decay},
        {"params": [model.proto_residual], "lr": cfg.lr_proto, "weight_decay": cfg.weight_decay},
        {"params": [model.class_theta, model.class_bias], "lr": cfg.lr_classifier, "weight_decay": 0.0},
    ])
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and DEVICE == "cuda"))

    start_epoch = 1
    best_acc = -1.0
    resume_path = pick_resume_path(cfg.save_dir)
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt.get("scaler", scaler.state_dict()))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_acc = float(ckpt.get("best_acc", -1.0))
        print(f"[Resume] {resume_path} -> epoch {start_epoch}, best_acc={best_acc:.4f}")
    elif cfg.bootstrap_memory:
        bootstrap_memory_from_part_boxes(model, dl_train)

    print(f"[Data] train={len(ds_train)} test={len(ds_test)} parts={cfg.parts} K={cfg.k_per_part}")
    print(f"[Train] device={DEVICE} amp={cfg.amp} score_mode={cfg.score_mode}")
    log_path = os.path.join(cfg.save_dir, "history.jsonl")

    global_step = 0
    for epoch in range(start_epoch, cfg.epochs + 1):
        if epoch == cfg.freeze_backbone_epochs + 1 and cfg.freeze_backbone_epochs > 0:
            set_dino_trainability(model.backbone, cfg.unfreeze_last_blocks, cfg.unfreeze_norm)
            # Add newly trainable params by recreating optimizer; simpler and safe for first trial.
            optimizer = torch.optim.AdamW([
                {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay},
                {"params": [model.part_queries, model.null_logits], "lr": cfg.lr_router, "weight_decay": cfg.weight_decay},
                {"params": [model.proto_residual], "lr": cfg.lr_proto, "weight_decay": cfg.weight_decay},
                {"params": [model.class_theta, model.class_bias], "lr": cfg.lr_classifier, "weight_decay": 0.0},
            ])
            print(f"[DINO] unfroze last {cfg.unfreeze_last_blocks} blocks at epoch {epoch}")

        model.train()
        t0 = time.time()
        lam_route = route_lambda(epoch)
        totals = {"loss":0.0, "ce":0.0, "route":0.0, "vis":0.0, "lb":0.0, "div":0.0, "sparse":0.0, "correct":0, "count":0}
        pbar = tqdm(dl_train, desc=f"Train {epoch}/{cfg.epochs}", ncols=160)
        for images, boxes, y, _ in pbar:
            images = images.to(DEVICE, non_blocking=True).float()
            boxes = boxes.to(DEVICE, non_blocking=True).float()
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and DEVICE == "cuda")):
                out = model(images)
                h, w = int(out["grid_h"].item()), int(out["grid_w"].item())
                q_sem, valid_bp = boxes_to_soft_q(boxes, h, w, cfg.image_size)
                loss_ce = F.cross_entropy(out["logits"], y, label_smoothing=cfg.label_smoothing)
                loss_route = semantic_route_loss(out["part_map"], q_sem, valid_bp)
                loss_vis = visible_part_loss(out["visibility"], valid_bp)
                loss_lb = semantic_proto_load_balance_loss(out["proto_assign"], q_sem, valid_bp)
                loss_div = within_part_prototype_diversity_loss(out["prototypes"])
                loss_sparse = classifier_sparsity_loss(out["class_weight"])
                loss = (cfg.lambda_ce * loss_ce + lam_route * loss_route + cfg.lambda_vis * loss_vis
                        + cfg.lambda_proto_lb * loss_lb + cfg.lambda_proto_div * loss_div
                        + cfg.lambda_cls_sparse * loss_sparse)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if epoch >= cfg.ema_start_epoch and (global_step % max(1, cfg.ema_every_steps) == 0):
                model.ema_update_memory(out["Xn"].detach(), out["part_map"].detach(), out["proto_assign"].detach(), q_sem.detach(), valid_bp.detach())
            global_step += 1

            bs = y.shape[0]
            totals["count"] += bs
            totals["correct"] += int((out["logits"].argmax(1) == y).sum().item())
            for key, val in [("loss", loss), ("ce", loss_ce), ("route", loss_route), ("vis", loss_vis), ("lb", loss_lb), ("div", loss_div), ("sparse", loss_sparse)]:
                totals[key] += float(val.detach().item()) * bs
            pbar.set_postfix(acc=totals["correct"] / max(1, totals["count"]), loss=totals["loss"] / max(1, totals["count"]))

        n = max(1, totals["count"])
        train_metrics = {k: (v / n if k not in {"correct", "count"} else v) for k, v in totals.items()}
        train_metrics["acc"] = totals["correct"] / n
        record: Dict[str, Any] = {"epoch": epoch, "time_sec": time.time() - t0, "train": train_metrics}

        if epoch % cfg.eval_every == 0:
            eval_metrics = compute_eval(model, dl_test, epoch)
            record["eval"] = eval_metrics
            acc = float(eval_metrics["acc"])
            print(f"[Epoch {epoch}] train_acc={train_metrics['acc']:.4f} eval_acc={acc:.4f} scan_purity={eval_metrics.get('scan_part_purity')}")
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(os.path.join(cfg.save_dir, "best.pth"), model, optimizer, scaler, epoch, best_acc)
        else:
            print(f"[Epoch {epoch}] train_acc={train_metrics['acc']:.4f}")

        record["best_acc"] = best_acc
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if epoch % cfg.save_every == 0:
            save_checkpoint(os.path.join(cfg.save_dir, "last.pth"), model, optimizer, scaler, epoch, best_acc)

    print(f"[Done] best_acc={best_acc:.4f}; save_dir={cfg.save_dir}")


if __name__ == "__main__":
    main()
