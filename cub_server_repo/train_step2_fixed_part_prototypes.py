#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step-2 baseline:
  Fixed part prototypes from GDINO/CUB-visibility part masks.

Pipeline:
  1. Load cached DINOv2 patch features.
  2. Load GDINO part boxes.
  3. Convert boxes -> q_sem over DINO patch tokens.
  4. Collect DINO tokens inside each semantic part region.
  5. Run spherical KMeans per part to build fixed prototypes.
  6. Freeze prototypes.
  7. Train only a sparse non-negative additive classifier.

No learned routing.
No learned prototypes.
No EMA memory.
No load-balance loss.
No route loss.
"""

import os
import math
import json
import time
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# Config
# ============================================================
@dataclass
class CFG:
    CUB_ROOT: str = "./data/CUB_200_2011"

    DINO_CACHE_DIR: str = "./artifacts/dino_vitb14_gtbbox_warp224"
    DINO_FEATURE_SUBDIR: str = "features"

    GDINO_BOX_DIR: str = "./artifacts/gdino_part_boxes_gtbbox_warp224_sep"
    GDINO_TRAIN_BOX_FILE: str = "train_part_boxes_gtbbox_warp518.pt"
    GDINO_TEST_BOX_FILE: str = "test_part_boxes_gtbbox_warp518.pt"

    MODEL_IM_SIZE: int = 224

    SAVE_DIR: str = "./runs/step2_fixed_part_prototypes"
    BEST_NAME: str = "best.pth"
    LAST_NAME: str = "last.pth"
    LOG_NAME: str = "history.jsonl"
    PROTOTYPE_NAME: str = "fixed_part_prototypes.pt"

    NUM_CLASSES: int = 200
    PARTS: Tuple[str, ...] = ("beak", "head", "wing", "body", "tail", "feet")
    K_PER_PART: int = 20

    BOX_TARGET_GAUSSIAN: bool = True
    BOX_GAUSSIAN_SIGMA_SCALE: float = 0.50

    USE_CLS_IN_PATCH: bool = False
    CLS_ALPHA: float = 1.0

    SCORE_POOL: str = "max"  # max | avg
    CLASS_THETA_INIT: float = -3.0
    LAMBDA_SPARSE: float = 1e-4

    LR: float = 1e-3
    WEIGHT_DECAY: float = 0.0
    GRAD_CLIP: float = 5.0
    LABEL_SMOOTHING: float = 0.0

    EPOCHS: int = 100
    BATCH_SIZE: int = 128
    NUM_WORKERS: int = 4
    AMP: bool = True
    SEED: int = 0

    # Prototype construction
    MAX_TOKENS_PER_PART: int = 30000
    KMEANS_ITERS: int = 30
    REBUILD_PROTOTYPES: bool = False

    AUTO_RESUME: bool = True
    RESUME_FROM: str = "last"  # last | best | auto

    EPS: float = 1e-9


cfg = CFG()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Args
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Step-2 fixed part prototype baseline")

    p.add_argument("--cub-root", default=os.environ.get("CUB_ROOT", cfg.CUB_ROOT))
    p.add_argument("--dino-cache-dir", default=os.environ.get("DINO_CACHE_DIR", cfg.DINO_CACHE_DIR))
    p.add_argument("--gdino-box-dir", default=os.environ.get("GDINO_BOX_DIR", cfg.GDINO_BOX_DIR))
    p.add_argument("--gdino-train-box-file", default=cfg.GDINO_TRAIN_BOX_FILE)
    p.add_argument("--gdino-test-box-file", default=cfg.GDINO_TEST_BOX_FILE)
    p.add_argument("--save-dir", default=os.environ.get("SAVE_DIR", cfg.SAVE_DIR))
    p.add_argument("--prototype-path", default=None)

    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS)
    p.add_argument("--seed", type=int, default=cfg.SEED)

    p.add_argument("--k-per-part", type=int, default=cfg.K_PER_PART)
    p.add_argument("--max-tokens-per-part", type=int, default=cfg.MAX_TOKENS_PER_PART)
    p.add_argument("--kmeans-iters", type=int, default=cfg.KMEANS_ITERS)
    p.add_argument("--rebuild-prototypes", action="store_true")

    p.add_argument("--lr", type=float, default=cfg.LR)
    p.add_argument("--weight-decay", type=float, default=cfg.WEIGHT_DECAY)
    p.add_argument("--lambda-sparse", type=float, default=cfg.LAMBDA_SPARSE)
    p.add_argument("--class-theta-init", type=float, default=cfg.CLASS_THETA_INIT)
    p.add_argument("--label-smoothing", type=float, default=cfg.LABEL_SMOOTHING)

    p.add_argument("--score-pool", choices=["max", "avg"], default=cfg.SCORE_POOL)
    p.add_argument("--model-im-size", type=int, default=cfg.MODEL_IM_SIZE)

    p.add_argument("--use-cls-in-patch", action="store_true")
    p.add_argument("--no-gaussian", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--resume-from", choices=["last", "best", "auto"], default=cfg.RESUME_FROM)

    return p.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    cfg.CUB_ROOT = os.path.abspath(os.path.expanduser(args.cub_root))
    cfg.DINO_CACHE_DIR = os.path.abspath(os.path.expanduser(args.dino_cache_dir))
    cfg.GDINO_BOX_DIR = os.path.abspath(os.path.expanduser(args.gdino_box_dir))
    cfg.SAVE_DIR = os.path.abspath(os.path.expanduser(args.save_dir))

    cfg.GDINO_TRAIN_BOX_FILE = args.gdino_train_box_file
    cfg.GDINO_TEST_BOX_FILE = args.gdino_test_box_file

    if args.prototype_path is not None:
        cfg.PROTOTYPE_NAME = os.path.abspath(os.path.expanduser(args.prototype_path))

    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.NUM_WORKERS = args.num_workers
    cfg.SEED = args.seed

    cfg.K_PER_PART = args.k_per_part
    cfg.MAX_TOKENS_PER_PART = args.max_tokens_per_part
    cfg.KMEANS_ITERS = args.kmeans_iters
    cfg.REBUILD_PROTOTYPES = bool(args.rebuild_prototypes)

    cfg.LR = args.lr
    cfg.WEIGHT_DECAY = args.weight_decay
    cfg.LAMBDA_SPARSE = args.lambda_sparse
    cfg.CLASS_THETA_INIT = args.class_theta_init
    cfg.LABEL_SMOOTHING = args.label_smoothing

    cfg.SCORE_POOL = args.score_pool
    cfg.MODEL_IM_SIZE = args.model_im_size
    cfg.USE_CLS_IN_PATCH = bool(args.use_cls_in_patch)
    cfg.BOX_TARGET_GAUSSIAN = not args.no_gaussian
    cfg.AMP = not args.no_amp
    cfg.AUTO_RESUME = not args.no_resume
    cfg.RESUME_FROM = args.resume_from


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2n(x: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


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


def build_cub_relpath_order(cub_root: str, split: str) -> List[str]:
    id2path = read_kv_txt(os.path.join(cub_root, "images.txt"))
    id2is_train = read_kv_int_txt(os.path.join(cub_root, "train_test_split.txt"))
    want_train = split == "train"

    relpaths: List[str] = []
    for image_id in sorted(id2path):
        if (id2is_train[image_id] == 1) == want_train:
            relpaths.append(str(id2path[image_id]))
    return relpaths


def dino_cache_paths(cache_dir: str, split: str) -> Dict[str, str]:
    feat_dir = os.path.join(cache_dir, cfg.DINO_FEATURE_SUBDIR)
    return {
        "meta": os.path.join(feat_dir, f"{split}_meta.json"),
        "cls": os.path.join(feat_dir, f"{split}_cls.dat"),
        "patch": os.path.join(feat_dir, f"{split}_patch.dat"),
        "y": os.path.join(feat_dir, f"{split}_labels.npy"),
    }


def load_dino_memmaps(cache_dir: str, split: str):
    paths = dino_cache_paths(cache_dir, split)
    for name, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing DINO cache {name}: {path}")

    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    dtype = np.float16 if meta.get("dtype", "float16") == "float16" else np.float32
    n = int(meta["N"])
    d = int(meta["D"])
    p = int(meta["P"])

    cls_mm = np.memmap(paths["cls"], mode="r", dtype=dtype, shape=(n, d))
    patch_mm = np.memmap(paths["patch"], mode="r", dtype=dtype, shape=(n, p, d))
    labels = np.load(paths["y"]).astype(np.int64)

    return cls_mm, patch_mm, labels, meta


def preflight_check() -> None:
    missing = []

    required_cub = [
        "images.txt",
        "image_class_labels.txt",
        "train_test_split.txt",
        "bounding_boxes.txt",
        "images",
    ]
    for x in required_cub:
        path = os.path.join(cfg.CUB_ROOT, x)
        if not os.path.exists(path):
            missing.append(path)

    for split in ("train", "test"):
        for path in dino_cache_paths(cfg.DINO_CACHE_DIR, split).values():
            if not os.path.isfile(path):
                missing.append(path)

    for name in (cfg.GDINO_TRAIN_BOX_FILE, cfg.GDINO_TEST_BOX_FILE):
        path = os.path.join(cfg.GDINO_BOX_DIR, name)
        if not os.path.isfile(path):
            missing.append(path)

    if missing:
        raise FileNotFoundError(
            "Missing required inputs:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )


# ============================================================
# GDINO box loading
# ============================================================
def _as_xyxy_2d_np(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        x = np.asarray(x)
    elif not isinstance(x, np.ndarray):
        return None

    if x.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    x = x.astype(np.float32, copy=False)

    if x.ndim == 1 and x.shape[0] == 4:
        return x.reshape(1, 4)

    if x.ndim >= 2 and x.shape[-1] == 4:
        return x.reshape(-1, 4)

    return None


def _parse_one_sample_part_boxes(elem: Any, parts_in_file: List[str]) -> np.ndarray:
    p = len(parts_in_file)

    if isinstance(elem, dict):
        for key in ["boxes", "xyxy", "boxes_xyxy", "part_boxes_xyxy_pix", "part_boxes"]:
            if key in elem:
                arr = _as_xyxy_2d_np(elem[key])
                if arr is not None:
                    out = np.full((p, 1, 4), -1.0, dtype=np.float32)
                    m = min(p, arr.shape[0])
                    out[:m, 0] = arr[:m]
                    return out

        part_dict = elem["parts"] if "parts" in elem and isinstance(elem["parts"], dict) else elem
        out = np.full((p, 1, 4), -1.0, dtype=np.float32)

        for i, part_name in enumerate(parts_in_file):
            if part_name not in part_dict:
                continue
            arr = _as_xyxy_2d_np(part_dict[part_name])
            if arr is None or arr.size == 0:
                continue
            valid = (
                (arr[:, 0] >= 0)
                & (arr[:, 2] > arr[:, 0])
                & (arr[:, 3] > arr[:, 1])
            )
            arr = arr[valid]
            if arr.size > 0:
                out[i, 0] = arr[0]

        return out

    arr = _as_xyxy_2d_np(elem)
    if arr is None:
        raise TypeError(f"Unsupported box element: {type(elem)}")

    out = np.full((p, 1, 4), -1.0, dtype=np.float32)
    m = min(p, arr.shape[0])
    out[:m, 0] = arr[:m]
    return out


def _load_gdino_part_boxes_tensor(gd: Dict[str, Any], parts_in_file: List[str]) -> torch.Tensor:
    pb = gd["part_boxes_xyxy_pix"]
    p_need = len(parts_in_file)

    if isinstance(pb, torch.Tensor):
        pb = pb.float()

        if pb.ndim == 3 and pb.shape[-1] == 4:
            pb = pb.unsqueeze(2)

        if pb.ndim == 4 and pb.shape[-1] == 4:
            out = torch.full(
                (pb.shape[0], p_need, pb.shape[2], 4),
                -1.0,
                dtype=torch.float32,
            )
            m = min(p_need, pb.shape[1])
            out[:, :m] = pb[:, :m]
            return out

        raise ValueError(f"Unexpected part box tensor shape: {tuple(pb.shape)}")

    if isinstance(pb, (list, tuple)):
        parsed = [_parse_one_sample_part_boxes(elem, parts_in_file) for elem in pb]
        max_m = max(x.shape[1] for x in parsed)
        out = torch.full((len(parsed), p_need, max_m, 4), -1.0, dtype=torch.float32)
        for i, arr in enumerate(parsed):
            out[i, :, : arr.shape[1]] = torch.from_numpy(arr)
        return out

    raise TypeError(f"Unsupported part_boxes_xyxy_pix container: {type(pb)}")


# ============================================================
# Dataset
# ============================================================
class CUBDINOWithPartBoxes(Dataset):
    def __init__(self, split: str, gdino_path: str):
        if split not in {"train", "test"}:
            raise ValueError(split)

        self.split = split
        self.cls_mm, self.patch_mm, self.y, meta = load_dino_memmaps(cfg.DINO_CACHE_DIR, split)

        self.N = int(meta["N"])
        self.D = int(meta["D"])
        self.PATCHES = int(meta["P"])

        grid = int(round(math.sqrt(self.PATCHES)))
        if grid * grid != self.PATCHES:
            raise ValueError(f"Patch count must be square, got {self.PATCHES}")

        self.H = grid
        self.W = grid

        self.relpaths = build_cub_relpath_order(cfg.CUB_ROOT, split)
        if len(self.relpaths) != self.N:
            raise RuntimeError(f"CUB order mismatch: {len(self.relpaths)} vs {self.N}")

        gd = torch.load(gdino_path, map_location="cpu")
        if "relpaths" not in gd or "part_boxes_xyxy_pix" not in gd:
            raise KeyError(f"Bad GDINO cache: {gdino_path}")

        src_size = int(gd.get("img_size", cfg.MODEL_IM_SIZE))
        scale = float(cfg.MODEL_IM_SIZE) / float(src_size)

        if "parts" in gd:
            parts_in_file = [str(x).lower() for x in gd["parts"]]
        else:
            parts_in_file = ["beak", "head", "wing", "body", "tail", "feet"]

        boxes_pf = _load_gdino_part_boxes_tensor(gd, parts_in_file)

        gd_relpaths = [str(x) for x in gd["relpaths"]]
        gd_index = {rp: i for i, rp in enumerate(gd_relpaths)}

        def match_indices(target: str) -> List[int]:
            t = target.lower()
            if t == "wing":
                return [i for i, n in enumerate(parts_in_file) if "wing" in n]
            if t == "beak":
                return [i for i, n in enumerate(parts_in_file) if "beak" in n or "bill" in n]
            if t == "feet":
                return [i for i, n in enumerate(parts_in_file) if "feet" in n or "foot" in n or "leg" in n]
            return [i for i, n in enumerate(parts_in_file) if n == t]

        max_boxes = int(boxes_pf.shape[2])
        aligned_src = torch.full(
            (boxes_pf.shape[0], len(cfg.PARTS), max_boxes, 4),
            -1.0,
            dtype=torch.float32,
        )

        for target_p, part_name in enumerate(cfg.PARTS):
            source_indices = match_indices(part_name)
            if not source_indices:
                continue

            candidates = boxes_pf[:, source_indices].reshape(boxes_pf.shape[0], -1, 4)
            valid = (
                (candidates[..., 0] >= 0)
                & (candidates[..., 2] > candidates[..., 0])
                & (candidates[..., 3] > candidates[..., 1])
            )

            for image_i in range(candidates.shape[0]):
                selected = candidates[image_i][valid[image_i]][:max_boxes]
                if selected.numel() > 0:
                    aligned_src[image_i, target_p, : selected.shape[0]] = selected

        valid_aligned = aligned_src[..., 0] >= 0
        aligned_model = aligned_src.clone()
        aligned_model[valid_aligned] = aligned_src[valid_aligned] * scale

        self.boxes = torch.full(
            (self.N, len(cfg.PARTS), max_boxes, 4),
            -1.0,
            dtype=torch.float32,
        )

        missing = 0
        for i, rp in enumerate(self.relpaths):
            j = gd_index.get(rp)
            if j is None:
                missing += 1
            else:
                self.boxes[i] = aligned_model[j]

        print(
            f"[{split}] N={self.N}; D={self.D}; grid={self.H}x{self.W}; "
            f"GDINO aligned={self.N - missing}/{self.N}; missing={missing}; "
            f"box_slots={max_boxes}; src_size={src_size}; scale={scale:.4f}"
        )

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        return int(idx), self.boxes[idx], int(self.y[idx])


def make_collate(ds: CUBDINOWithPartBoxes):
    def collate(batch):
        idxs = np.asarray([x[0] for x in batch], dtype=np.int64)
        boxes = torch.stack([x[1] for x in batch], dim=0)
        labels = torch.tensor([x[2] for x in batch], dtype=torch.long)

        patch = ds.patch_mm[idxs].astype(np.float32, copy=False)

        if cfg.USE_CLS_IN_PATCH:
            cls = ds.cls_mm[idxs].astype(np.float32, copy=False)
            patch = patch + float(cfg.CLS_ALPHA) * cls[:, None, :]

        patch = patch.copy()
        return torch.from_numpy(patch), boxes, labels

    return collate


# ============================================================
# Boxes -> q_sem
# ============================================================
@torch.no_grad()
def boxes_to_soft_q(
    boxes: torch.Tensor,
    h: int,
    w: int,
    image_size: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if boxes.ndim == 3 and boxes.shape[-1] == 4:
        boxes = boxes.unsqueeze(2)

    if boxes.ndim != 4 or boxes.shape[-1] != 4:
        raise ValueError(
            "Expected boxes with shape (B,P,4) or (B,P,M,4), "
            f"got {tuple(boxes.shape)}"
        )

    bsz, parts, max_boxes, _ = boxes.shape
    device = boxes.device

    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * (
        float(image_size) / h
    )
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * (
        float(image_size) / w
    )
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    q = torch.zeros((bsz, parts, h, w), device=device, dtype=torch.float32)
    valid_bp = torch.zeros((bsz, parts), device=device, dtype=torch.bool)

    for b in range(bsz):
        for p in range(parts):
            combined = torch.zeros((h, w), device=device, dtype=torch.float32)

            for m in range(max_boxes):
                x1, y1, x2, y2 = boxes[b, p, m]

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
                    dist = (xx - cx).pow(2) + (yy - cy).pow(2)
                    mask = torch.zeros_like(mask)
                    mask.view(-1)[int(dist.argmin().item())] = True

                if cfg.BOX_TARGET_GAUSSIAN:
                    cx = 0.5 * (x1 + x2)
                    cy = 0.5 * (y1 + y2)
                    sx = (
                        (x2 - x1) * cfg.BOX_GAUSSIAN_SIGMA_SCALE
                    ).clamp_min(float(image_size) / w)
                    sy = (
                        (y2 - y1) * cfg.BOX_GAUSSIAN_SIGMA_SCALE
                    ).clamp_min(float(image_size) / h)

                    weight = torch.exp(
                        -0.5
                        * (
                            ((xx - cx) / sx).pow(2)
                            + ((yy - cy) / sy).pow(2)
                        )
                    ) * mask.float()
                else:
                    weight = mask.float()

                combined = torch.maximum(combined, weight)

            if combined.sum() > 0:
                q[b, p] = combined / combined.sum().clamp_min(eps)
                valid_bp[b, p] = True

    return q.flatten(2), valid_bp


# ============================================================
# Prototype construction
# ============================================================
@torch.no_grad()
def collect_part_tokens(
    loader: DataLoader,
    h: int,
    w: int,
    dim: int,
) -> List[torch.Tensor]:
    pools: List[List[torch.Tensor]] = [[] for _ in cfg.PARTS]
    counts = [0 for _ in cfg.PARTS]

    pbar = tqdm(loader, desc="Collect part tokens", ncols=140)

    for patch, boxes, _ in pbar:
        patch = patch.to(DEVICE, non_blocking=True).float()
        boxes = boxes.to(DEVICE, non_blocking=True).float()

        q_sem, valid_bp = boxes_to_soft_q(boxes, h, w, cfg.MODEL_IM_SIZE, cfg.EPS)
        patch = l2n(patch, dim=-1, eps=cfg.EPS)

        for p in range(len(cfg.PARTS)):
            if counts[p] >= cfg.MAX_TOKENS_PER_PART:
                continue

            mask = q_sem[:, p] > 0
            feats = patch[mask]

            if feats.numel() == 0:
                continue

            room = cfg.MAX_TOKENS_PER_PART - counts[p]
            if feats.shape[0] > room:
                idx = torch.randperm(feats.shape[0], device=feats.device)[:room]
                feats = feats[idx]

            pools[p].append(feats.detach().cpu().to(torch.float16))
            counts[p] += feats.shape[0]

        pbar.set_postfix({cfg.PARTS[i]: counts[i] for i in range(len(cfg.PARTS))})

        if all(c >= cfg.MAX_TOKENS_PER_PART for c in counts):
            break

    out: List[torch.Tensor] = []
    for p, name in enumerate(cfg.PARTS):
        if pools[p]:
            x = torch.cat(pools[p], dim=0).float()
        else:
            x = torch.empty((0, dim), dtype=torch.float32)
        print(f"[Collect] {name}: {x.shape[0]} tokens")
        out.append(x)

    return out


@torch.no_grad()
def spherical_kmeans(
    x_cpu: torch.Tensor,
    k: int,
    iters: int,
    seed: int,
    name: str,
) -> torch.Tensor:
    if x_cpu.ndim != 2:
        raise ValueError(f"Expected 2D tokens, got {tuple(x_cpu.shape)}")

    n, d = x_cpu.shape
    if n == 0:
        print(f"[KMeans] warning: no tokens for {name}; using random prototypes")
        return l2n(torch.randn(k, d), dim=-1, eps=cfg.EPS)

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    x = l2n(x_cpu.float(), dim=-1, eps=cfg.EPS).to(DEVICE)

    if n >= k:
        init_idx = torch.randperm(n, generator=g)[:k].to(DEVICE)
    else:
        init_idx = torch.randint(0, n, (k,), generator=g).to(DEVICE)

    centroids = x[init_idx].clone()
    centroids = l2n(centroids, dim=-1, eps=cfg.EPS)

    for it in range(1, iters + 1):
        sim = x @ centroids.t()
        assign = sim.argmax(dim=1)

        sums = torch.zeros((k, d), device=DEVICE, dtype=torch.float32)
        sums.index_add_(0, assign, x)

        counts = torch.bincount(assign, minlength=k).float().to(DEVICE)

        empty = counts <= 0
        if empty.any():
            repl = torch.randint(0, n, (int(empty.sum().item()),), device=DEVICE)
            sums[empty] = x[repl]
            counts[empty] = 1.0

        centroids = sums / counts.unsqueeze(1).clamp_min(1.0)
        centroids = l2n(centroids, dim=-1, eps=cfg.EPS)

        if it == 1 or it == iters or it % 10 == 0:
            obj = sim.max(dim=1).values.mean().item()
            print(f"[KMeans] {name} iter={it:03d}/{iters} mean_cos={obj:.4f}")

    return centroids.detach().cpu()


@torch.no_grad()
def build_or_load_prototypes(
    train_loader_for_collect: DataLoader,
    h: int,
    w: int,
    dim: int,
    prototype_path: str,
) -> torch.Tensor:
    if os.path.isfile(prototype_path) and not cfg.REBUILD_PROTOTYPES:
        data = torch.load(prototype_path, map_location="cpu")
        prototypes = data["prototypes"].float()
        if tuple(data.get("parts", cfg.PARTS)) != tuple(cfg.PARTS):
            raise RuntimeError(f"Prototype part mismatch: {data.get('parts')} vs {cfg.PARTS}")
        if int(data.get("k_per_part", prototypes.shape[1])) != cfg.K_PER_PART:
            raise RuntimeError(
                f"Prototype K mismatch: file K={data.get('k_per_part')} cfg K={cfg.K_PER_PART}"
            )
        print(f"[Prototypes] loaded {prototype_path}: {tuple(prototypes.shape)}")
        return prototypes

    print("[Prototypes] building fixed part prototypes")
    pools = collect_part_tokens(train_loader_for_collect, h, w, dim)

    part_centers = []
    for p, name in enumerate(cfg.PARTS):
        centers = spherical_kmeans(
            pools[p],
            k=cfg.K_PER_PART,
            iters=cfg.KMEANS_ITERS,
            seed=cfg.SEED + 1009 * (p + 1),
            name=name,
        )
        part_centers.append(centers)

    prototypes = torch.stack(part_centers, dim=0).float()
    prototypes = l2n(prototypes, dim=-1, eps=cfg.EPS)

    os.makedirs(os.path.dirname(prototype_path), exist_ok=True)
    torch.save(
        {
            "prototypes": prototypes,
            "parts": list(cfg.PARTS),
            "k_per_part": int(cfg.K_PER_PART),
            "dim": int(dim),
            "cfg": asdict(cfg),
        },
        prototype_path,
    )
    print(f"[Prototypes] saved {prototype_path}: {tuple(prototypes.shape)}")
    return prototypes


# ============================================================
# Model
# ============================================================
class FixedPartPrototypeClassifier(nn.Module):
    def __init__(self, prototypes: torch.Tensor, classes: int):
        super().__init__()

        if prototypes.ndim != 3:
            raise ValueError(f"Expected prototypes (P,K,D), got {tuple(prototypes.shape)}")

        prototypes = l2n(prototypes.float(), dim=-1, eps=cfg.EPS)

        self.register_buffer("prototypes", prototypes)
        self.parts = int(prototypes.shape[0])
        self.k = int(prototypes.shape[1])
        self.dim = int(prototypes.shape[2])
        self.classes = int(classes)

        self.class_theta = nn.Parameter(
            torch.full((classes, self.parts, self.k), float(cfg.CLASS_THETA_INIT))
        )
        self.class_bias = nn.Parameter(torch.zeros(classes))

    def class_weights(self) -> torch.Tensor:
        return F.softplus(self.class_theta.float())

    def prototype_scores(
        self,
        patch: torch.Tensor,
        q_sem: torch.Tensor,
        valid_bp: torch.Tensor,
    ) -> torch.Tensor:
        patch = l2n(patch.float(), dim=-1, eps=cfg.EPS)
        proto = l2n(self.prototypes.float(), dim=-1, eps=cfg.EPS)

        sim = torch.einsum("bnc,pkc->bpnk", patch, proto)
        sim = F.relu(sim)

        if cfg.SCORE_POOL == "avg":
            score = (q_sem.unsqueeze(-1) * sim).sum(dim=2)
            score = score * valid_bp.float().unsqueeze(-1)
            return score

        if cfg.SCORE_POOL == "max":
            mask = q_sem > 0
            sim_masked = sim.masked_fill(~mask.unsqueeze(-1), -1e4)
            score = sim_masked.max(dim=2).values
            score = torch.where(
                valid_bp.unsqueeze(-1),
                score.clamp_min(0.0),
                torch.zeros_like(score),
            )
            return score

        raise ValueError(f"Bad SCORE_POOL={cfg.SCORE_POOL}")

    def forward(
        self,
        patch: torch.Tensor,
        q_sem: torch.Tensor,
        valid_bp: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        proto_score = self.prototype_scores(patch, q_sem, valid_bp)
        class_weight = self.class_weights()

        contributions = proto_score[:, None] * class_weight[None]
        logits = self.class_bias.float().view(1, -1) + contributions.sum(dim=(2, 3))

        return {
            "logits": logits,
            "proto_score": proto_score,
            "class_weight": class_weight,
            "contributions": contributions,
        }


# ============================================================
# Train / Eval
# ============================================================
@torch.inference_mode()
def evaluate(
    model: FixedPartPrototypeClassifier,
    loader: DataLoader,
    h: int,
    w: int,
) -> Dict[str, Any]:
    model.eval()

    total = 0
    correct = 0
    ce_sum = 0.0

    valid_part_sum = torch.zeros(len(cfg.PARTS), dtype=torch.float64)
    score_sum = torch.zeros(len(cfg.PARTS), cfg.K_PER_PART, dtype=torch.float64)
    score_count = 0

    top_contrib_cover_sum = 0.0

    for patch, boxes, y in tqdm(loader, desc="Eval", ncols=140):
        patch = patch.to(DEVICE, non_blocking=True).float()
        boxes = boxes.to(DEVICE, non_blocking=True).float()
        y = y.to(DEVICE, non_blocking=True)

        q_sem, valid_bp = boxes_to_soft_q(boxes, h, w, cfg.MODEL_IM_SIZE, cfg.EPS)
        out = model(patch, q_sem, valid_bp)

        bs = y.shape[0]
        total += bs
        pred = out["logits"].argmax(dim=1)
        correct += (pred == y).sum().item()
        ce_sum += F.cross_entropy(out["logits"], y, reduction="sum").item()

        valid_part_sum += valid_bp.sum(dim=0).double().cpu()
        score_sum += out["proto_score"].sum(dim=0).double().cpu()
        score_count += bs

        contrib_pred = out["contributions"][torch.arange(bs, device=DEVICE), pred]
        flat = contrib_pred.flatten(1)
        sorted_vals, _ = flat.sort(dim=1, descending=True)
        total_pos = sorted_vals.sum(dim=1).clamp_min(cfg.EPS)
        k10 = min(10, sorted_vals.shape[1])
        top_contrib_cover_sum += (sorted_vals[:, :k10].sum(dim=1) / total_pos).sum().item()

    avg_score = score_sum / max(1, score_count)
    active_by_score = (avg_score > 1e-3).sum(dim=1)

    class_weight = model.class_weights().detach().cpu()
    active_by_weight = (class_weight > 0.05).float().sum(dim=(0, 2))

    return {
        "acc": correct / max(1, total),
        "ce": ce_sum / max(1, total),
        "valid_part_rate": (valid_part_sum / max(1, total)).tolist(),
        "active_prototypes_by_score": active_by_score.tolist(),
        "active_prototypes_by_weight": active_by_weight.tolist(),
        "top10_contribution_cover": top_contrib_cover_sum / max(1, total),
    }


def pick_resume_path(last_path: str, best_path: str) -> Optional[str]:
    mode = cfg.RESUME_FROM.lower()

    if mode == "last":
        return last_path if os.path.isfile(last_path) else None

    if mode == "best":
        return best_path if os.path.isfile(best_path) else None

    if mode == "auto":
        if os.path.isfile(last_path):
            return last_path
        if os.path.isfile(best_path):
            return best_path
        return None

    raise ValueError(f"Bad RESUME_FROM: {cfg.RESUME_FROM}")


def save_checkpoint(
    path: str,
    epoch: int,
    best_acc: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    eval_metrics: Dict[str, Any],
    prototype_path: str,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "best_acc": float(best_acc),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "eval": eval_metrics,
            "prototype_path": prototype_path,
            "cfg": asdict(cfg),
        },
        path,
    )


def main() -> None:
    preflight_check()
    set_seed(cfg.SEED)

    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    best_path = os.path.join(cfg.SAVE_DIR, cfg.BEST_NAME)
    last_path = os.path.join(cfg.SAVE_DIR, cfg.LAST_NAME)
    log_path = os.path.join(cfg.SAVE_DIR, cfg.LOG_NAME)

    if os.path.isabs(cfg.PROTOTYPE_NAME):
        prototype_path = cfg.PROTOTYPE_NAME
    else:
        prototype_path = os.path.join(cfg.SAVE_DIR, cfg.PROTOTYPE_NAME)

    train_box_path = os.path.join(cfg.GDINO_BOX_DIR, cfg.GDINO_TRAIN_BOX_FILE)
    test_box_path = os.path.join(cfg.GDINO_BOX_DIR, cfg.GDINO_TEST_BOX_FILE)

    ds_train = CUBDINOWithPartBoxes("train", train_box_path)
    ds_test = CUBDINOWithPartBoxes("test", test_box_path)

    if (ds_train.D, ds_train.H, ds_train.W) != (ds_test.D, ds_test.H, ds_test.W):
        raise RuntimeError("Train/test feature shape mismatch")

    pin = DEVICE == "cuda"

    dl_collect = DataLoader(
        ds_train,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin,
        drop_last=False,
        collate_fn=make_collate(ds_train),
    )

    prototypes = build_or_load_prototypes(
        dl_collect,
        h=ds_train.H,
        w=ds_train.W,
        dim=ds_train.D,
        prototype_path=prototype_path,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin,
        drop_last=True,
        collate_fn=make_collate(ds_train),
    )

    dl_test = DataLoader(
        ds_test,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin,
        drop_last=False,
        collate_fn=make_collate(ds_test),
    )

    model = FixedPartPrototypeClassifier(
        prototypes=prototypes,
        classes=cfg.NUM_CLASSES,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        [model.class_theta, model.class_bias],
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.AMP and DEVICE == "cuda"))

    start_epoch = 1
    best_acc = -1.0

    resume_path = pick_resume_path(last_path, best_path) if cfg.AUTO_RESUME else None
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_acc = float(ckpt.get("best_acc", -1.0))
        print(f"[Resume] {resume_path} -> epoch {start_epoch}; best={best_acc:.6f}")

    print(f"[Device] {DEVICE}; AMP={cfg.AMP}")
    print(f"[Data] train={len(ds_train)} test={len(ds_test)} feature={ds_train.D} grid={ds_train.H}x{ds_train.W}")
    print(f"[Parts] {cfg.PARTS}")
    print(f"[Prototypes] shape={tuple(prototypes.shape)} path={prototype_path}")
    print(f"[Model] fixed prototypes; K={cfg.K_PER_PART}; score_pool={cfg.SCORE_POOL}; non-negative additive classifier")
    print(f"[Loss] CE + {cfg.LAMBDA_SPARSE} * mean(softplus(class_theta))")

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        model.train()
        t0 = time.time()

        total = 0
        correct = 0
        ce_sum = 0.0
        sparse_sum = 0.0
        loss_sum = 0.0

        pbar = tqdm(dl_train, desc=f"Train {epoch}/{cfg.EPOCHS}", ncols=180)

        for patch, boxes, y in pbar:
            patch = patch.to(DEVICE, non_blocking=True).float()
            boxes = boxes.to(DEVICE, non_blocking=True).float()
            y = y.to(DEVICE, non_blocking=True)

            q_sem, valid_bp = boxes_to_soft_q(boxes, ds_train.H, ds_train.W, cfg.MODEL_IM_SIZE, cfg.EPS)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.AMP and DEVICE == "cuda")):
                out = model(patch, q_sem, valid_bp)
                loss_ce = F.cross_entropy(
                    out["logits"],
                    y,
                    label_smoothing=cfg.LABEL_SMOOTHING,
                )
                loss_sparse = out["class_weight"].mean()
                loss = loss_ce + float(cfg.LAMBDA_SPARSE) * loss_sparse

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            bs = y.shape[0]
            total += bs
            correct += (out["logits"].argmax(dim=1) == y).sum().item()
            ce_sum += float(loss_ce.detach().item()) * bs
            sparse_sum += float(loss_sparse.detach().item()) * bs
            loss_sum += float(loss.detach().item()) * bs

            pbar.set_postfix(
                {
                    "acc": f"{correct / max(1, total):.4f}",
                    "ce": f"{ce_sum / max(1, total):.4f}",
                    "sparse": f"{sparse_sum / max(1, total):.4f}",
                }
            )

        train_metrics = {
            "acc": correct / max(1, total),
            "loss": loss_sum / max(1, total),
            "ce": ce_sum / max(1, total),
            "sparse": sparse_sum / max(1, total),
        }

        eval_metrics = evaluate(model, dl_test, ds_test.H, ds_test.W)
        elapsed = time.time() - t0

        valid_text = " | ".join(
            f"{name}:{eval_metrics['valid_part_rate'][i]:.3f}"
            for i, name in enumerate(cfg.PARTS)
        )

        active_score_text = " | ".join(
            f"{name}:{int(eval_metrics['active_prototypes_by_score'][i])}/{cfg.K_PER_PART}"
            for i, name in enumerate(cfg.PARTS)
        )

        active_weight_text = " | ".join(
            f"{name}:{int(eval_metrics['active_prototypes_by_weight'][i])}"
            for i, name in enumerate(cfg.PARTS)
        )

        print(
            f"[Epoch {epoch:03d}] {elapsed:.1f}s "
            f"train_acc={train_metrics['acc']:.4f} "
            f"test_acc={eval_metrics['acc']:.4f} "
            f"train_CE={train_metrics['ce']:.4f} "
            f"test_CE={eval_metrics['ce']:.4f} "
            f"sparse={train_metrics['sparse']:.4f} "
            f"top10cover={eval_metrics['top10_contribution_cover']:.3f}"
        )
        print("[Eval valid part rate] " + valid_text)
        print("[Eval active prototypes by score] " + active_score_text)
        print("[Eval active class weights by part] " + active_weight_text)

        record = {
            "epoch": epoch,
            "seconds": elapsed,
            "train": train_metrics,
            "eval": eval_metrics,
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if eval_metrics["acc"] > best_acc:
            best_acc = float(eval_metrics["acc"])
            save_checkpoint(best_path, epoch, best_acc, model, optimizer, scaler, eval_metrics, prototype_path)
            print(f"[Best] acc={best_acc:.6f} -> {best_path}")

        save_checkpoint(last_path, epoch, best_acc, model, optimizer, scaler, eval_metrics, prototype_path)

    print(f"Done. Best test accuracy={best_acc:.6f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Prototype file: {prototype_path}")


if __name__ == "__main__":
    apply_args(parse_args())
    main()
