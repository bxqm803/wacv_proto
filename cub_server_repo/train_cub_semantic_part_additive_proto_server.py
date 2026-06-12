import os
import argparse
import time
import math
import json
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================
@dataclass
class CFG:
    # ---- CUB ----
    CUB_ROOT: str = "./data/CUB_200_2011"

    # ---- cached DINOv2 ViT-B/14 features ----
    # Expected files under <DINO_CACHE_DIR>/features:
    #   train_meta.json, train_cls.dat, train_patch.dat, train_labels.npy
    #   test_meta.json,  test_cls.dat,  test_patch.dat,  test_labels.npy
    DINO_CACHE_DIR: str = "./artifacts/dino_vitb14_bbox224"
    DINO_FEATURE_SUBDIR: str = "features"

    # ---- GroundingDINO part boxes ----
    GDINO_BOX_DIR: str = "./artifacts/gdino_part_boxes_gtbbox_warp518"
    GDINO_TRAIN_BOX_FILE: str = "train_part_boxes_gtbbox_warp518.pt"
    GDINO_TEST_BOX_FILE: str = "test_part_boxes_gtbbox_warp518.pt"
    GDINO_SRC_IM_SIZE: int = 518
    MODEL_IM_SIZE: int = 224

    # ---- output ----
    SAVE_DIR: str = "./run_cub_semantic_part_additive_proto"
    BEST_NAME: str = "best.pth"
    LAST_NAME: str = "last.pth"
    LOG_NAME: str = "history.jsonl"

    # ---- task ----
    NUM_CLASSES: int = 200
    PARTS: Tuple[str, ...] = ("beak", "head", "wing", "body", "tail", "feet")
    K_PER_PART: int = 50

    # IMPORTANT: use local patch tokens only by default.
    # Turning this on re-injects global CLS information into every local token.
    USE_CLS_IN_PATCH: bool = False
    CLS_ALPHA: float = 1.0

    # ---- semantic part router ----
    TAU_PART: float = 0.20
    NULL_LOGIT_INIT: float = 0.0

    # ---- within-part prototypes ----
    TAU_PROTO: float = 0.05
    RESIDUAL_SCALE: float = 0.20
    EMA_RHO: float = 0.80
    EMA_SEM_MIX: float = 0.50
    EMA_MIN_MASS: float = 1e-3

    # ---- GroundingDINO target ----
    BOX_TARGET_GAUSSIAN: bool = True
    BOX_GAUSSIAN_SIGMA_SCALE: float = 0.50

    # ---- losses ----
    LABEL_SMOOTHING: float = 0.0
    LAMBDA_CE: float = 1.0
    LAMBDA_ROUTE: float = 1.0
    ROUTE_FINAL_RATIO: float = 0.25
    ROUTE_DECAY_EPOCHS: int = 60
    LAMBDA_VIS: float = 0.05
    LAMBDA_PROTO_LB: float = 0.02
    LAMBDA_PROTO_DIV: float = 0.02
    PROTO_DIV_MARGIN: float = 0.30
    LAMBDA_CLS_SPARSE: float = 1e-5

    # ---- optimization ----
    LR_ROUTER: float = 3e-5
    LR_PROTO: float = 3e-5
    LR_CLASSIFIER: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 5.0

    # ---- train ----
    EPOCHS: int = 300
    BATCH_SIZE: int = 128
    NUM_WORKERS: int = 0
    AMP: bool = True
    SEED: int = 0

    # ---- semantic bootstrap ----
    BOOTSTRAP_FROM_GDINO: bool = True
    BOOTSTRAP_BATCHES: int = 8
    BOOTSTRAP_MAX_TOKENS_PER_PART: int = 12000

    # ---- visualization ----
    VIZ_ON: bool = True
    VIZ_EVERY: int = 10
    VIZ_N: int = 4
    VIZ_DIR: str = "./viz_cub_semantic_part_additive_proto"

    # ---- resume ----
    AUTO_RESUME: bool = True
    RESUME_FROM: str = "last"  # last | best | auto

    EPS: float = 1e-9


cfg = CFG()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CUB semantic-part additive prototype model from precomputed caches."
    )
    parser.add_argument("--cub-root", default=os.environ.get("CUB_ROOT", cfg.CUB_ROOT))
    parser.add_argument("--dino-cache-dir", default=os.environ.get("DINO_CACHE_DIR", cfg.DINO_CACHE_DIR))
    parser.add_argument("--gdino-box-dir", default=os.environ.get("GDINO_BOX_DIR", cfg.GDINO_BOX_DIR))
    parser.add_argument("--save-dir", default=os.environ.get("SAVE_DIR", cfg.SAVE_DIR))
    parser.add_argument("--viz-dir", default=None, help="Default: <save-dir>/viz")

    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--k-per-part", type=int, default=cfg.K_PER_PART)
    parser.add_argument("--lr-router", type=float, default=cfg.LR_ROUTER)
    parser.add_argument("--lr-proto", type=float, default=cfg.LR_PROTO)
    parser.add_argument("--lr-classifier", type=float, default=cfg.LR_CLASSIFIER)

    parser.add_argument("--use-cls-in-patch", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--resume-from", choices=["last", "best", "auto"], default=cfg.RESUME_FROM)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--no-viz", action="store_true")
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    cfg.CUB_ROOT = os.path.abspath(os.path.expanduser(args.cub_root))
    cfg.DINO_CACHE_DIR = os.path.abspath(os.path.expanduser(args.dino_cache_dir))
    cfg.GDINO_BOX_DIR = os.path.abspath(os.path.expanduser(args.gdino_box_dir))
    cfg.SAVE_DIR = os.path.abspath(os.path.expanduser(args.save_dir))
    cfg.VIZ_DIR = os.path.abspath(os.path.expanduser(args.viz_dir or os.path.join(cfg.SAVE_DIR, "viz")))

    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.NUM_WORKERS = args.num_workers
    cfg.SEED = args.seed
    cfg.K_PER_PART = args.k_per_part
    cfg.LR_ROUTER = args.lr_router
    cfg.LR_PROTO = args.lr_proto
    cfg.LR_CLASSIFIER = args.lr_classifier

    cfg.USE_CLS_IN_PATCH = bool(args.use_cls_in_patch)
    cfg.AMP = not args.no_amp
    cfg.AUTO_RESUME = not args.no_resume
    cfg.RESUME_FROM = args.resume_from
    cfg.BOOTSTRAP_FROM_GDINO = not args.skip_bootstrap
    cfg.VIZ_ON = not args.no_viz


def preflight_check() -> None:
    required_cub = [
        "images.txt", "image_class_labels.txt", "train_test_split.txt",
        "bounding_boxes.txt", "images",
    ]
    missing = [os.path.join(cfg.CUB_ROOT, x) for x in required_cub if not os.path.exists(os.path.join(cfg.CUB_ROOT, x))]

    for split in ("train", "test"):
        for path in dino_cache_paths(cfg.DINO_CACHE_DIR, split).values():
            if not os.path.isfile(path):
                missing.append(path)

    for name in (cfg.GDINO_TRAIN_BOX_FILE, cfg.GDINO_TEST_BOX_FILE):
        path = os.path.join(cfg.GDINO_BOX_DIR, name)
        if not os.path.isfile(path):
            missing.append(path)

    if missing:
        lines = "\n".join(f"  - {x}" for x in missing)
        raise FileNotFoundError(
            "Required inputs are missing:\n"
            f"{lines}\n\n"
            "Build them first with:\n"
            "  bash scripts/download_cub.sh ./data\n"
            "  python tools/build_dino_cache.py --cub-root ./data/CUB_200_2011 --output-dir ./artifacts/dino_vitb14_bbox224\n"
            "  python tools/build_gdino_part_boxes.py --cub-root ./data/CUB_200_2011 --output-dir ./artifacts/gdino_part_boxes_gtbbox_warp518\n"
        )


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


def route_lambda(epoch: int) -> float:
    if cfg.ROUTE_DECAY_EPOCHS <= 0:
        return cfg.LAMBDA_ROUTE
    t = min(max(epoch - 1, 0) / float(cfg.ROUTE_DECAY_EPOCHS), 1.0)
    ratio = 1.0 + t * (cfg.ROUTE_FINAL_RATIO - 1.0)
    return cfg.LAMBDA_ROUTE * ratio


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


# ============================================================
# DINO feature cache
# ============================================================
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


# ============================================================
# GroundingDINO box parsing
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


def _union_xyxy_np(boxes: np.ndarray) -> np.ndarray:
    if boxes is None or boxes.size == 0:
        return np.array([-1, -1, -1, -1], dtype=np.float32)
    valid = (
        (boxes[:, 0] >= 0)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    if not np.any(valid):
        return np.array([-1, -1, -1, -1], dtype=np.float32)
    b = boxes[valid]
    return np.array(
        [b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()],
        dtype=np.float32,
    )


def _parse_one_sample_part_boxes(elem: Any, parts_in_file: List[str]) -> np.ndarray:
    p = len(parts_in_file)
    if isinstance(elem, dict):
        for key in ["boxes", "xyxy", "boxes_xyxy", "part_boxes_xyxy_pix", "part_boxes"]:
            if key in elem:
                arr = _as_xyxy_2d_np(elem[key])
                if arr is not None:
                    out = np.full((p, 4), -1.0, dtype=np.float32)
                    m = min(p, arr.shape[0])
                    out[:m] = arr[:m]
                    return out

        part_dict = elem["parts"] if "parts" in elem and isinstance(elem["parts"], dict) else elem
        out = np.full((p, 4), -1.0, dtype=np.float32)
        for i, part_name in enumerate(parts_in_file):
            if part_name not in part_dict:
                continue
            arr = _as_xyxy_2d_np(part_dict[part_name])
            if arr is not None:
                out[i] = _union_xyxy_np(arr)
        return out

    arr = _as_xyxy_2d_np(elem)
    if arr is None:
        raise TypeError(f"Unsupported box element: {type(elem)}")
    out = np.full((p, 4), -1.0, dtype=np.float32)
    m = min(p, arr.shape[0])
    out[:m] = arr[:m]
    return out


def _load_gdino_part_boxes_tensor(gd: Dict[str, Any], parts_in_file: List[str]) -> torch.Tensor:
    pb = gd["part_boxes_xyxy_pix"]
    p_need = len(parts_in_file)

    if isinstance(pb, torch.Tensor):
        pb = pb.float()
        if pb.ndim == 3 and pb.shape[-1] == 4:
            out = torch.full((pb.shape[0], p_need, 4), -1.0, dtype=torch.float32)
            m = min(p_need, pb.shape[1])
            out[:, :m] = pb[:, :m]
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
            m = min(p_need, merged.shape[1])
            out[:, :m] = merged[:, :m]
            return out
        raise ValueError(f"Unexpected part box tensor shape: {tuple(pb.shape)}")

    if isinstance(pb, (list, tuple)):
        out = torch.full((len(pb), p_need, 4), -1.0, dtype=torch.float32)
        for i, elem in enumerate(pb):
            out[i] = torch.from_numpy(_parse_one_sample_part_boxes(elem, parts_in_file))
        return out

    raise TypeError(f"Unsupported part_boxes_xyxy_pix container: {type(pb)}")


# ============================================================
# Dataset
# ============================================================
class CUBCachedDINOWithBoxes(Dataset):
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
            raise ValueError(f"Patch count must be a square, got {self.PATCHES}")
        self.H = grid
        self.W = grid

        self.relpaths = build_cub_relpath_order(cfg.CUB_ROOT, split)
        if len(self.relpaths) != self.N:
            raise RuntimeError(f"CUB order mismatch: {len(self.relpaths)} vs {self.N}")

        gd = torch.load(gdino_path, map_location="cpu")
        if "relpaths" not in gd or "part_boxes_xyxy_pix" not in gd:
            raise KeyError(f"Bad GDINO cache: {gdino_path}")

        src_size = int(gd.get("img_size", cfg.GDINO_SRC_IM_SIZE))
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

        merged_src = torch.full((boxes_pf.shape[0], len(cfg.PARTS), 4), -1.0, dtype=torch.float32)
        for p, part_name in enumerate(cfg.PARTS):
            indices = match_indices(part_name)
            if not indices:
                continue
            bx = boxes_pf[:, indices, :]
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

        merged_224 = merged_src * scale
        self.boxes = torch.full((self.N, len(cfg.PARTS), 4), -1.0, dtype=torch.float32)
        missing = 0
        for i, rp in enumerate(self.relpaths):
            j = gd_index.get(rp)
            if j is None:
                missing += 1
            else:
                self.boxes[i] = merged_224[j]
        print(f"[{split}] GDINO aligned={self.N - missing}/{self.N}; missing={missing}")

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        return int(idx), self.boxes[idx], int(self.y[idx])


def make_collate(ds: CUBCachedDINOWithBoxes):
    def collate(batch):
        idxs = np.asarray([x[0] for x in batch], dtype=np.int64)
        boxes = torch.stack([x[1] for x in batch], dim=0)
        labels = torch.tensor([x[2] for x in batch], dtype=torch.long)

        patch = ds.patch_mm[idxs].astype(np.float32, copy=False)
        if cfg.USE_CLS_IN_PATCH:
            cls = ds.cls_mm[idxs].astype(np.float32, copy=False)
            patch = patch + float(cfg.CLS_ALPHA) * cls[:, None, :]

        b = patch.shape[0]
        fm = patch.reshape(b, ds.H, ds.W, ds.D).transpose(0, 3, 1, 2).copy()
        return torch.from_numpy(fm), boxes, labels

    return collate


# ============================================================
# Boxes -> soft semantic token targets
# ============================================================
@torch.no_grad()
def boxes_to_soft_q(
    boxes: torch.Tensor,
    h: int,
    w: int,
    image_size: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    boxes: (B,P,4) in xyxy pixel coordinates on image_size x image_size.
    returns:
      q_sem:   (B,P,N), normalized over N for valid boxes
      valid_bp:(B,P) bool
    """
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
                # Tiny box fallback: nearest patch center.
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                dist = (xx - cx).pow(2) + (yy - cy).pow(2)
                flat_idx = int(dist.argmin().item())
                mask = torch.zeros_like(mask)
                mask.view(-1)[flat_idx] = True

            if cfg.BOX_TARGET_GAUSSIAN:
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                sx = ((x2 - x1) * cfg.BOX_GAUSSIAN_SIGMA_SCALE).clamp_min(float(image_size) / w)
                sy = ((y2 - y1) * cfg.BOX_GAUSSIAN_SIGMA_SCALE).clamp_min(float(image_size) / h)
                weight = torch.exp(-0.5 * (((xx - cx) / sx).pow(2) + ((yy - cy) / sy).pow(2)))
                weight = weight * mask.float()
            else:
                weight = mask.float()

            if weight.sum() > 0:
                q[b, p] = weight / weight.sum().clamp_min(eps)
                valid_bp[b, p] = True

    return q.flatten(2), valid_bp


# ============================================================
# Model
# ============================================================
class SemanticPartAdditivePrototypeNet(nn.Module):
    """
    No prediction bypass:
      local tokens -> semantic part maps -> within-part prototypes
      -> non-negative class-specific additive contributions -> logits
    """

    def __init__(self, dim: int, parts: int, prototypes_per_part: int, classes: int):
        super().__init__()
        self.dim = int(dim)
        self.parts = int(parts)
        self.k = int(prototypes_per_part)
        self.classes = int(classes)

        # WHERE: one semantic query per named part.
        self.part_queries = nn.Parameter(torch.randn(parts, dim) * 0.02)
        self.null_logits = nn.Parameter(torch.full((parts,), float(cfg.NULL_LOGIT_INIT)))

        # WHAT: semantic memory plus bounded discriminative residual.
        memory = l2n(torch.randn(parts, prototypes_per_part, dim), dim=-1)
        self.register_buffer("memory", memory)
        self.proto_residual = nn.Parameter(torch.zeros(parts, prototypes_per_part, dim))

        # WHY: exact non-negative class/prototype contributions.
        # softplus(-4) ~= 0.018, so the classifier starts close to sparse/weak.
        self.class_theta = nn.Parameter(torch.full((classes, parts, prototypes_per_part), -4.0))
        self.class_bias = nn.Parameter(torch.zeros(classes))

    def encode_tokens(self, fm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = fm.flatten(2).transpose(1, 2).contiguous()  # B,N,C
        return x, l2n(x.float(), dim=-1, eps=cfg.EPS)

    def effective_prototypes(self) -> torch.Tensor:
        delta = float(cfg.RESIDUAL_SCALE) * torch.tanh(self.proto_residual.float())
        return l2n(self.memory.float() + delta, dim=-1, eps=cfg.EPS)

    def class_weights(self) -> torch.Tensor:
        return F.softplus(self.class_theta.float())

    def forward(self, fm: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, _, h, w = fm.shape
        n = h * w
        x, xn = self.encode_tokens(fm)

        # Independent part distributions over N spatial tokens + one null token.
        # +log(N) makes the single null state competitive with N spatial states.
        q_part = l2n(self.part_queries.float(), dim=-1, eps=cfg.EPS)
        spatial_logits = torch.einsum("bnc,pc->bpn", xn, q_part) / max(cfg.TAU_PART, 1e-6)
        null = self.null_logits.float().view(1, self.parts, 1) + math.log(max(n, 1))
        null = null.expand(bsz, -1, -1)
        part_prob_full = F.softmax(torch.cat([spatial_logits, null], dim=-1), dim=-1)
        part_map = part_prob_full[..., :n]                    # B,P,N
        visibility = 1.0 - part_prob_full[..., -1]           # B,P

        prototypes = self.effective_prototypes()              # P,K,C
        sim = torch.einsum("bnc,pkc->bpnk", xn, prototypes) # B,P,N,K
        proto_assign = F.softmax(sim / max(cfg.TAU_PROTO, 1e-6), dim=-1)
        responsibility = part_map.unsqueeze(-1) * proto_assign

        # A prototype score is both localized and similarity-sensitive.
        positive_match = F.relu(sim)
        proto_score = (responsibility * positive_match).sum(dim=2)  # B,P,K
        utilization = responsibility.sum(dim=2)                      # B,P,K

        class_weight = self.class_weights()                           # C,P,K
        contributions = proto_score[:, None] * class_weight[None]     # B,C,P,K
        logits = self.class_bias.float().view(1, -1) + contributions.sum(dim=(2, 3))

        return {
            "logits": logits,
            "X": x,
            "Xn": xn,
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
        # Self-routed target.
        self_resp = part_map.unsqueeze(-1) * proto_assign
        self_den = self_resp.sum(dim=(0, 2))                           # P,K
        self_num = torch.einsum("bpnk,bnc->pkc", self_resp, xn.float())
        self_target = l2n(self_num / self_den.unsqueeze(-1).clamp_min(cfg.EPS), dim=-1, eps=cfg.EPS)

        # Semantic-region target. Unlike copying one vector to all K prototypes,
        # proto_assign partitions each named region into different within-part modes.
        sem_resp = q_sem.unsqueeze(-1) * proto_assign
        sem_den = sem_resp.sum(dim=(0, 2))                             # P,K
        sem_num = torch.einsum("bpnk,bnc->pkc", sem_resp, xn.float())
        sem_target = l2n(sem_num / sem_den.unsqueeze(-1).clamp_min(cfg.EPS), dim=-1, eps=cfg.EPS)

        valid_part = valid_bp.any(dim=0).float().view(self.parts, 1, 1)
        sem_valid_pk = (sem_den > cfg.EMA_MIN_MASS).float().unsqueeze(-1)
        alpha = float(cfg.EMA_SEM_MIX) * valid_part * sem_valid_pk
        target = l2n((1.0 - alpha) * self_target + alpha * sem_target, dim=-1, eps=cfg.EPS)

        update_mask = ((self_den > cfg.EMA_MIN_MASS) | (sem_den > cfg.EMA_MIN_MASS)).unsqueeze(-1)
        candidate = l2n(
            float(cfg.EMA_RHO) * self.memory.float()
            + (1.0 - float(cfg.EMA_RHO)) * target,
            dim=-1,
            eps=cfg.EPS,
        )
        self.memory.copy_(torch.where(update_mask, candidate, self.memory.float()))


# ============================================================
# Losses
# ============================================================
def semantic_route_loss(
    part_map: torch.Tensor,
    q_sem: torch.Tensor,
    valid_bp: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    pred = part_map / part_map.sum(dim=-1, keepdim=True).clamp_min(eps)
    ce = -(q_sem * pred.clamp_min(eps).log()).sum(dim=-1)  # B,P
    valid = valid_bp.float()
    return (ce * valid).sum() / valid.sum().clamp_min(1.0)


def visible_part_loss(visibility: torch.Tensor, valid_bp: torch.Tensor, eps: float) -> torch.Tensor:
    # A missing GDINO box is not treated as proof that the part is absent.
    valid = valid_bp.float()
    loss = -visibility.clamp_min(eps).log()
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def semantic_proto_load_balance_loss(
    proto_assign: torch.Tensor,
    q_sem: torch.Tensor,
    valid_bp: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # Use only semantic regions to measure K-way usage inside each named part.
    usage = (q_sem.unsqueeze(-1) * proto_assign).sum(dim=(0, 2))  # P,K
    part_valid = valid_bp.any(dim=0)
    if not part_valid.any():
        return usage.new_tensor(0.0)
    prob = usage / usage.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(prob.clamp_min(eps) * prob.clamp_min(eps).log()).sum(dim=-1)
    return (math.log(max(2, usage.shape[-1])) - entropy[part_valid]).mean()


def within_part_prototype_diversity_loss(
    prototypes: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    q = l2n(prototypes.float(), dim=-1, eps=cfg.EPS)
    sim = torch.matmul(q, q.transpose(1, 2))  # P,K,K
    k = q.shape[1]
    eye = torch.eye(k, device=q.device, dtype=torch.bool).unsqueeze(0)
    penalty = F.relu(sim - float(margin)).pow(2).masked_fill(eye, 0.0)
    return penalty.sum() / max(1, q.shape[0] * k * (k - 1))


def classifier_sparsity_loss(class_weight: torch.Tensor) -> torch.Tensor:
    return class_weight.mean()


# ============================================================
# Semantic initialization
# ============================================================
@torch.no_grad()
def bootstrap_from_gdino(
    model: SemanticPartAdditivePrototypeNet,
    loader: DataLoader,
    h: int,
    w: int,
) -> None:
    print(f"[Bootstrap] collecting up to {cfg.BOOTSTRAP_BATCHES} batches")
    sums = torch.zeros((model.parts, model.dim), device=DEVICE)
    counts = torch.zeros((model.parts,), device=DEVICE)
    pools: List[List[torch.Tensor]] = [[] for _ in range(model.parts)]
    pool_sizes = [0 for _ in range(model.parts)]

    for bi, (fm, boxes, _) in enumerate(loader):
        if bi >= cfg.BOOTSTRAP_BATCHES:
            break
        fm = fm.to(DEVICE, non_blocking=True).float()
        boxes = boxes.to(DEVICE, non_blocking=True).float()
        _, xn = model.encode_tokens(fm)
        q_sem, valid = boxes_to_soft_q(boxes, h, w, cfg.MODEL_IM_SIZE, cfg.EPS)

        sums += torch.einsum("bpn,bnc->pc", q_sem, xn)
        counts += q_sem.sum(dim=(0, 2))

        for p in range(model.parts):
            mask = q_sem[:, p] > 0
            feats = xn[mask]
            if feats.numel() == 0:
                continue
            room = cfg.BOOTSTRAP_MAX_TOKENS_PER_PART - pool_sizes[p]
            if room <= 0:
                continue
            if feats.shape[0] > room:
                perm = torch.randperm(feats.shape[0], device=feats.device)[:room]
                feats = feats[perm]
            pools[p].append(feats.detach().cpu())
            pool_sizes[p] += feats.shape[0]

    query_target = l2n(sums / counts.unsqueeze(-1).clamp_min(cfg.EPS), dim=-1, eps=cfg.EPS)
    query_valid = counts > 0
    model.part_queries.data[query_valid] = query_target[query_valid].to(model.part_queries.dtype)

    for p in range(model.parts):
        if not pools[p]:
            print(f"[Bootstrap] warning: no semantic tokens for {cfg.PARTS[p]}")
            continue
        feats = l2n(torch.cat(pools[p], dim=0).to(DEVICE).float(), dim=-1, eps=cfg.EPS)
        if feats.shape[0] >= model.k:
            idx = torch.randperm(feats.shape[0], device=DEVICE)[:model.k]
        else:
            idx = torch.randint(0, feats.shape[0], (model.k,), device=DEVICE)
        model.memory[p].copy_(feats[idx])
        print(f"[Bootstrap] {cfg.PARTS[p]}: pool={feats.shape[0]} -> K={model.k}")


# ============================================================
# Evaluation and explanations
# ============================================================
@torch.inference_mode()
def evaluate(
    model: SemanticPartAdditivePrototypeNet,
    loader: DataLoader,
    h: int,
    w: int,
) -> Dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    ce_sum = 0.0
    route_sum = 0.0
    route_count = 0
    visibility_sum = torch.zeros(model.parts, dtype=torch.float64)
    utilization_sum = torch.zeros(model.parts, model.k, dtype=torch.float64)

    for fm, boxes, y in tqdm(loader, desc="Eval", ncols=140):
        fm = fm.to(DEVICE, non_blocking=True).float()
        boxes = boxes.to(DEVICE, non_blocking=True).float()
        y = y.to(DEVICE, non_blocking=True)
        out = model(fm)

        bs = y.shape[0]
        total += bs
        correct += (out["logits"].argmax(dim=1) == y).sum().item()
        ce_sum += F.cross_entropy(out["logits"], y, reduction="sum").item()

        q_sem, valid = boxes_to_soft_q(boxes, h, w, cfg.MODEL_IM_SIZE, cfg.EPS)
        n_valid = int(valid.sum().item())
        if n_valid > 0:
            route_sum += float(semantic_route_loss(out["part_map"], q_sem, valid, cfg.EPS).item()) * n_valid
            route_count += n_valid

        visibility_sum += out["visibility"].sum(dim=0).double().cpu()
        utilization_sum += out["utilization"].sum(dim=0).double().cpu()

    util = utilization_sum / max(1, total)
    active_per_part = (util > 1e-4).sum(dim=1)
    return {
        "acc": correct / max(1, total),
        "ce": ce_sum / max(1, total),
        "route_ce": route_sum / max(1, route_count),
        "visibility_per_part": (visibility_sum / max(1, total)).tolist(),
        "active_prototypes_per_part": active_per_part.tolist(),
    }


@torch.inference_mode()
def explain_one(
    model: SemanticPartAdditivePrototypeNet,
    fm: torch.Tensor,
    topk: int = 10,
) -> Dict[str, Any]:
    model.eval()
    out = model(fm)
    pred = int(out["logits"].argmax(dim=1).item())
    contrib = out["contributions"][0, pred]  # P,K
    flat = contrib.flatten()
    k = min(topk, flat.numel())
    values, indices = torch.topk(flat, k=k)
    entries = []
    for value, idx in zip(values.tolist(), indices.tolist()):
        p = idx // model.k
        proto = idx % model.k
        entries.append({
            "part": cfg.PARTS[p],
            "prototype": int(proto),
            "contribution": float(value),
            "prototype_score": float(out["proto_score"][0, p, proto].item()),
            "class_weight": float(out["class_weight"][pred, p, proto].item()),
        })
    return {
        "prediction": pred,
        "logit": float(out["logits"][0, pred].item()),
        "bias": float(model.class_bias[pred].item()),
        "top_contributions": entries,
        "part_contributions": {
            cfg.PARTS[p]: float(contrib[p].sum().item()) for p in range(model.parts)
        },
    }


@torch.inference_mode()
def save_visualizations(
    epoch: int,
    model: SemanticPartAdditivePrototypeNet,
    ds: CUBCachedDINOWithBoxes,
) -> None:
    if not cfg.VIZ_ON:
        return
    os.makedirs(cfg.VIZ_DIR, exist_ok=True)
    out_dir = os.path.join(cfg.VIZ_DIR, f"epoch_{epoch:04d}")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(cfg.SEED + epoch)
    indices = rng.choice(len(ds), size=min(cfg.VIZ_N, len(ds)), replace=False).tolist()

    for idx in indices:
        patch = ds.patch_mm[idx].astype(np.float32, copy=False)
        if cfg.USE_CLS_IN_PATCH:
            cls = ds.cls_mm[idx].astype(np.float32, copy=False)
            patch = patch + float(cfg.CLS_ALPHA) * cls[None]
        fm_np = patch.reshape(ds.H, ds.W, ds.D).transpose(2, 0, 1).copy()
        fm = torch.from_numpy(fm_np).unsqueeze(0).to(DEVICE).float()
        boxes = ds.boxes[idx].unsqueeze(0).to(DEVICE).float()
        q_sem, _ = boxes_to_soft_q(boxes, ds.H, ds.W, cfg.MODEL_IM_SIZE, cfg.EPS)
        out = model(fm)

        pred_map = out["part_map"][0].reshape(model.parts, ds.H, ds.W).cpu().numpy()
        target_map = q_sem[0].reshape(model.parts, ds.H, ds.W).cpu().numpy()

        fig, axes = plt.subplots(2, model.parts, figsize=(2.4 * model.parts, 5.0), squeeze=False)
        for p, name in enumerate(cfg.PARTS):
            axes[0, p].imshow(pred_map[p])
            axes[0, p].set_title(f"pred | {name}")
            axes[0, p].axis("off")
            axes[1, p].imshow(target_map[p])
            axes[1, p].set_title(f"GDINO | {name}")
            axes[1, p].axis("off")
        fig.suptitle(f"epoch={epoch} idx={idx} y={int(ds.y[idx])} pred={int(out['logits'].argmax(1).item())}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"idx_{idx}_part_maps.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

        explanation = explain_one(model, fm, topk=12)
        explanation["index"] = int(idx)
        explanation["label"] = int(ds.y[idx])
        explanation["relpath"] = ds.relpaths[idx]
        with open(os.path.join(out_dir, f"idx_{idx}_explanation.json"), "w", encoding="utf-8") as f:
            json.dump(explanation, f, ensure_ascii=False, indent=2)


# ============================================================
# Resume/save
# ============================================================
def pick_resume_path(last_path: str, best_path: str) -> Optional[str]:
    mode = cfg.RESUME_FROM.lower()
    if mode == "last":
        return last_path if os.path.isfile(last_path) else None
    if mode == "best":
        return best_path if os.path.isfile(best_path) else None
    if mode == "auto":
        if os.path.isfile(last_path):
            return last_path
        return best_path if os.path.isfile(best_path) else None
    raise ValueError(f"Unknown RESUME_FROM={cfg.RESUME_FROM}")


def save_checkpoint(
    path: str,
    epoch: int,
    best_acc: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    eval_metrics: Dict[str, Any],
) -> None:
    torch.save({
        "epoch": int(epoch),
        "best_acc": float(best_acc),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "eval": eval_metrics,
        "cfg": asdict(cfg),
    }, path)


# ============================================================
# Main
# ============================================================
def main() -> None:
    preflight_check()
    set_seed(cfg.SEED)
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    best_path = os.path.join(cfg.SAVE_DIR, cfg.BEST_NAME)
    last_path = os.path.join(cfg.SAVE_DIR, cfg.LAST_NAME)
    log_path = os.path.join(cfg.SAVE_DIR, cfg.LOG_NAME)

    train_box_path = os.path.join(cfg.GDINO_BOX_DIR, cfg.GDINO_TRAIN_BOX_FILE)
    test_box_path = os.path.join(cfg.GDINO_BOX_DIR, cfg.GDINO_TEST_BOX_FILE)
    for path in [train_box_path, test_box_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    ds_train = CUBCachedDINOWithBoxes("train", train_box_path)
    ds_test = CUBCachedDINOWithBoxes("test", test_box_path)
    if (ds_train.D, ds_train.H, ds_train.W) != (ds_test.D, ds_test.H, ds_test.W):
        raise RuntimeError("Train/test feature shape mismatch")

    pin = DEVICE == "cuda"
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

    model = SemanticPartAdditivePrototypeNet(
        dim=ds_train.D,
        parts=len(cfg.PARTS),
        prototypes_per_part=cfg.K_PER_PART,
        classes=cfg.NUM_CLASSES,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW([
        {
            "params": [model.part_queries, model.null_logits],
            "lr": cfg.LR_ROUTER,
            "weight_decay": cfg.WEIGHT_DECAY,
        },
        {
            "params": [model.proto_residual],
            "lr": cfg.LR_PROTO,
            "weight_decay": cfg.WEIGHT_DECAY,
        },
        {
            "params": [model.class_theta, model.class_bias],
            "lr": cfg.LR_CLASSIFIER,
            "weight_decay": 0.0,
        },
    ])
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.AMP and DEVICE == "cuda"))

    start_epoch = 1
    best_acc = -1.0
    resumed = False
    resume_path = pick_resume_path(last_path, best_path) if cfg.AUTO_RESUME else None
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_acc = float(checkpoint.get("best_acc", -1.0))
        resumed = True
        print(f"[Resume] {resume_path} -> epoch {start_epoch}; best={best_acc:.6f}")

    if cfg.BOOTSTRAP_FROM_GDINO and not resumed:
        bootstrap_from_gdino(model, dl_train, ds_train.H, ds_train.W)

    print(f"[Device] {DEVICE}; AMP={cfg.AMP}")
    print(f"[Data] train={len(ds_train)} test={len(ds_test)} feature={ds_train.D} grid={ds_train.H}x{ds_train.W}")
    print(f"[Model] parts={cfg.PARTS}; K={cfg.K_PER_PART}; use_cls={cfg.USE_CLS_IN_PATCH}")
    print("[Prediction] no old-head fusion; logits are exact sums of non-negative prototype contributions")

    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        model.train()
        t0 = time.time()
        lam_route = route_lambda(epoch)

        totals = {
            "loss": 0.0,
            "ce": 0.0,
            "route": 0.0,
            "vis": 0.0,
            "lb": 0.0,
            "div": 0.0,
            "sparse": 0.0,
            "correct": 0,
            "count": 0,
        }

        pbar = tqdm(dl_train, desc=f"Train {epoch}/{cfg.EPOCHS}", ncols=200)
        for fm, boxes, y in pbar:
            fm = fm.to(DEVICE, non_blocking=True).float()
            boxes = boxes.to(DEVICE, non_blocking=True).float()
            y = y.to(DEVICE, non_blocking=True)
            q_sem, valid_bp = boxes_to_soft_q(boxes, ds_train.H, ds_train.W, cfg.MODEL_IM_SIZE, cfg.EPS)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.AMP and DEVICE == "cuda")):
                out = model(fm)
                loss_ce = F.cross_entropy(
                    out["logits"],
                    y,
                    label_smoothing=cfg.LABEL_SMOOTHING,
                )
                loss_route = semantic_route_loss(out["part_map"], q_sem, valid_bp, cfg.EPS)
                loss_vis = visible_part_loss(out["visibility"], valid_bp, cfg.EPS)
                loss_lb = semantic_proto_load_balance_loss(out["proto_assign"], q_sem, valid_bp, cfg.EPS)
                loss_div = within_part_prototype_diversity_loss(out["prototypes"], cfg.PROTO_DIV_MARGIN)
                loss_sparse = classifier_sparsity_loss(out["class_weight"])

                loss = (
                    cfg.LAMBDA_CE * loss_ce
                    + lam_route * loss_route
                    + cfg.LAMBDA_VIS * loss_vis
                    + cfg.LAMBDA_PROTO_LB * loss_lb
                    + cfg.LAMBDA_PROTO_DIV * loss_div
                    + cfg.LAMBDA_CLS_SPARSE * loss_sparse
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            model.ema_update_memory(
                out["Xn"].detach(),
                out["part_map"].detach(),
                out["proto_assign"].detach(),
                q_sem.detach(),
                valid_bp.detach(),
            )

            bs = y.shape[0]
            totals["count"] += bs
            totals["correct"] += (out["logits"].argmax(1) == y).sum().item()
            for key, value in [
                ("loss", loss),
                ("ce", loss_ce),
                ("route", loss_route),
                ("vis", loss_vis),
                ("lb", loss_lb),
                ("div", loss_div),
                ("sparse", loss_sparse),
            ]:
                totals[key] += float(value.detach().item()) * bs

            denom = max(1, totals["count"])
            pbar.set_postfix({
                "acc": f"{totals['correct'] / denom:.4f}",
                "ce": f"{totals['ce'] / denom:.3f}",
                "route": f"{totals['route'] / denom:.3f}",
                "lb": f"{totals['lb'] / denom:.3f}",
            })

        train_count = max(1, totals["count"])
        train_metrics = {
            "acc": totals["correct"] / train_count,
            "loss": totals["loss"] / train_count,
            "ce": totals["ce"] / train_count,
            "route": totals["route"] / train_count,
            "vis": totals["vis"] / train_count,
            "lb": totals["lb"] / train_count,
            "div": totals["div"] / train_count,
            "sparse": totals["sparse"] / train_count,
        }
        eval_metrics = evaluate(model, dl_test, ds_test.H, ds_test.W)

        elapsed = time.time() - t0
        parts = list(cfg.PARTS)
        vis_text = " | ".join(
            f"{parts[i]}:{eval_metrics['visibility_per_part'][i]:.3f}"
            for i in range(len(parts))
        )
        active_text = " | ".join(
            f"{parts[i]}:{int(eval_metrics['active_prototypes_per_part'][i])}/{cfg.K_PER_PART}"
            for i in range(len(parts))
        )
        print(
            f"[Epoch {epoch:03d}] {elapsed:.1f}s "
            f"train_acc={train_metrics['acc']:.4f} test_acc={eval_metrics['acc']:.4f} "
            f"CE={eval_metrics['ce']:.4f} routeCE={eval_metrics['route_ce']:.4f} "
            f"lambda_route={lam_route:.3f}"
        )
        print("[Eval visibility] " + vis_text)
        print("[Eval active prototypes] " + active_text)

        record = {
            "epoch": epoch,
            "seconds": elapsed,
            "lambda_route": lam_route,
            "train": train_metrics,
            "eval": eval_metrics,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if eval_metrics["acc"] > best_acc:
            best_acc = float(eval_metrics["acc"])
            save_checkpoint(best_path, epoch, best_acc, model, optimizer, scaler, eval_metrics)
            print(f"[Best] acc={best_acc:.6f} -> {best_path}")
        save_checkpoint(last_path, epoch, best_acc, model, optimizer, scaler, eval_metrics)

        if cfg.VIZ_ON and (epoch == 1 or epoch % cfg.VIZ_EVERY == 0):
            save_visualizations(epoch, model, ds_test)

    print(f"Done. Best test accuracy={best_acc:.6f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    apply_args(parse_args())
    main()
