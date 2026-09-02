#!/usr/bin/env python3
"""
Evaluate whether the learned CUB part queries / part-prototype banks actually
align with their *named* semantic parts using the official CUB-200-2011
part keypoint annotations.

This script is designed for:
    wacv_proto/cub_server_repo

Place it next to:
    train_cub_semantic_part_additive_proto_server.py

It DOES NOT use GroundingDINO boxes during evaluation.

Metrics
-------
1) Query Named-Part Alignment
   For each image and named part p, take argmax_n part_map[p, n].
   Compare that patch-center location with the official CUB keypoints belonging
   to the named part.

2) Prototype Named-Part Alignment (top-K retrieval)
   For every prototype (p, k), rank test images by the maximum local
   part-gated prototype evidence

       responsibility[p,n,k] * ReLU(sim[p,n,k])

   matching the local evidence used by the model.  For each top-K retrieval,
   take the maximally activated patch and test whether it lands on the named
   CUB part assigned to bank p.

   We report:
     - strict alignment:
         target part must be visible AND peak must hit it.
     - conditional alignment:
         hit rate only among retrievals where target part is visible.
     - target-visible rate:
         fraction of top-K retrievals where the named target part is visible.
     - Named-Part Consistency@tau:
         fraction of prototypes whose strict top-K alignment >= tau.

CUB 15 keypoints -> 6 named banks
--------------------------------
beak : beak
head : crown, forehead, left eye, right eye, nape, throat
wing : left wing, right wing
body : back, belly, breast
tail : tail
feet : left leg, right leg

The mapping is fixed before evaluation and covers all 15 official CUB parts.

Coordinate system
-----------------
The repository's DINO cache uses:
  raw image -> CUB GT bbox crop -> square warp to 224 x 224.

This script reproduces the *exact same* bbox floor/ceil/clamp geometry before
transforming CUB keypoints into the 224 x 224 model coordinate system.

Example
-------
python eval_cub_named_part_alignment.py \
  --cub-root ./data/CUB_200_2011 \
  --dino-cache-dir ./artifacts/dino_vitb14_bbox224 \
  --checkpoint ./runs/semantic_part_additive_proto/best.pth \
  --output-dir ./runs/semantic_part_additive_proto/named_part_eval \
  --batch-size 128 \
  --proto-topk 10 30

Run the same command on the no-GroundingDINO checkpoint and compare:
  accuracy vs query/prototype named-part alignment.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import train_cub_semantic_part_additive_proto_server as train_mod
except ImportError as exc:
    raise ImportError(
        "Could not import train_cub_semantic_part_additive_proto_server.py. "
        "Place this script in cub_server_repo/ next to the training script, "
        "or run it with cub_server_repo on PYTHONPATH."
    ) from exc


# ---------------------------------------------------------------------
# Fixed semantic mapping: official CUB part names -> our six named banks
# ---------------------------------------------------------------------
BANK_TO_CUB_PART_NAMES: Dict[str, Tuple[str, ...]] = {
    "beak": ("beak",),
    "head": ("crown", "forehead", "left eye", "right eye", "nape", "throat"),
    "wing": ("left wing", "right wing"),
    "body": ("back", "belly", "breast"),
    "tail": ("tail",),
    "feet": ("left leg", "right leg"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate CUB named-part alignment using official keypoints."
    )
    p.add_argument("--cub-root", required=True)
    p.add_argument("--dino-cache-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Same local response window used by the paper's CUB interpretability eval.
    p.add_argument(
        "--region-size",
        type=float,
        default=72.0,
        help="Square local response region size in 224-space. Default: 72.",
    )
    p.add_argument(
        "--pck",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.20],
        help="PCK radii as fractions of the 224 warped crop.",
    )

    p.add_argument(
        "--proto-topk",
        nargs="+",
        type=int,
        default=[10, 30],
        help="Top-K image retrieval sizes used for prototype named-part alignment.",
    )
    p.add_argument(
        "--consistency-threshold",
        type=float,
        default=0.80,
        help="Threshold for Named-Part Consistency@tau.",
    )
    p.add_argument(
        "--proto-local-score",
        choices=["evidence", "responsibility"],
        default="evidence",
        help=(
            "Local score used to rank/localize a prototype. "
            "'evidence' = responsibility * ReLU(sim), matching Eq. 3; "
            "'responsibility' = routing responsibility only."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# CUB metadata readers
# ---------------------------------------------------------------------
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


def normalize_part_name(name: str) -> str:
    return " ".join(
        name.strip().lower().replace("_", " ").replace("-", " ").split()
    )


def read_cub_part_names(cub_root: str) -> Dict[int, str]:
    path = os.path.join(cub_root, "parts", "parts.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing official CUB part-name file: {path}"
        )
    out: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pid, name = line.split(maxsplit=1)
            out[int(pid)] = normalize_part_name(name)
    return out


def read_cub_part_locs(
    cub_root: str,
) -> Dict[int, Dict[int, Tuple[float, float, int]]]:
    """
    Returns:
        image_id -> part_id -> (x, y, visible)
    """
    path = os.path.join(cub_root, "parts", "part_locs.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing official CUB part-location file: {path}"
        )
    out: Dict[int, Dict[int, Tuple[float, float, int]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.strip().split()
            if not vals:
                continue
            image_id = int(vals[0])
            part_id = int(vals[1])
            x = float(vals[2])
            y = float(vals[3])
            visible = int(vals[4])
            out.setdefault(image_id, {})[part_id] = (x, y, visible)
    return out


def exact_bbox_crop_xyxy(
    image_path: str,
    bbox_xywh: Tuple[float, float, float, float],
) -> Tuple[int, int, int, int]:
    """
    Reproduce tools/build_dino_cache.py::crop_bbox exactly.
    """
    x, y, w, h = bbox_xywh
    with Image.open(image_path) as image:
        width, height = image.size

    x1 = max(0, min(width - 1, int(np.floor(x))))
    y1 = max(0, min(height - 1, int(np.floor(y))))
    x2 = max(x1 + 1, min(width, int(np.ceil(x + w))))
    y2 = max(y1 + 1, min(height, int(np.ceil(y + h))))
    return x1, y1, x2, y2


# ---------------------------------------------------------------------
# Test dataset: DINO cache + official CUB keypoints, NO GroundingDINO
# ---------------------------------------------------------------------
class CUBNamedPartEvalDataset(Dataset):
    def __init__(
        self,
        cub_root: str,
        dino_cache_dir: str,
        parts: Tuple[str, ...],
        model_im_size: int,
        use_cls_in_patch: bool,
        cls_alpha: float,
    ):
        self.cub_root = os.path.abspath(os.path.expanduser(cub_root))
        self.dino_cache_dir = os.path.abspath(os.path.expanduser(dino_cache_dir))
        self.parts = tuple(str(x).lower() for x in parts)
        self.model_im_size = int(model_im_size)
        self.use_cls_in_patch = bool(use_cls_in_patch)
        self.cls_alpha = float(cls_alpha)

        for name in self.parts:
            if name not in BANK_TO_CUB_PART_NAMES:
                raise ValueError(
                    f"Unknown named bank '{name}'. "
                    f"Expected subset of {tuple(BANK_TO_CUB_PART_NAMES)}"
                )

        # Reuse the repository's exact DINO memmap reader.
        train_mod.cfg.DINO_CACHE_DIR = self.dino_cache_dir
        self.cls_mm, self.patch_mm, self.labels, self.meta = (
            train_mod.load_dino_memmaps(self.dino_cache_dir, "test")
        )
        self.N = int(self.meta["N"])
        self.D = int(self.meta["D"])
        self.PATCHES = int(self.meta["P"])

        grid = int(round(math.sqrt(self.PATCHES)))
        if grid * grid != self.PATCHES:
            raise ValueError(
                f"Patch count must form a square grid, got P={self.PATCHES}"
            )
        self.H = grid
        self.W = grid

        # Exact test image order used by build_dino_cache.py:
        # sorted image_id filtered by train_test_split.
        id2path = read_kv_text(os.path.join(self.cub_root, "images.txt"))
        id2split = read_kv_int(os.path.join(self.cub_root, "train_test_split.txt"))
        self.image_ids = [
            image_id
            for image_id in sorted(id2path)
            if id2split[image_id] == 0
        ]
        self.relpaths = [id2path[i] for i in self.image_ids]

        if len(self.image_ids) != self.N:
            raise RuntimeError(
                f"CUB/DINO test order mismatch: CUB={len(self.image_ids)}, cache={self.N}"
            )

        # If the cache saved relpaths, verify them too.
        relpath_json = os.path.join(
            self.dino_cache_dir, train_mod.cfg.DINO_FEATURE_SUBDIR,
            "test_relpaths.json"
        )
        if os.path.isfile(relpath_json):
            with open(relpath_json, "r", encoding="utf-8") as f:
                cache_relpaths = [str(x) for x in json.load(f)]
            if cache_relpaths != self.relpaths:
                raise RuntimeError(
                    "test_relpaths.json does not match official CUB test order."
                )

        bboxes = read_bboxes(os.path.join(self.cub_root, "bounding_boxes.txt"))
        part_names = read_cub_part_names(self.cub_root)
        part_locs = read_cub_part_locs(self.cub_root)

        # Resolve CUB part IDs dynamically from parts.txt rather than hard-coding IDs.
        name_to_pid = {name: pid for pid, name in part_names.items()}
        self.bank_part_ids: Dict[str, List[int]] = {}
        for bank in self.parts:
            wanted = BANK_TO_CUB_PART_NAMES[bank]
            missing = [x for x in wanted if x not in name_to_pid]
            if missing:
                raise RuntimeError(
                    f"CUB parts.txt is missing names needed by bank '{bank}': {missing}. "
                    f"Available names: {sorted(name_to_pid)}"
                )
            self.bank_part_ids[bank] = [name_to_pid[x] for x in wanted]

        # group_points[i][p] is an array [num_visible_keypoints, 2] in warped 224-space.
        self.group_points: List[List[np.ndarray]] = []

        visible_counts = np.zeros(len(self.parts), dtype=np.int64)
        outside_crop_counts = np.zeros(len(self.parts), dtype=np.int64)

        print("[GT] Transforming official CUB keypoints into model crop coordinates...")
        for image_id, relpath in tqdm(
            zip(self.image_ids, self.relpaths),
            total=self.N,
            desc="CUB keypoints",
            ncols=120,
        ):
            image_path = os.path.join(self.cub_root, "images", relpath)
            x1, y1, x2, y2 = exact_bbox_crop_xyxy(
                image_path, bboxes[image_id]
            )
            crop_w = float(x2 - x1)
            crop_h = float(y2 - y1)

            sample_groups: List[np.ndarray] = []
            image_locs = part_locs.get(image_id, {})

            for p, bank in enumerate(self.parts):
                pts: List[Tuple[float, float]] = []
                for pid in self.bank_part_ids[bank]:
                    if pid not in image_locs:
                        continue
                    x, y, visible = image_locs[pid]
                    if not visible:
                        continue

                    # Same crop origin + square warp as DINO cache.
                    xw = (x - x1) * self.model_im_size / crop_w
                    yw = (y - y1) * self.model_im_size / crop_h

                    # If an official visible keypoint lies outside the actual cropped
                    # image seen by the model, do not evaluate the model on that point.
                    if 0.0 <= xw < self.model_im_size and 0.0 <= yw < self.model_im_size:
                        pts.append((float(xw), float(yw)))
                    else:
                        outside_crop_counts[p] += 1

                arr = (
                    np.asarray(pts, dtype=np.float32).reshape(-1, 2)
                    if pts
                    else np.zeros((0, 2), dtype=np.float32)
                )
                if len(arr) > 0:
                    visible_counts[p] += 1
                sample_groups.append(arr)

            self.group_points.append(sample_groups)

        print("[GT] Evaluable test images by named part:")
        for p, bank in enumerate(self.parts):
            print(
                f"  {bank:>5s}: {visible_counts[p]}/{self.N} images "
                f"(visible GT group); outside-crop keypoints={outside_crop_counts[p]}"
            )

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        return int(idx), int(self.labels[idx])

    def collate(self, batch):
        idxs = np.asarray([x[0] for x in batch], dtype=np.int64)
        y = torch.tensor([x[1] for x in batch], dtype=torch.long)

        patch = self.patch_mm[idxs].astype(np.float32, copy=False)
        if self.use_cls_in_patch:
            cls = self.cls_mm[idxs].astype(np.float32, copy=False)
            patch = patch + self.cls_alpha * cls[:, None, :]

        b = patch.shape[0]
        fm = (
            patch.reshape(b, self.H, self.W, self.D)
            .transpose(0, 3, 1, 2)
            .copy()
        )
        return torch.from_numpy(fm), y, torch.from_numpy(idxs)


# ---------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------
def patch_indices_to_xy(
    idx: np.ndarray,
    h: int,
    w: int,
    image_size: int,
) -> np.ndarray:
    """
    idx: arbitrary shape of flattened patch indices.
    returns: idx.shape + (2,), with (x, y) patch centers in model pixels.
    """
    idx = np.asarray(idx)
    row = idx // w
    col = idx % w
    x = (col.astype(np.float32) + 0.5) * (float(image_size) / w)
    y = (row.astype(np.float32) + 0.5) * (float(image_size) / h)
    return np.stack([x, y], axis=-1)


def point_hit_square(
    peak_xy: np.ndarray,
    gt_points: np.ndarray,
    region_size: float,
) -> bool:
    if gt_points.shape[0] == 0:
        return False
    half = float(region_size) / 2.0
    delta = np.abs(gt_points - peak_xy[None, :])
    # A 72x72 local region centered at the activation peak.
    return bool(np.any((delta[:, 0] <= half) & (delta[:, 1] <= half)))


def min_distance(
    peak_xy: np.ndarray,
    gt_points: np.ndarray,
) -> float:
    if gt_points.shape[0] == 0:
        return float("nan")
    d = np.linalg.norm(gt_points - peak_xy[None, :], axis=1)
    return float(d.min())


def summarize_query_alignment(
    query_peak_idx: np.ndarray,  # [N,P]
    ds: CUBNamedPartEvalDataset,
    region_size: float,
    pck_fracs: List[float],
) -> Dict[str, Any]:
    peak_xy = patch_indices_to_xy(
        query_peak_idx, ds.H, ds.W, ds.model_im_size
    )  # [N,P,2]

    per_part = {}
    total_valid = 0
    total_hits = 0
    total_dist = 0.0
    pck_total_hits = {float(t): 0 for t in pck_fracs}

    for p, bank in enumerate(ds.parts):
        valid = 0
        hits = 0
        dist_sum = 0.0
        pck_hits = {float(t): 0 for t in pck_fracs}

        for i in range(ds.N):
            gt = ds.group_points[i][p]
            if len(gt) == 0:
                continue
            valid += 1
            d = min_distance(peak_xy[i, p], gt)
            dist_sum += d
            if point_hit_square(peak_xy[i, p], gt, region_size):
                hits += 1
            for t in pck_fracs:
                radius = float(t) * ds.model_im_size
                if d <= radius:
                    pck_hits[float(t)] += 1

        part_summary = {
            "valid_images": int(valid),
            "region_hit_rate": float(hits / max(1, valid)),
            "mean_distance_px": float(dist_sum / max(1, valid)),
            "mean_normalized_distance": float(
                (dist_sum / max(1, valid)) / ds.model_im_size
            ),
            "pck": {
                str(t): float(pck_hits[float(t)] / max(1, valid))
                for t in pck_fracs
            },
        }
        per_part[bank] = part_summary

        total_valid += valid
        total_hits += hits
        total_dist += dist_sum
        for t in pck_fracs:
            pck_total_hits[float(t)] += pck_hits[float(t)]

    return {
        "overall_micro": {
            "valid_image_parts": int(total_valid),
            "region_hit_rate": float(total_hits / max(1, total_valid)),
            "mean_distance_px": float(total_dist / max(1, total_valid)),
            "mean_normalized_distance": float(
                (total_dist / max(1, total_valid)) / ds.model_im_size
            ),
            "pck": {
                str(t): float(pck_total_hits[float(t)] / max(1, total_valid))
                for t in pck_fracs
            },
        },
        "per_part": per_part,
    }


def summarize_prototype_alignment(
    proto_scores: np.ndarray,    # [N,P,K]
    proto_peak_idx: np.ndarray,  # [N,P,K]
    ds: CUBNamedPartEvalDataset,
    topk_values: List[int],
    region_size: float,
    consistency_threshold: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    peak_xy = patch_indices_to_xy(
        proto_peak_idx, ds.H, ds.W, ds.model_im_size
    )  # [N,P,K,2]

    _, P, K = proto_scores.shape
    all_summary: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []

    for requested_topk in topk_values:
        topk = min(int(requested_topk), ds.N)

        strict_by_proto = np.zeros((P, K), dtype=np.float64)
        cond_by_proto = np.full((P, K), np.nan, dtype=np.float64)
        visible_by_proto = np.zeros((P, K), dtype=np.float64)

        for p, bank in enumerate(ds.parts):
            for k in range(K):
                scores = proto_scores[:, p, k]
                # Stable descending sort so equal scores are deterministic.
                order = np.argsort(-scores, kind="stable")[:topk]

                hit_count = 0
                visible_count = 0

                for i in order.tolist():
                    gt = ds.group_points[i][p]
                    visible = len(gt) > 0
                    if visible:
                        visible_count += 1
                        if point_hit_square(
                            peak_xy[i, p, k], gt, region_size
                        ):
                            hit_count += 1

                strict = hit_count / max(1, topk)
                conditional = (
                    hit_count / visible_count
                    if visible_count > 0
                    else float("nan")
                )
                visible_rate = visible_count / max(1, topk)

                strict_by_proto[p, k] = strict
                cond_by_proto[p, k] = conditional
                visible_by_proto[p, k] = visible_rate

                csv_rows.append({
                    "part": bank,
                    "prototype": k,
                    "topk": topk,
                    "strict_named_part_alignment": strict,
                    "conditional_named_part_alignment": conditional,
                    "target_visible_rate": visible_rate,
                    "named_part_consistent": int(strict >= consistency_threshold),
                    "mean_topk_score": float(scores[order].mean()),
                    "min_topk_score": float(scores[order].min()),
                    "max_topk_score": float(scores[order].max()),
                })

        per_part = {}
        for p, bank in enumerate(ds.parts):
            finite_cond = np.isfinite(cond_by_proto[p])
            per_part[bank] = {
                "num_prototypes": int(K),
                "strict_named_part_alignment_mean": float(
                    strict_by_proto[p].mean()
                ),
                "strict_named_part_alignment_std": float(
                    strict_by_proto[p].std(ddof=0)
                ),
                "conditional_named_part_alignment_mean": float(
                    np.nanmean(cond_by_proto[p])
                    if finite_cond.any() else float("nan")
                ),
                "target_visible_rate_mean": float(
                    visible_by_proto[p].mean()
                ),
                f"named_part_consistency@{consistency_threshold:g}": float(
                    (strict_by_proto[p] >= consistency_threshold).mean()
                ),
            }

        finite_cond_all = np.isfinite(cond_by_proto)
        all_summary[str(topk)] = {
            "topk": int(topk),
            "overall_macro_over_prototypes": {
                "strict_named_part_alignment_mean": float(
                    strict_by_proto.mean()
                ),
                "strict_named_part_alignment_std": float(
                    strict_by_proto.std(ddof=0)
                ),
                "conditional_named_part_alignment_mean": float(
                    np.nanmean(cond_by_proto)
                    if finite_cond_all.any() else float("nan")
                ),
                "target_visible_rate_mean": float(
                    visible_by_proto.mean()
                ),
                f"named_part_consistency@{consistency_threshold:g}": float(
                    (strict_by_proto >= consistency_threshold).mean()
                ),
            },
            "per_part": per_part,
        }

    return all_summary, csv_rows


# ---------------------------------------------------------------------
# Checkpoint/model loading
# ---------------------------------------------------------------------
def apply_checkpoint_cfg(
    checkpoint: Dict[str, Any],
    cub_root: str,
    dino_cache_dir: str,
) -> Dict[str, Any]:
    ckpt_cfg = checkpoint.get("cfg", {})
    if isinstance(ckpt_cfg, dict):
        for key, value in ckpt_cfg.items():
            if hasattr(train_mod.cfg, key):
                setattr(train_mod.cfg, key, value)

    # Evaluation paths always come from this command line.
    train_mod.cfg.CUB_ROOT = os.path.abspath(os.path.expanduser(cub_root))
    train_mod.cfg.DINO_CACHE_DIR = os.path.abspath(
        os.path.expanduser(dino_cache_dir)
    )

    # torch.save/asdict can preserve tuple, but normalize defensively.
    train_mod.cfg.PARTS = tuple(str(x).lower() for x in train_mod.cfg.PARTS)

    return ckpt_cfg if isinstance(ckpt_cfg, dict) else {}


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    device: str,
) -> train_mod.SemanticPartAdditivePrototypeNet:
    if "model" not in checkpoint:
        raise KeyError("Checkpoint does not contain key 'model'.")

    state = checkpoint["model"]

    # Infer dimensions from the actual checkpoint, not from today's defaults.
    q = state["part_queries"]
    residual = state["proto_residual"]
    theta = state["class_theta"]

    parts = int(q.shape[0])
    dim = int(q.shape[1])
    k = int(residual.shape[1])
    classes = int(theta.shape[0])

    if len(train_mod.cfg.PARTS) != parts:
        raise RuntimeError(
            f"Checkpoint has {parts} part banks but cfg.PARTS="
            f"{train_mod.cfg.PARTS} ({len(train_mod.cfg.PARTS)})."
        )

    # Important: forward() reads cfg.K/etc-related temperatures/scales globally,
    # while shape is determined from the checkpoint itself.
    train_mod.cfg.K_PER_PART = k
    train_mod.cfg.NUM_CLASSES = classes

    model = train_mod.SemanticPartAdditivePrototypeNet(
        dim=dim,
        parts=parts,
        prototypes_per_part=k,
        classes=classes,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------
@torch.inference_mode()
def run(args: argparse.Namespace) -> Dict[str, Any]:
    cub_root = os.path.abspath(os.path.expanduser(args.cub_root))
    dino_cache_dir = os.path.abspath(os.path.expanduser(args.dino_cache_dir))
    checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_cfg = apply_checkpoint_cfg(
        checkpoint,
        cub_root=cub_root,
        dino_cache_dir=dino_cache_dir,
    )

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"--device={device} requested but CUDA is not available."
        )

    model = build_model_from_checkpoint(checkpoint, device)

    ds = CUBNamedPartEvalDataset(
        cub_root=cub_root,
        dino_cache_dir=dino_cache_dir,
        parts=tuple(train_mod.cfg.PARTS),
        model_im_size=int(train_mod.cfg.MODEL_IM_SIZE),
        use_cls_in_patch=bool(train_mod.cfg.USE_CLS_IN_PATCH),
        cls_alpha=float(train_mod.cfg.CLS_ALPHA),
    )

    if model.dim != ds.D:
        raise RuntimeError(
            f"Checkpoint D={model.dim}, but test cache D={ds.D}."
        )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
        collate_fn=ds.collate,
    )

    N = ds.N
    P = model.parts
    K = model.k

    query_peak_idx = np.zeros((N, P), dtype=np.int16)
    proto_peak_idx = np.zeros((N, P, K), dtype=np.int16)
    proto_scores = np.zeros((N, P, K), dtype=np.float32)

    total = 0
    correct = 0

    for fm, y, idxs in tqdm(loader, desc="Named-part eval", ncols=140):
        fm = fm.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        out = model(fm)

        bs = y.shape[0]
        total += bs
        correct += (out["logits"].argmax(dim=1) == y).sum().item()

        qpeak = out["part_map"].argmax(dim=-1)  # B,P

        if args.proto_local_score == "evidence":
            local = out["responsibility"] * F.relu(out["sim"])
        else:
            local = out["responsibility"]

        # B,P,K; B,P,K
        pscores, ppeak = local.max(dim=2)

        idx_np = idxs.numpy()
        query_peak_idx[idx_np] = qpeak.cpu().numpy().astype(np.int16)
        proto_peak_idx[idx_np] = ppeak.cpu().numpy().astype(np.int16)
        proto_scores[idx_np] = pscores.cpu().numpy().astype(np.float32)

    test_acc = correct / max(1, total)

    query_summary = summarize_query_alignment(
        query_peak_idx=query_peak_idx,
        ds=ds,
        region_size=args.region_size,
        pck_fracs=list(args.pck),
    )

    proto_summary, proto_rows = summarize_prototype_alignment(
        proto_scores=proto_scores,
        proto_peak_idx=proto_peak_idx,
        ds=ds,
        topk_values=list(args.proto_topk),
        region_size=args.region_size,
        consistency_threshold=args.consistency_threshold,
    )

    summary: Dict[str, Any] = {
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_acc": (
            float(checkpoint["best_acc"])
            if "best_acc" in checkpoint
            else None
        ),
        "recomputed_test_acc": float(test_acc),
        "device": device,
        "num_test_images": int(ds.N),
        "feature_dim": int(ds.D),
        "patch_grid": [int(ds.H), int(ds.W)],
        "model_image_size": int(ds.model_im_size),
        "parts": list(ds.parts),
        "k_per_part": int(K),
        "use_cls_in_patch": bool(train_mod.cfg.USE_CLS_IN_PATCH),
        "proto_local_score": args.proto_local_score,
        "region_size": float(args.region_size),
        "pck": [float(x) for x in args.pck],
        "proto_topk": [int(x) for x in args.proto_topk],
        "consistency_threshold": float(args.consistency_threshold),
        "mapping": {
            k: list(v) for k, v in BANK_TO_CUB_PART_NAMES.items()
        },
        "checkpoint_cfg": ckpt_cfg,
        "query_named_part_alignment": query_summary,
        "prototype_named_part_alignment": proto_summary,
    }

    json_path = os.path.join(output_dir, "named_part_alignment_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, "named_part_alignment_per_prototype.csv")
    fieldnames = [
        "part",
        "prototype",
        "topk",
        "strict_named_part_alignment",
        "conditional_named_part_alignment",
        "target_visible_rate",
        "named_part_consistent",
        "mean_topk_score",
        "min_topk_score",
        "max_topk_score",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(proto_rows)

    # Compact arrays are useful for later threshold/sensitivity analysis
    # without rerunning the model.
    npz_path = os.path.join(output_dir, "named_part_alignment_raw.npz")
    np.savez_compressed(
        npz_path,
        query_peak_idx=query_peak_idx,
        proto_peak_idx=proto_peak_idx,
        proto_scores=proto_scores,
    )

    print("\n" + "=" * 88)
    print("CUB NAMED-PART ALIGNMENT")
    print("=" * 88)
    print(f"checkpoint : {checkpoint_path}")
    print(f"epoch      : {summary['checkpoint_epoch']}")
    print(f"test acc   : {test_acc:.4f}")
    print(
        f"query hit  : "
        f"{query_summary['overall_micro']['region_hit_rate']:.4f} "
        f"(region={args.region_size:g}x{args.region_size:g})"
    )
    for t in args.pck:
        print(
            f"query PCK@{t:g}: "
            f"{query_summary['overall_micro']['pck'][str(t)]:.4f}"
        )

    print("\nQuery alignment by named part:")
    for bank in ds.parts:
        x = query_summary["per_part"][bank]
        print(
            f"  {bank:>5s}: hit={x['region_hit_rate']:.4f} "
            f"mean_dist={x['mean_distance_px']:.2f}px "
            f"valid={x['valid_images']}"
        )

    for requested_topk in args.proto_topk:
        key = str(min(int(requested_topk), ds.N))
        x = proto_summary[key]["overall_macro_over_prototypes"]
        cons_key = f"named_part_consistency@{args.consistency_threshold:g}"
        print(f"\nPrototype top-{key}:")
        print(
            f"  strict alignment      = "
            f"{x['strict_named_part_alignment_mean']:.4f}"
        )
        print(
            f"  conditional alignment = "
            f"{x['conditional_named_part_alignment_mean']:.4f}"
        )
        print(
            f"  target-visible rate   = "
            f"{x['target_visible_rate_mean']:.4f}"
        )
        print(
            f"  {cons_key:<22s} = {x[cons_key]:.4f}"
        )

    print(f"\nSaved summary : {json_path}")
    print(f"Saved per-proto: {csv_path}")
    print(f"Saved raw     : {npz_path}")
    print("=" * 88)

    return summary


def main() -> None:
    args = parse_args()

    # Basic argument checks.
    args.proto_topk = sorted(set(int(x) for x in args.proto_topk if int(x) > 0))
    if not args.proto_topk:
        raise ValueError("--proto-topk must contain at least one positive integer.")
    if args.region_size <= 0:
        raise ValueError("--region-size must be > 0.")
    if not (0.0 <= args.consistency_threshold <= 1.0):
        raise ValueError("--consistency-threshold must be in [0,1].")
    for t in args.pck:
        if t <= 0:
            raise ValueError("--pck values must be > 0.")

    run(args)


if __name__ == "__main__":
    main()
