#!/usr/bin/env python3
import os, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import train_cub_shared_part_proto_finetune as T


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ProtoScoreMLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def apply_ckpt_cfg(ckpt, args):
    c = ckpt.get("cfg", {})
    for k, v in c.items():
        if hasattr(T.cfg, k):
            setattr(T.cfg, k, v)

    T.cfg.cub_root = os.path.abspath(args.cub_root)
    T.cfg.gdino_box_dir = os.path.abspath(args.gdino_box_dir)
    T.cfg.gdino_train_file = args.gdino_train_file
    T.cfg.gdino_test_file = args.gdino_test_file
    T.cfg.batch_size = args.extract_batch_size
    T.cfg.num_workers = args.num_workers
    T.cfg.score_mode = args.score_mode
    T.cfg.score_scale = args.score_scale

    # extraction 阶段不需要训练；readout_mode 不影响 proto_score
    if hasattr(T.cfg, "readout_mode"):
        T.cfg.readout_mode = "nonneg"


@torch.no_grad()
def extract_proto_scores(model, loader, device, split_name):
    model.eval()
    xs, ys, rels = [], [], []
    for images, boxes, y, relpath in tqdm(loader, desc=f"Extract {split_name}", ncols=120):
        images = images.to(device, non_blocking=True).float()
        out = model(images)
        feat = out["proto_score"].flatten(1).detach().cpu()
        xs.append(feat)
        ys.append(y.cpu())
        rels.extend(list(relpath))
    return {
        "x": torch.cat(xs, dim=0),
        "y": torch.cat(ys, dim=0),
        "relpaths": rels,
    }


def eval_mlp(model, x, y, batch_size, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i+batch_size].to(device)
            yb = y[i:i+batch_size].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            correct += int((logits.argmax(1) == yb).sum().item())
            total += int(yb.numel())
            loss_sum += float(loss.item()) * int(yb.numel())
    return correct / max(1, total), loss_sum / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--cub-root", default="./data/CUB_200_2011")
    ap.add_argument("--gdino-box-dir", default="./artifacts/gdino_part_boxes_gtbbox_warp224_sep")
    ap.add_argument("--gdino-train-file", default="train_part_boxes_gtbbox_warp518.pt")
    ap.add_argument("--gdino-test-file", default="test_part_boxes_gtbbox_warp518.pt")

    ap.add_argument("--score-mode", default="resp_sum", choices=["resp_sum", "scan_max", "scan_topk"])
    ap.add_argument("--score-scale", type=float, default=8.0)

    ap.add_argument("--extract-batch-size", type=int, default=64)
    ap.add_argument("--train-batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--force-extract", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_cache = os.path.join(args.out_dir, "train_proto_scores.pt")
    test_cache = os.path.join(args.out_dir, "test_proto_scores.pt")

    if args.force_extract or (not os.path.exists(train_cache)) or (not os.path.exists(test_cache)):
        ckpt = torch.load(args.ckpt, map_location="cpu")
        apply_ckpt_cfg(ckpt, args)

        train_gdino = os.path.join(T.cfg.gdino_box_dir, T.cfg.gdino_train_file)
        test_gdino = os.path.join(T.cfg.gdino_box_dir, T.cfg.gdino_test_file)

        ds_train = T.CUBImageWithPartBoxes("train", train_gdino)
        ds_test = T.CUBImageWithPartBoxes("test", test_gdino)

        dl_train = DataLoader(
            ds_train,
            batch_size=args.extract_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device == "cuda"),
            drop_last=False,
        )
        dl_test = DataLoader(
            ds_test,
            batch_size=args.extract_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device == "cuda"),
            drop_last=False,
        )

        print(f"[DINO] loading {T.cfg.dino_model}")
        backbone = T.load_dinov2(T.cfg.dino_model).to(device)
        T.set_dino_trainability(backbone, 0, False)

        with torch.no_grad():
            x0, _, _, _ = next(iter(DataLoader(ds_train, batch_size=1, shuffle=False, num_workers=0)))
            x0 = x0.to(device)
            feats = backbone.forward_features(x0)
            dim = int(feats["x_norm_patchtokens"].shape[-1])

        model = T.SharedPartPrototypeDINO(
            backbone,
            dim=dim,
            parts=len(T.cfg.parts),
            k=T.cfg.k_per_part,
            classes=T.cfg.num_classes,
        ).to(device)

        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print("[Load ckpt]", args.ckpt)
        print("[Missing keys]", missing)
        print("[Unexpected keys]", unexpected)

        for p in model.parameters():
            p.requires_grad = False

        train_pack = extract_proto_scores(model, dl_train, device, "train")
        test_pack = extract_proto_scores(model, dl_test, device, "test")

        torch.save(train_pack, train_cache)
        torch.save(test_pack, test_cache)
        print("[Saved]", train_cache)
        print("[Saved]", test_cache)
    else:
        print("[Use cached features]")
        train_pack = torch.load(train_cache, map_location="cpu")
        test_pack = torch.load(test_cache, map_location="cpu")

    x_train = train_pack["x"].float()
    y_train = train_pack["y"].long()
    x_test = test_pack["x"].float()
    y_test = test_pack["y"].long()

    if not args.no_standardize:
        mean = x_train.mean(dim=0, keepdim=True)
        std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        torch.save({"mean": mean, "std": std}, os.path.join(args.out_dir, "standardize.pt"))

    print(f"[Feature] train={tuple(x_train.shape)} test={tuple(x_test.shape)}")

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, drop_last=False)

    mlp = ProtoScoreMLP(
        in_dim=x_train.shape[1],
        hidden=args.hidden,
        num_classes=200,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = -1.0
    hist_path = os.path.join(args.out_dir, "mlp_history.jsonl")

    for epoch in range(1, args.epochs + 1):
        mlp.train()
        loss_sum, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = mlp(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=args.label_smoothing)
            loss.backward()
            opt.step()

            loss_sum += float(loss.item()) * int(yb.numel())
            correct += int((logits.argmax(1) == yb).sum().item())
            total += int(yb.numel())

        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)
        test_acc, test_loss = eval_mlp(mlp, x_test, y_test, args.train_batch_size, device)

        rec = {
            "epoch": epoch,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "best_acc": max(best_acc, test_acc),
        }
        with open(hist_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch,
                "best_acc": best_acc,
                "model": mlp.state_dict(),
                "args": vars(args),
            }, os.path.join(args.out_dir, "best_mlp.pth"))

        if epoch == 1 or epoch % 10 == 0 or test_acc >= best_acc:
            print(
                f"[Epoch {epoch:03d}] "
                f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
                f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
                f"best={best_acc:.4f}"
            )

    print(f"[Done] best_acc={best_acc:.6f}")
    print(f"[Out] {args.out_dir}")


if __name__ == "__main__":
    main()
