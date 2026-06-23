#!/usr/bin/env python3
# Patch train_cub_shared_part_proto_finetune_reg_vitb16.py in place so it can
# initialize a CLIP ViT-B/16 visual tower from a previously finetuned CUB checkpoint.

from __future__ import annotations

import argparse
import os
import py_compile
import shutil
import tempfile
from pathlib import Path


VISUAL_SECTION = r'''
# -----------------------------
# Visual backbone
# -----------------------------
DINO_V1_HUB_MODELS = {
    "dino_vits16",
    "dino_vits8",
    "dino_vitb16",
    "dino_vitb8",
}


def is_clip_backbone(model_name: str) -> bool:
    return str(model_name).lower() in {
        "clip_vitb16",
        "clip-vit-base-patch16",
        "openai/clip-vit-base-patch16",
    }


def input_normalization() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if is_clip_backbone(cfg.dino_model):
        return CLIP_MEAN, CLIP_STD
    return IMAGENET_MEAN, IMAGENET_STD


def _torch_load_cpu(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_clip_encoder_init(backbone: nn.Module, checkpoint_path: str) -> None:
    if not checkpoint_path:
        print("[CLIP init] no finetuned checkpoint supplied; using OpenAI CLIP initialization.")
        return

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"CLIP init checkpoint not found: {checkpoint_path}")

    payload = _torch_load_cpu(checkpoint_path)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(
            "CLIP init checkpoint must be a state_dict or a checkpoint containing a 'model' state_dict."
        )

    encoder_state: Dict[str, Any] = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("encoder."):
            encoder_state[key[len("encoder."):]] = value

    if not encoder_state:
        examples = list(state.keys())[:8]
        raise KeyError(
            "No 'encoder.*' weights found in the supplied checkpoint. "
            f"Expected a checkpoint from the CLIP finetune script; first keys: {examples}"
        )

    missing, unexpected = backbone.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "CLIP encoder checkpoint is not structurally compatible. "
            f"missing={missing[:12]} unexpected={unexpected[:12]}"
        )

    source_epoch = payload.get("epoch", "?") if isinstance(payload, dict) else "?"
    source_best = payload.get("best_acc", "?") if isinstance(payload, dict) else "?"
    print(
        f"[CLIP init] restored {len(encoder_state)} encoder tensors from {checkpoint_path}; "
        f"source_epoch={source_epoch}; source_best_acc={source_best}; "
        "ignored source head.*"
    )


def load_visual_backbone(model_name: str) -> nn.Module:
    if is_clip_backbone(model_name):
        try:
            from transformers import CLIPVisionModel
        except ImportError as exc:
            raise ImportError(
                "CLIP support requires transformers. Install it in the training environment first."
            ) from exc

        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        model._proto_is_clip = True
        model._proto_is_dino_v1 = False
        load_clip_encoder_init(model, cfg.clip_init_checkpoint)
        return model

    if model_name in DINO_V1_HUB_MODELS:
        model = torch.hub.load("facebookresearch/dino:main", model_name)
        model._proto_is_dino_v1 = True
        model._proto_is_clip = False
        return model

    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model._proto_is_dino_v1 = False
    model._proto_is_clip = False
    return model


def extract_patch_tokens(backbone: nn.Module, images: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    if bool(getattr(backbone, "_proto_is_clip", False)):
        out = backbone(pixel_values=images, return_dict=True)
        x = out.last_hidden_state[:, 1:, :]
    elif bool(getattr(backbone, "_proto_is_dino_v1", False)):
        if not hasattr(backbone, "get_intermediate_layers"):
            raise RuntimeError("Original DINO backbone lacks get_intermediate_layers.")
        layers = backbone.get_intermediate_layers(images, n=1)
        if not isinstance(layers, (list, tuple)) or not layers:
            raise RuntimeError("Could not extract tokens from original DINO.")
        x = layers[-1]
        if isinstance(x, (list, tuple)):
            x = x[0]
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise RuntimeError("Unexpected original-DINO token output.")

        patch_size = getattr(getattr(backbone, "patch_embed", None), "patch_size", 16)
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        expected_n = (images.shape[-2] // int(patch_size)) * (images.shape[-1] // int(patch_size))
        if x.shape[1] == expected_n + 1:
            x = x[:, 1:]
    else:
        feats = backbone.forward_features(images)
        if isinstance(feats, dict):
            x = feats.get("x_norm_patchtokens", None)
            if x is None:
                x = feats.get("x_prenorm", None)
                if x is not None and x.ndim == 3:
                    x = x[:, 1:]
        elif isinstance(feats, torch.Tensor) and feats.ndim == 3:
            x = feats
        else:
            x = None

        if x is None or x.ndim != 3:
            raise RuntimeError("Could not extract patch tokens from DINOv2 output.")

    n = x.shape[1]
    g = int(round(math.sqrt(n)))
    if g * g != n:
        g_after_cls = int(round(math.sqrt(max(0, n - 1))))
        if g_after_cls * g_after_cls == n - 1:
            x = x[:, 1:]
            n = x.shape[1]
            g = int(round(math.sqrt(n)))

    if x.ndim != 3 or g * g != x.shape[1]:
        raise RuntimeError(f"Patch tokens are not a square grid: shape={tuple(x.shape)}")
    return x.float(), g, g


def set_dino_trainability(backbone: nn.Module, last_blocks: int, unfreeze_norm: bool) -> int:
    for p in backbone.parameters():
        p.requires_grad = False

    if bool(getattr(backbone, "_proto_is_clip", False)):
        blocks = list(backbone.vision_model.encoder.layers)
        if last_blocks > 0:
            for blk in blocks[-int(last_blocks):]:
                for p in blk.parameters():
                    p.requires_grad = True
        if unfreeze_norm:
            # Patch tokens come from last_hidden_state; post_layernorm is only used for pooled CLS.
            for p in backbone.vision_model.pre_layrnorm.parameters():
                p.requires_grad = True
    else:
        if last_blocks > 0 and hasattr(backbone, "blocks"):
            blocks = list(backbone.blocks)
            for blk in blocks[-int(last_blocks):]:
                for p in blk.parameters():
                    p.requires_grad = True
        if unfreeze_norm and hasattr(backbone, "norm"):
            for p in backbone.norm.parameters():
                p.requires_grad = True

    return sum(p.numel() for p in backbone.parameters() if p.requires_grad)
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Could not uniquely patch {label}: found {count} occurrences.")
    return text.replace(old, new, 1)


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    i = text.find(start)
    j = text.find(end, i)
    if i < 0 or j < 0 or j <= i:
        raise RuntimeError("Could not locate the visual-backbone section.")
    return text[:i] + replacement + "\n\n" + text[j:]


def patch_source(text: str) -> str:
    if "def load_clip_encoder_init(" in text:
        raise RuntimeError("The target already appears to contain this CLIP-transfer patch.")

    text = replace_once(
        text,
        'IMAGENET_MEAN = (0.485, 0.456, 0.406)\nIMAGENET_STD = (0.229, 0.224, 0.225)\nDEFAULT_PARTS = ("beak", "head", "wing", "body", "tail", "feet")',
        'IMAGENET_MEAN = (0.485, 0.456, 0.406)\nIMAGENET_STD = (0.229, 0.224, 0.225)\nCLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)\nCLIP_STD = (0.26862954, 0.26130258, 0.27577711)\nDEFAULT_PARTS = ("beak", "head", "wing", "body", "tail", "feet")',
        "normalization constants",
    )

    text = replace_once(
        text,
        '    # Torch hub backbone name. Examples: dinov2_vitb14 or dino_vitb16.\n    dino_model: str = "dinov2_vitb14"\n    image_size: int = 224',
        '    # Backbone name: dinov2_vitb14, dino_vitb16, or clip_vitb16.\n    dino_model: str = "dinov2_vitb14"\n    # A checkpoint from train_cub_clip_vitb16_finetune_offline_aug_all_ddp.py.\n    # Only encoder.* is transferred; its incompatible 200-way linear head is ignored.\n    clip_init_checkpoint: str = ""\n    image_size: int = 224',
        "config",
    )

    text = replace_once(
        text,
        '        self.boxes = load_aligned_part_boxes(self.samples, gdino_path, cfg.image_size)\n        self.transform = transforms.Compose([\n            transforms.Resize((cfg.image_size, cfg.image_size), interpolation=transforms.InterpolationMode.BICUBIC),\n            transforms.ToTensor(),\n            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),\n        ])',
        '        self.boxes = load_aligned_part_boxes(self.samples, gdino_path, cfg.image_size)\n        mean, std = input_normalization()\n        self.transform = transforms.Compose([\n            transforms.Resize((cfg.image_size, cfg.image_size), interpolation=transforms.InterpolationMode.BICUBIC),\n            transforms.ToTensor(),\n            transforms.Normalize(mean, std),\n        ])',
        "dataset normalization",
    )

    text = replace_between(
        text,
        "# -----------------------------\n# Visual backbone\n# -----------------------------",
        "# -----------------------------\n# Model\n# -----------------------------",
        VISUAL_SECTION.strip(),
    )

    text = replace_once(
        text,
        '    p.add_argument("--dino-model", default=cfg.dino_model)\n    p.add_argument("--image-size", type=int, default=cfg.image_size)',
        '    p.add_argument("--dino-model", default=cfg.dino_model,\n                   help="dinov2_vitb14, dino_vitb16, or clip_vitb16")\n    p.add_argument("--clip-init-checkpoint", default=cfg.clip_init_checkpoint,\n                   help="Checkpoint from the CLIP CUB finetune; only encoder.* is restored.")\n    p.add_argument("--image-size", type=int, default=cfg.image_size)',
        "CLI",
    )

    text = replace_once(
        text,
        '    cfg.dino_model = args.dino_model\n    cfg.image_size = args.image_size',
        '    cfg.dino_model = args.dino_model\n    cfg.clip_init_checkpoint = (os.path.abspath(os.path.expanduser(args.clip_init_checkpoint))\n                                if args.clip_init_checkpoint else "")\n    cfg.image_size = args.image_size',
        "argument application",
    )

    text = text.replace(
        'print(f"[DINO] trainable params after unfreeze={n_trainable:,}")',
        'print(f"[Backbone] trainable params after unfreeze={n_trainable:,}")',
    )
    text = text.replace(
        'print(f"[DINO] backbone frozen for first {cfg.freeze_backbone_epochs} epochs")',
        'print(f"[Backbone] frozen for first {cfg.freeze_backbone_epochs} epochs")',
    )
    text = text.replace(
        'print(f"[DINO] unfroze last {cfg.unfreeze_last_blocks} blocks at epoch {epoch}")',
        'print(f"[Backbone] unfroze last {cfg.unfreeze_last_blocks} blocks at epoch {epoch}")',
    )

    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="train_cub_shared_part_proto_finetune_reg_vitb16.py",
        help="Target script to patch in place.",
    )
    args = parser.parse_args()

    target = Path(args.target).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Target file not found: {target}")

    backup = target.with_name(target.stem + ".before_clip_init" + target.suffix)
    if backup.exists():
        raise FileExistsError(
            f"Backup already exists: {backup}. Remove or rename it before patching again."
        )

    original = target.read_text(encoding="utf-8")
    patched = patch_source(original)

    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(patched, encoding="utf-8")
        py_compile.compile(str(tmp), doraise=True)
        shutil.copy2(target, backup)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()

    print(f"Patched: {target}")
    print(f"Backup : {backup}")


if __name__ == "__main__":
    main()
