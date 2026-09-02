#!/usr/bin/env python3
import eval_cub_active_resp_consistency_stability as base

_original_load_model = base.load_model

def _detect_backbone(checkpoint):
    raw_state = checkpoint.get("model", checkpoint)
    if not isinstance(raw_state, dict):
        return None
    keys = []
    for k in raw_state.keys():
        if k.startswith("module."):
            k = k[7:]
        keys.append(k)
    if any(k.startswith(("backbone.blocks.", "backbone.patch_embed.",
                         "backbone.cls_token", "backbone.pos_embed",
                         "backbone.mask_token")) for k in keys):
        return "dino"
    if any(k.startswith("backbone.vision_model.") for k in keys):
        return "clip"
    return None

def _autofix_load_model(train_module, checkpoint, device):
    detected = _detect_backbone(checkpoint)
    if detected is None:
        print("[Backbone autofix] Could not infer backbone; using saved config.")
        return _original_load_model(train_module, checkpoint, device)

    ckpt = dict(checkpoint)
    if isinstance(checkpoint.get("config"), dict):
        cfg_key = "config"
    elif isinstance(checkpoint.get("cfg"), dict):
        cfg_key = "cfg"
    else:
        cfg_key = "config"

    saved = dict(checkpoint.get(cfg_key, {}))
    old = saved.get("backbone", None)
    saved["backbone"] = detected

    if detected == "dino" and not saved.get("dino_model") and hasattr(train_module.cfg, "dino_model"):
        saved["dino_model"] = getattr(train_module.cfg, "dino_model")

    ckpt[cfg_key] = saved
    print(f"[Backbone autofix] checkpoint state_dict => {detected}")
    print(f"[Backbone autofix] saved config backbone: {old!r} -> {detected!r}")

    return _original_load_model(train_module, ckpt, device)

base.load_model = _autofix_load_model

if __name__ == "__main__":
    base.main()
