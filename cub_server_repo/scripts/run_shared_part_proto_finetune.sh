#!/usr/bin/env bash
set -euo pipefail

CUB_ROOT=${CUB_ROOT:-./data/CUB_200_2011}
GDINO_BOX_DIR=${GDINO_BOX_DIR:-./artifacts/gdino_part_boxes_gtbbox_warp518}
SAVE_DIR=${SAVE_DIR:-./runs/shared_part_proto_finetune_vitb14}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
EPOCHS=${EPOCHS:-100}

python train_cub_shared_part_proto_finetune.py \
  --cub-root "$CUB_ROOT" \
  --gdino-box-dir "$GDINO_BOX_DIR" \
  --save-dir "$SAVE_DIR" \
  --dino-model dinov2_vitb14 \
  --image-size 224 \
  --parts head,wing,body,tail \
  --k-per-part 16 \
  --score-mode resp_sum \
  --unfreeze-last-blocks 2 \
  --lr-backbone 1e-5 \
  --lr-router 3e-5 \
  --lr-proto 3e-5 \
  --lr-classifier 1e-4 \
  --ema-rho 0.95 \
  --ema-sem-mix 0.5 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  "$@"
