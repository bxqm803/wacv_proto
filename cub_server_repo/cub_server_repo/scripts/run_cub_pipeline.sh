#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/CUB_200_2011}"
DINO_CACHE="${DINO_CACHE:-${ROOT}/artifacts/dino_vitb14_bbox224}"
GDINO_CACHE="${GDINO_CACHE:-${ROOT}/artifacts/gdino_part_boxes_gtbbox_warp518}"
SAVE_DIR="${SAVE_DIR:-${ROOT}/runs/semantic_part_additive_proto}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"

cd "${ROOT}"

if [[ ! -f "${DATA_ROOT}/images.txt" ]]; then
  bash scripts/download_cub.sh "${ROOT}/data"
fi

python tools/build_dino_cache.py \
  --cub-root "${DATA_ROOT}" \
  --output-dir "${DINO_CACHE}" \
  --batch-size "${DINO_BATCH_SIZE:-64}" \
  --num-workers "${NUM_WORKERS}"

python tools/build_gdino_part_boxes.py \
  --cub-root "${DATA_ROOT}" \
  --output-dir "${GDINO_CACHE}" \
  --batch-size "${GDINO_BATCH_SIZE:-4}" \
  --resume

python train_cub_semantic_part_additive_proto_server.py \
  --cub-root "${DATA_ROOT}" \
  --dino-cache-dir "${DINO_CACHE}" \
  --gdino-box-dir "${GDINO_CACHE}" \
  --save-dir "${SAVE_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  "$@"
