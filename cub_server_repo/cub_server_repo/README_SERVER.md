# CUB semantic-part additive prototype pipeline

This repository keeps **code only** under Git. CUB, DINO feature memmaps, GroundingDINO boxes, checkpoints, and visualizations are excluded by `.gitignore`.

## Repository layout

```text
.
├── train_cub_semantic_part_additive_proto_server.py
├── tools/
│   ├── build_dino_cache.py
│   └── build_gdino_part_boxes.py
├── scripts/
│   ├── download_cub.sh
│   ├── setup_env.sh
│   └── run_cub_pipeline.sh
├── requirements_server.txt
└── .gitignore
```

## 1. Put the code on GitHub

Run these commands on the machine that contains the code:

```bash
git init
git add .
git commit -m "Add CUB semantic part prototype pipeline"
git branch -M main
git remote add origin git@github.com:YOUR_NAME/YOUR_REPO.git
git push -u origin main
```

Do not commit `data/`, `artifacts/`, or `runs/`.

## 2. Clone on the server

```bash
git clone git@github.com:YOUR_NAME/YOUR_REPO.git
cd YOUR_REPO
```

Use an existing CUDA PyTorch environment, or create one appropriate for the server. After `torch` and `torchvision` are available:

```bash
bash scripts/setup_env.sh .venv
source .venv/bin/activate
```

On an HPC system that already supplies a Python environment, it is also acceptable to skip the virtual environment and run:

```bash
python -m pip install -r requirements_server.txt
```

## 3. Download CUB

```bash
bash scripts/download_cub.sh ./data
```

The script downloads the official Caltech archive, checks MD5, and extracts it to:

```text
./data/CUB_200_2011
```

## 4. Build the two required caches

Downloading CUB alone is not enough. The training program consumes:

1. DINOv2 ViT-B/14 CLS and patch-token memmaps.
2. GroundingDINO part boxes on CUB ground-truth bird crops.

Build DINO features:

```bash
python tools/build_dino_cache.py \
  --cub-root ./data/CUB_200_2011 \
  --output-dir ./artifacts/dino_vitb14_bbox224 \
  --batch-size 64 \
  --num-workers 4
```

Build GroundingDINO boxes:

```bash
python tools/build_gdino_part_boxes.py \
  --cub-root ./data/CUB_200_2011 \
  --output-dir ./artifacts/gdino_part_boxes_gtbbox_warp518 \
  --batch-size 4 \
  --resume
```

`--resume` periodically saves progress and continues from the saved `processed` index.

## 5. Train

```bash
python train_cub_semantic_part_additive_proto_server.py \
  --cub-root ./data/CUB_200_2011 \
  --dino-cache-dir ./artifacts/dino_vitb14_bbox224 \
  --gdino-box-dir ./artifacts/gdino_part_boxes_gtbbox_warp518 \
  --save-dir ./runs/semantic_part_additive_proto \
  --batch-size 128 \
  --num-workers 4
```

Or run the whole preparation and training chain:

```bash
bash scripts/run_cub_pipeline.sh
```

Extra training arguments are forwarded, for example:

```bash
BATCH_SIZE=64 NUM_WORKERS=8 bash scripts/run_cub_pipeline.sh --epochs 300 --no-viz
```

## 6. Run in a persistent server session

With `tmux`:

```bash
tmux new -s cubproto
source .venv/bin/activate
bash scripts/run_cub_pipeline.sh 2>&1 | tee run.log
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t cubproto
```

For Slurm, adapt this example:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=cub-proto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd /PATH/TO/YOUR_REPO
source .venv/bin/activate
NUM_WORKERS=8 BATCH_SIZE=128 bash scripts/run_cub_pipeline.sh
```

## Important consistency note

Both cache builders use the same image geometry:

```text
raw CUB image -> CUB ground-truth bird bbox crop -> square warp
```

DINO uses 224x224; GroundingDINO uses 518x518. The training loader scales GroundingDINO coordinates from 518 to 224. Do not mix these outputs with caches generated from full uncropped images.
