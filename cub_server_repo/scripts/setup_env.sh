#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"
python3 -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

python - <<'PY'
try:
    import torch
    print("Existing PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "available:", torch.cuda.is_available())
except Exception as exc:
    raise SystemExit(
        "PyTorch is not available in this environment. Install the CUDA build suitable for your server first, then rerun this script.\n"
        f"Original error: {exc}"
    )
PY

python -m pip install -r requirements_server.txt
python - <<'PY'
import torch, transformers, PIL, numpy
print("Environment check passed")
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
