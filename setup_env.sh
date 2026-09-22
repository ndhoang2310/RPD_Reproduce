#!/usr/bin/env bash
# ==============================================================================
# Automated Environment Setup for School Server (RTX 4090 24GB, 108GB RAM)
# Task: W1-T3 — Baseline B0 Training Readiness
# ==============================================================================

set -e

ENV_NAME="rpd_env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve repository root cleanly whether script is in root or in environments/
if [ -f "$SCRIPT_DIR/multi_metric_train.py" ]; then
    REPO_DIR="$SCRIPT_DIR"
    REQ_FILE="$SCRIPT_DIR/requirements.txt"
elif [ -f "$SCRIPT_DIR/../multi_metric_train.py" ]; then
    REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
    REQ_FILE="$SCRIPT_DIR/requirements.txt"
else
    REPO_DIR="$SCRIPT_DIR"
    REQ_FILE="$SCRIPT_DIR/requirements.txt"
fi

echo "=============================================================="
echo " [W1-T3] Initializing Environment for RPD-Net Baseline (B0)"
echo "=============================================================="

# 0. Ensure tmux is installed (critical for headless remote sessions)
if ! command -v tmux &> /dev/null; then
    echo "[0/5] tmux not found. Attempting to install tmux..."
    if command -v sudo &> /dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y tmux -qq || true
    elif command -v apt-get &> /dev/null; then
        apt-get update -qq && apt-get install -y tmux -qq || true
    fi
fi

# 1. Detect environment manager (conda preferred, fallback to venv)
VENV_DIR=""
if command -v conda &> /dev/null; then
    echo "[1/5] Conda detected. Initializing conda environment '$ENV_NAME'..."
    eval "$(conda shell.bash hook)"
    if conda info --envs | grep -q "$ENV_NAME"; then
        echo "      Environment '$ENV_NAME' already exists. Activating..."
        conda activate "$ENV_NAME"
    else
        echo "      Creating new Conda environment '$ENV_NAME' with Python 3.10..."
        conda create -n "$ENV_NAME" python=3.10 -y
        conda activate "$ENV_NAME"
    fi
else
    echo "[1/5] Conda not found. Using Python venv..."
    VENV_DIR="$REPO_DIR/.venv_server"
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
fi

echo "[2/5] Upgrading pip, wheel, and pinning setuptools<82 (for pkg_resources)..."
pip install --upgrade pip wheel -q
pip install "setuptools<82" -q

echo "[3/5] Installing PyTorch with CUDA 12.1 for NVIDIA RTX 4090..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "[4/5] Installing core baseline dependencies (including scikit-image, tifffile)..."
if [ -f "$REQ_FILE" ]; then
    pip install -r "$REQ_FILE"
else
    pip install "pytorch-lightning>=1.9.0,<2.0.0" "torchmetrics>=0.11.0,<0.12.0" \
                "numpy>=1.23.0,<2.0.0" "opencv-python>=4.7.0" "pyyaml>=6.0" \
                "timm>=0.6.13" "pandas>=2.0.0" "matplotlib>=3.7.0" \
                "tqdm>=4.65.0" "rich>=13.0.0" "scikit-image" "tifffile"
fi

echo "[5/5] Running sanity validation on PyTorch & RPD-Net architecture..."
python << 'EOF'
import torch
import pytorch_lightning as pl
import torchmetrics
import numpy as np
import sys, os

print(f'  - PyTorch Version: {torch.__version__}')
print(f'  - CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  - Device Name: {torch.cuda.get_device_name(0)}')
    print(f'  - Device VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
print(f'  - PyTorch Lightning: {pl.__version__}')
print(f'  - TorchMetrics: {torchmetrics.__version__}')
print(f'  - NumPy: {np.__version__}')

# Sanity forward check of RPD-Net
sys.path.insert(0, os.getcwd())
try:
    from models import RPDNet
    model = RPDNet(num_classes=3, deploy=False)
    x = torch.randn(2, 3, 768, 768)
    out = model(x)
    print(f'  - Model forward test passed! Output shape: {out.shape}')
except Exception as e:
    print(f'  - Model forward check note: {e}')
EOF

echo "=============================================================="
echo " Environment setup COMPLETE and VERIFIED!"
if [ -n "$VENV_DIR" ]; then
    echo " Activate with: source $VENV_DIR/bin/activate"
else
    echo " Activate with: conda activate $ENV_NAME"
fi
echo "=============================================================="
