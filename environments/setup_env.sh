#!/usr/bin/env bash
# ==============================================================================
# Automated Environment Setup for School Server (RTX 4090 24GB, 108GB RAM)
# Task: W1-T3 — Baseline B0 Training Readiness
# ==============================================================================

set -e

ENV_NAME="rpd_env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================================="
echo " [W1-T3] Initializing Environment for RPD-Net Baseline (B0)"
echo "=============================================================="

# 1. Detect environment manager (conda preferred, fallback to venv)
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
    VENV_DIR="$PROJECT_ROOT/.venv_server"
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
fi

echo "[2/5] Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel -q

echo "[3/5] Installing PyTorch with CUDA 12.1 for NVIDIA RTX 4090..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "[4/5] Installing core baseline dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "[5/5] Running sanity validation on PyTorch & RPD-Net architecture..."
python -c "
import torch
import pytorch_lightning as pl
import torchmetrics
import numpy as np

print(f'  - PyTorch Version: {torch.__version__}')
print(f'  - CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  - Device Name: {torch.cuda.get_device_name(0)}')
    print(f'  - Device VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
print(f'  - PyTorch Lightning: {pl.__version__}')
print(f'  - TorchMetrics: {torchmetrics.__version__}')
print(f'  - NumPy: {np.__version__}')

# Sanity forward check of RPD-Net
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) if "__file__" in locals() else os.getcwd())
from models.RPDNet import RPDNet
model = RPDNet(num_classes=3, theta=0.5)
x = torch.randn(2, 3, 768, 768)
out = model(x)
print(f'  - Model forward test passed! Output shape: {out.shape}')
"

echo "=============================================================="
echo " Environment setup COMPLETE and VERIFIED for School Server!"
echo " Activate with: conda activate $ENV_NAME (or source $VENV_DIR/bin/activate)"
echo "=============================================================="
