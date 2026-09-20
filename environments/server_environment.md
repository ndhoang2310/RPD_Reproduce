# School Server Environment Specification & Execution Guide

**Task ID**: W1-T3 — Prepare B0 Training Experiment  
**Target Hardware**: School Server (NVIDIA GeForce RTX 4090 24GB VRAM, 108GB RAM)  
**Baseline Model**: B0 — RPD-Net ($\theta = 0.5$) on PhenoBench  

---

## 1. Hardware & System Specifications

| Component | Specification | Notes |
|---|---|---|
| **GPU** | 1x NVIDIA GeForce RTX 4090 (24GB GDDR6X VRAM) | Compute Capability 8.9 (Ada Lovelace) |
| **System RAM** | 108 GB DDR4/DDR5 | Enables high `num_workers` (8–12) for dataset caching |
| **Host OS** | Linux (Ubuntu 20.04 / 22.04 LTS recommended) | Standard Linux cluster environment |
| **NVIDIA Driver** | >= 535.xx | Required for CUDA 12.1+ support |
| **CUDA Toolkit** | CUDA 12.1 / CUDA 12.2 | PyTorch wheels built with cu121 |
| **Storage** | NVMe SSD / Local Scratch | Checkpoint and dataset IO speed optimization |

---

## 2. VRAM Budget & OOM Prevention

### 2.1 Model & Batch Memory Footprint
- **Model Parameter Count**: ~189,000 parameters (only ~0.75 MB weights).
- **Batch Dimensions**:
  - Training: Batch size $, Crop size  \times 768 \times 3$.
  - Validation: Batch size $, Original size  \times 1024 \times 3$.
- **Observed VRAM Consumption**:
  - Model weights + Adam optimizer states: ~5 MB.
  - Forward + Backward activation maps: ~3.2 GB to 4.2 GB VRAM.
  - Validation inference: ~1.8 GB VRAM.
- **Headroom**: With 24GB VRAM available on the RTX 4090, training consumes under **18%** of total VRAM, leaving over **19GB of safety margin**.

### 2.2 Critical Memory Leaks Resolved
Before training on the server, the baseline codebase was patched against two historical memory leak vectors:
1. **PyTorch Lightning Tensor Accumulation**: In `models/model_multimetrics.py`, detached loss scalars and avoided keeping full-graph logits across validation/training loops.
2. **CUDA Fragmentation**: Set environment variable:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
   ```
   This prevents CUDA memory fragmentation during dynamic validation and visualization loops.

---

## 3. Storage & Directory Layout

To ensure reproducibility and prevent accidental loss of artifacts if the SSH session terminates:

```text
/home/<user>/
├── data/
│   └── PhenoBench/                <-- Dataset directory (persistent)
│       ├── train/
│       │   ├── images/
│       │   └── semantics/
│       └── val/
│           ├── images/
│           └── semantics/
└── RPD_Reproduce/                 <-- Cloned repository
    ├── configs/
    │   └── B0.yaml                <-- Frozen baseline config
    ├── environments/              <-- Setup scripts and requirements
    ├── results/
    │   ├── checkpoints/           <-- Checkpoints (.ckpt) saved here
    │   └── logs/                  <-- TensorBoard and Lightning logs
    └── run_train.sh               <-- Main execution wrapper
```

---

## 4. Persistent Execution via Tmux (Prevent SSH Disconnection Drops)

Always run the training process inside a `tmux` session so that training continues seamlessly if the SSH connection drops or network timeouts occur.

### Step 1: Create a dedicated tmux session
```bash
tmux new -s rpd_b0
```

### Step 2: Activate environment & configure CUDA
```bash
conda activate rpd_env
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

### Step 3: Launch training with logging
```bash
bash run_train.sh --dataset_dir /path/to/PhenoBench --max_epoch 4096 --check_val_every_n_epoch 10 2>&1 | tee results/logs/training.log
```

### Step 4: Detach from session safely
Press `Ctrl + B`, then release and press `D`.

### Step 5: Re-attach to check progress
```bash
tmux attach -t rpd_b0
```

---

## 5. Monitoring & Health Checks

While training is running in background, check GPU and system health in a separate terminal:

```bash
# Real-time GPU monitoring (updates every 1s)
watch -n 1 nvidia-smi

# System RAM and CPU worker utilization
htop

# Live-tailing training loss and validation mIoU
tail -f results/logs/training.log
```

---

## 6. Checkpoint Persistence & Artifact Recovery

- **Checkpoints saved**:
  - Best by validation loss: `results/checkpoints/B0_RPDNet_PhenoBench_<epoch>_<val_loss>.ckpt`
  - Best by validation mIoU: `results/checkpoints/B0_RPDNet_PhenoBench_<epoch>_<val_mIoU>.ckpt`
  - Latest state: `results/checkpoints/last.ckpt`
- **Automatic Resume Support**:
  If a session is interrupted, resume instantly by passing:
  ```bash
  python multi_metric_train.py --config configs/B0.yaml --resume True --ckpt_path results/checkpoints/last.ckpt
  ```
