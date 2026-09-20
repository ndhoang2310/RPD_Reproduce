# Development Log: W1-T3 — Prepare B0 Training Experiment

| Metadata | Details |
|---|---|
| **Task ID** | W1-T3 |
| **Week** | Week 1 |
| **Task Name** | Prepare B0 Training Experiment |
| **Owner** | HoangND — Learning Method / Baseline Lead |
| **Support** | PhuongTU — Data & Evaluation Lead |
| **Date** | 2026-09-20 |
| **Status** | Completed (Ready for School Server / W1-T4) |
| **Target Hardware** | School Server (1x NVIDIA RTX 4090 24GB VRAM, 108GB RAM) |
| **Git Repository** | https://github.com/ndhoang2310/RPD_Reproduce.git |

---

## 1. Objective & Context

The primary goal of **W1-T3** is to make the entire baseline training pipeline for **B0 — RPD-Net** ($\theta = 0.5$) on the PhenoBench dataset **fully verified, hardened, and ready for immediate execution** before accessing limited GPU hours on the school server.

Zero GPU time should be spent debugging dependencies, fixing path syntax, troubleshooting deprecated PyTorch APIs, or crashing due to Out-Of-Memory (OOM) errors.

---

## 2. Baseline Source Compatibility Audit & Technical Fixes

Upon deep inspection of the original author repository (`RPD/`), multiple critical compatibility and architectural bugs were identified and fixed. All modifications are classified strictly as **compatibility & stability modifications**, preserving exact network math and training dynamics.

### 2.1 Package & Import Structure Normalization
- **Issue**: The original codebase had conflicting module names (`modules/` and `pdc_datasets/`), which broke execution when installed or imported in modern environments.
- **Fix**: Standardized directory hierarchy to `models/` and `datasets/`, adding backward-compatible symlinks (`modules -> models`, `pdc_datasets -> datasets`) to preserve both legacy scripts and clean imports.

### 2.2 NumPy 2.0 Compatibility Patch
- **Issue**: Under recent Python environments running NumPy $\ge 2.0$, calls to deprecated aliases (`np.Inf`, `np.bool`, `np.float`, `np.int`) crashed dataset transformations and IoU calculations.
- **Fix**: Injected runtime compatibility shims in `multi_metric_train.py`, `convert_multi_test.py`, and `deploy_convert_multi_test.py`:
  ```python
  if not hasattr(np, 'Inf'): np.Inf = np.inf
  if not hasattr(np, 'bool'): np.bool = bool
  if not hasattr(np, 'float'): np.float = float
  if not hasattr(np, 'int'): np.int = int
  ```

### 2.3 PyTorch Lightning GPU Accelerator Support
- **Issue**: Original scripts passed `gpus=1` (deprecated in PyTorch Lightning $\ge 1.9$ and removed in $2.0$). On modern multi-GPU or newer single-GPU nodes, it defaulted to CPU training.
- **Fix**: Replaced with modern `accelerator='gpu'`, `devices=cfg['train']['devices']`, and verified device placement.

### 2.4 Multiprocessing Serialization Error in `RPD_ops.py`
- **Issue**: The PyTorch DataLoader with `num_workers > 0` threw:
  `AttributeError: Can't get local object 'createConvFunc.<locals>.func'`
  because nested inner functions inside `createConvFunc` cannot be pickled by Python's multiprocessing engine.
- **Fix**: Refactored the functional closure into a top-level picklable callable class `ConvOp` inside `models/RPD_ops.py`. This allows multiprocessing data workers (`num_workers=8`) to function with zero worker IPC crashes.

### 2.5 CRITICAL OOM Memory Leak in `SegmentationNetwork`
- **Issue**: In `models/model_multimetrics.py`, the author accumulated un-detached output dictionaries in `self.training_step_outputs` and `self.validation_step_outputs`:
  ```python
  # Original leaky code:
  self.training_step_outputs.append({'loss': loss, 'logits': logits, 'anno': anno})
  ```
  Because `logits` retained the entire dynamic autograd computation graph of each $768 \times 768$ batch, VRAM grew monotonically every step until CUDA OOM crashed the run after 5–15 epochs.
- **Fix**:
  1. Detached loss scalars (`loss.detach()`).
  2. Prevented storing full-resolution logits and annotations across steps.
  3. Formally invoked `.clear()` on step output buffers at `on_train_epoch_end()` and `on_validation_epoch_end()`.
  4. Added `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` to eliminate CUDA caching fragmentation.

### 2.6 Removal of Hardcoded Author Paths & Parameterized CLI
- **Issue**: Author evaluation and re-parameterization scripts contained hardcoded paths to author's local workstation (`/home/cfh/...`).
- **Fix**: Parameterized all paths via CLI arguments and configuration files:
  `--dataset_dir`, `--batch_size`, `--max_epoch`, `--devices`, `--num_workers`, `--check_val_every_n_epoch`, `--early_stopping_patience`, `--no_early_stopping`, `--resume`, `--ckpt_path`.

---

## 3. Paper vs Codebase Discrepancy Analysis

| Dimension | Paper Specification | Author Codebase (`config_deeplearn.yaml`) | Decision for B0 Baseline |
|---|---|---|---|
| **Theta ($\theta$)** | Default $\theta=1.0$; best reported $\theta=0.5$ | $\theta=0.5$ in architecture | **Frozen at $\theta = 0.5$** |
| **Learning Rate** | 16-epoch warm-up to 1e-4, then poly decay $(1 - e/4096)^3$ | Base LR: 5e-4 with Adam optimizer | **Keep 5e-4** (align with official repo; document paper note) |
| **Class Weights** | Inversely proportional to pixel frequencies | Hardcoded `[1.47, 5.06, 10.02]` | **Frozen at `[1.47, 5.06, 10.02]`** |
| **Validation Interval** | Every 200 epochs | `check_val_every_n_epoch: 150` | **Set to 10 for profiling**, 150/200 for full runs |
| **Early Stopping** | 3 consecutive validations with val_loss $\le 0.1$ | PyTorch Lightning `EarlyStopping(monitor='val_loss', patience=3)` | **Supported via CLI** (can disable with `--no_early_stopping`) |
| **Batch Size** | 4 (crop 768 $\times$ 768) | 4 | **Frozen at 4** |
| **Optimizer** | Adam, weight decay 2e-4 | Adam, weight decay 2e-4 | **Frozen at Adam (2e-4)** |

---

## 4. Pre-Server Sanity Checks & Trial Run Verification

### 4.1 1-Batch Dry Run Sanity Check
- **Objective**: Verify that forward pass, weighted cross-entropy loss computation, backward pass, and Adam optimizer step execute without NaN, Inf, or memory spikes.
- **Result**: PASSED. Loss = 1.0941, gradients computed, optimizer stepped cleanly.

### 4.2 Full 70-Epoch Training Trial Run (Kaggle T4x2 / RTX Equivalent)
A complete trial run was executed to validate convergence behavior, early stopping, and checkpoint generation:
- **Training Duration**: 70 epochs (~3 min 40s per epoch).
- **Early Stopping Trigger**: Epoch 69 (patience = 3).
- **Peak Validation Performance (Epoch 54)**:
  - **Overall Mean IoU (mIoU)**: **84.73%**
  - **Background / Soil IoU**: **99.25%**
  - **Crop (Sugarbeet) IoU**: **94.38%**
  - **Weed IoU**: **60.56%**
  - **Overall Mean F1 (mF1)**: **90.72%**
  - **Weed F1**: **75.43%**
  - **Overall Recall**: **95.49%**
  - **Overall Precision**: **87.10%**
- **Checkpoint Verification**: Generated valid `.ckpt` files (2.8 MB each) under `results/checkpoints/`.

### 4.3 Structural Re-parameterization (Deploy) Verification
- Executed `run_convert.sh` to merge the 4-branch PDC module (`cv`, `cd`, `ad`, `rd`) into a single $5 \times 5$ convolution kernel.
- **Parameter Reduction**:
  - Training multi-branch model: ~189,000 parameters (~0.75 MB).
  - Deployed single-branch model: ~63,000 parameters (~0.25 MB) — **66.7% reduction**.
- **Inference Equivalence**: Verified that numerical difference between multi-branch and single-branch predictions is $< 1e-5$.

---

## 5. Artifact Verification & Git Readiness

All required W1-T3 deliverables are committed and ready in repository:
1. Frozen configuration: `configs/B0.yaml`
2. Environment package list: `environments/requirements.txt`
3. Automated setup script: `environments/setup_env.sh`
4. Server execution guide: `environments/server_environment.md`
5. Development log: `docs/dlogs/W1-T3_dlog.md`
6. Training launch note: `docs/training_launch_note.md`
7. Launcher scripts: `run_train.sh`, `run_convert.sh`, `summarize_results.py`

**Conclusion**: Pipeline is 100% prepared. Ready for handoff to **W1-T4 (Profile B0 Training and Assess Compute Feasibility)**.
