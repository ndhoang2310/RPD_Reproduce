# Training Launch Note: Baseline B0 (RPD-Net on PhenoBench)

**Task ID**: W1-T3 — Prepare B0 Training Experiment  
**Author**: HoangND — Learning Method / Baseline Lead  
**Target Next Task**: W1-T4 — Profile B0 Training and Assess Compute Feasibility  
**Target Hardware**: School Server (1x NVIDIA GeForce RTX 4090 24GB VRAM, 108GB RAM)  
**Git Repository**: `https://github.com/ndhoang2310/RPD_Reproduce.git`  

---

## Executive Summary: Answers to the 10 Core Questions

### 1. B0 sẽ được train bằng source code nào?
- **Source code base**: Toàn bộ source code trong repository `RPD_Reproduce` (thư mục `RPD/`).
- **Main Entry Point**: `multi_metric_train.py`.
- **Model Definition**: `models/RPDNet.py` kết hợp với `models/RPD_ops.py` (RPD block với 4 nhánh Pixel Difference Convolution: `cv`, `cd`, `ad`, `rd` và $\theta = 0.5$).
- **Lightning Module**: `models/model_multimetrics.py` (`SegmentationNetwork`).
- **Git Commit**: Commit mới nhất trên branch `main` của `https://github.com/ndhoang2310/RPD_Reproduce.git`.

### 2. B0 sử dụng chính xác configuration nào?
- **Config file chuẩn**: `configs/B0.yaml` (đã được freeze và lưu trữ trong repo).
- **Core Parameters**:
  - `backbone.name`: `RPDNet`
  - `backbone.num_classes`: 3 (0: Soil/Background, 1: Crop/Sugarbeet, 2: Weed)
  - `backbone.theta`: `0.5`
  - `backbone.pretrained`: `false` (train from scratch)
  - `train.batch_size`: `4`
  - `train.learning_rate`: `1.0e-4` (chuẩn IEEE TGRS 2026 paper)
  - `train.max_epoch`: `4096` (paper protocol) / điều chỉnh qua cờ `--max_epoch` khi profiling ở W1-T4
  - `val.check_val_every_n_epoch`: `200` (chuẩn paper: validate mỗi 200 epoch; override `--check_val_every_n_epoch 10` khi profiling)
  - `early_stopping`: `PaperEarlyStopping` (dừng sau 3 lần val liên tiếp có $\text{val\_loss} \le 0.1$)
  - `data.num_workers`: `8` (tận dụng 108GB RAM server)
  - `seed`: `1682409321`

### 3. Dataset được lấy từ đâu và code đọc nó như thế nào?
- **Dataset**: `PhenoBench` (official train & validation split từ W1-T2).
- **Directory Structure**:
  ```text
  PhenoBench/
  ├── train/
  │   ├── images/     # RGB .png (1024x1024)
  │   └── semantics/  # Semantic label .png (0: Bg, 1: Crop, 2: Weed, 255: Ignore)
  └── val/
      ├── images/
      └── semantics/
  ```
- **Dataset Loader**: Lớp `PDC` trong `datasets/pdc.py`.
- **Đọc dữ liệu**:
  - Training: Đọc ảnh RGB + ground truth semantics, áp dụng color jittering, random hflip/vflip (p=0.5), random scale (1.0–1.1), sau đó random crop về $768 \times 768$.
  - Validation: Center crop về $1024 \times 1024$.
  - Truyền đường dẫn linh hoạt qua cờ: `--dataset_dir /path/to/PhenoBench`.

### 4. Loss / optimizer / LR schedule / augmentation / checkpoint rule là gì?
- **Loss**: Weighted Cross Entropy (`CrossEntropy` trong `models/losses.py`).
  - Class weights: `[1.47, 5.06, 10.02]` (inverse pixel frequency).
  - Masking: Điểm ảnh có nhãn `255` (ignore/boundary) được loại bỏ khỏi loss (`anno != 255`).
- **Optimizer**: Adam với `weight_decay = 2.0e-4`.
- **Learning Rate Schedule**:
  - Base LR: `1.0e-4` (khớp hoàn toàn với Section IV-A-3 của bài báo).
  - Warm-up: 16 epoch đầu tăng tuyến tính từ 0 đến base LR ($1 \times 10^{-4}$).
  - Sau warm-up: Polynomial decay theo công thức $(1 - \frac{e - 17}{4096 - 17})^3$.
- **Augmentation**:
  - Brightness: $[0.6, 1.4]$, Contrast: $[0.6, 1.4]$, Saturation: $[0.8, 1.2]$, Hue: $[-0.0125, 0.0125]$.
  - HFlip: $p=0.5$, VFlip: $p=0.5$, Scale: $[1.0, 1.1]$, Crop: $768 \times 768$.
- **Checkpoint & Early Stopping Rule**:
  - **Early Stopping**: `PaperEarlyStopping` dừng sau **3 lần validation liên tiếp có $\text{val\_loss} \le 0.1$** (tương đương $\ge 600$ epoch).
  - Lưu checkpoint tốt nhất theo **Validation mIoU cao nhất** (`val_mIoU` max).
  - Lưu checkpoint tốt nhất theo **Validation Loss thấp nhất** (`val_loss` min).
  - Tự động duy trì `last.ckpt` để sẵn sàng resume.

### 5. Lệnh hoặc entry point nào dùng để bắt đầu training?
- **Script bao bọc (khuyên dùng)**:
  ```bash
  bash run_train.sh --dataset_dir /path/to/PhenoBench --max_epoch 4096
  ```
- **Lệnh Python trực tiếp (chuẩn Paper)**:
  ```bash
  python multi_metric_train.py \
    --config configs/B0.yaml \
    --dataset_dir /path/to/PhenoBench \
    --export_dir results \
    --devices 1 \
    --batch_size 4 \
    --max_epoch 4096
  ```
- **Lệnh Python khi chạy Profiling nhanh (W1-T4 / W1-T5)**:
  ```bash
  python multi_metric_train.py \
    --config configs/B0.yaml \
    --dataset_dir /path/to/PhenoBench \
    --export_dir results \
    --devices 1 \
    --batch_size 4 \
    --max_epoch 50 \
    --check_val_every_n_epoch 10
  ```

### 6. Training output sẽ được lưu ở đâu?
- **Toàn bộ output lưu trong thư mục `results/`**:
  - Checkpoints: `results/checkpoints/` (`*.ckpt`, bao gồm `last.ckpt`).
  - Metrics per-class: `results/evaluation/epoch-XXXXXX/` (`IoU.yaml`, `F1.yaml`, `Precision.yaml`, `Recall.yaml`).
  - Lightning logs & TensorBoard: `results/lightning_logs/`.
  - Console text log: `results/logs/training.log`.

### 7. Nếu server session kết thúc thì checkpoint/log có còn không?
- **Có**: Thư mục `results/` nằm trong ổ cứng persistent storage của user trên server, không bị xóa khi tắt terminal.
- **Để tránh tiến trình bị kill khi SSH disconnect**: Phải luôn chạy trong `tmux` hoặc `screen`.
- **Cơ chế Resume**: Nếu bị ngắt đột ngột (mất điện, timeout phiên), chạy lại lệnh với cờ `--resume True --ckpt_path results/checkpoints/last.ckpt`. Model sẽ tiếp tục train từ epoch bị dừng mà không mất tiến trình.

### 8. Server trường yêu cầu setup hoặc thao tác gì trước khi chạy?
1. Đăng nhập qua SSH: `ssh user@server_ip -p port`.
2. Kiểm tra GPU: `nvidia-smi` (xác nhận có 1x RTX 4090 24GB).
3. Tạo môi trường Python: chạy script `bash environments/setup_env.sh` (tự động cài PyTorch 2.1+ CUDA 12.1 và các package).
4. Xác nhận dataset path: Kiểm tra thư mục chứa PhenoBench trên server.

### 9. Source code hiện tại có lỗi compatibility nào cần sửa trước không?
- **Đã sửa xong 100% trước khi lên server**:
  - [x] Lỗi package import (`modules`, `pdc_datasets`).
  - [x] Lỗi tương thích NumPy 2.0 (`np.Inf`, `np.bool`, `np.float`).
  - [x] Lỗi PyTorch Lightning GPU device mapping.
  - [x] Lỗi IPC pickling trong DataLoader worker (`ConvOp`).
  - [x] **Lỗi rò rỉ bộ nhớ VRAM (OOM Leak)**: đã detach loss tensor và dọn dẹp bộ nhớ đệm step outputs.
  - [x] Thêm cấu hình chống phân mảnh: `expandable_segments:True`.
  - [x] Loại bỏ toàn bộ hardcoded paths của tác giả gốc.

### 10. Khi lên server, nhóm cần làm những bước nào trước khi bấm chạy training?
Thực hiện đúng 5 bước tuần tự trong mục "Standard Operating Procedure" dưới đây.

---

## Standard Operating Procedure (SOP) on School Server

```bash
# BƯỚC 1: Clone repo và di chuyển vào thư mục project
git clone https://github.com/ndhoang2310/RPD_Reproduce.git
cd RPD_Reproduce

# BƯỚC 2: Chạy script setup môi trường tự động (chỉ mất ~2 phút)
bash environments/setup_env.sh
conda activate rpd_env

# BƯỚC 3: Mở một phiên tmux mới để chống rớt kết nối SSH
tmux new -s b0_train

# BƯỚC 4: Chạy Sanity Check tối thiểu (1 batch forward/loss/backward)
python -c "
import torch
from models.RPDNet import RPDNet
from models.losses import CrossEntropy

model = RPDNet(num_classes=3, theta=0.5).cuda()
loss_fn = CrossEntropy(weights=[1.47, 5.06, 10.02]).cuda()
x = torch.randn(4, 3, 768, 768, device='cuda')
y = torch.randint(0, 3, (4, 768, 768), device='cuda')
logits = model(x)
loss = loss_fn(logits, y, mode='train')
loss.backward()
print('Sanity Check PASSED! Initial Loss:', float(loss))
"

# BƯỚC 5: Khởi chạy training chính thức (ghi log đồng thời ra màn hình và file)
mkdir -p results/logs
python multi_metric_train.py \
  --config configs/B0.yaml \
  --dataset_dir /path/to/PhenoBench \
  --export_dir results \
  --devices 1 \
  --batch_size 4 \
  --max_epoch 4096 \
  --check_val_every_n_epoch 10 2>&1 | tee results/logs/training.log

# Phím tắt tách khỏi tmux: Nhấn Ctrl+B rồi nhấn D
```
