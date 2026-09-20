# Hướng Dẫn Thực Chiến Reproduce RPD-Net Trên Server Trường (SSH)

Tài liệu này hướng dẫn chi tiết từng bước từ lúc kết nối SSH vào server phòng lab/trường (cấu hình **NVIDIA RTX 4090 24GB VRAM**, **108GB RAM**) cho tới khi huấn luyện, chuyển đổi mô hình và xuất báo cáo hoàn chỉnh.

---

## 1. Thông Số Huấn Luyện Chuẩn Theo Paper (IEEE TGRS 2026)

Mô hình và cấu hình đã được thiết lập sẵn trong file [`config/config_paper.yaml`](config/config_paper.yaml):

| Tham số | Giá trị chuẩn theo Paper | Ghi chú cho RTX 4090 & 108GB RAM |
| :--- | :--- | :--- |
| **Model** | `RPDNet` | Re-parameterized Pixel Difference Network |
| **Khối PDC** | `'pdc'` ($\theta = 0.5$) | 4 nhánh song song: Vanilla (`cv`), Center (`cd`), Angular (`ad`), Radial (`rd`) |
| **Số lớp** | `3` | 0: Background/Soil, 1: Crop (Sugarbeet), 2: Weed |
| **Loss** | Weighted CrossEntropy | Trọng số: `[1.47, 5.06, 10.02]` (Weed phạt nặng nhất) |
| **Optimizer** | `Adam` | `learning_rate: 5.0e-4`, `weight_decay: 2.0e-04` |
| **LR Schedule** | Polynomial Decay | Warmup 16 epochs, suy giảm lũy thừa bậc 3 |
| **Batch Size** | `4` | Chuẩn paper (chỉ chiếm ~3.5GB / 24GB VRAM trên RTX 4090) |
| **Crop Size** | $768 \times 768$ (Train) / $1024 \times 1024$ (Val) | Random scale (1.0 - 1.1), Random crop, Flips, Color Jitter |
| **num_workers** | `8` hoặc `12` | Server có 108GB RAM, nạp dữ liệu song song cực nhanh |
| **max_epoch** | `300` (hoặc `4096`) | Có thể tùy chỉnh bằng cờ `--max_epoch` |

---

## 2. Quy Trình 5 Bước Thực Hiện Trên Server

### Bước 1: Kết nối SSH & Mở phiên `tmux` (BẮT BUỘC)
> **Lưu ý quan trọng:** Khi chạy qua SSH, nếu mạng bị ngắt hoặc bạn gập màn hình máy Mac, tiến trình train sẽ bị hủy nếu không dùng `tmux` hoặc `nohup`.

```bash
# 1. SSH vào server trường
ssh username@server_ip

# 2. Tạo một phiên làm việc độc lập với tmux (đặt tên là rpd)
tmux new -s rpd
```
*(Nếu rớt mạng, bạn chỉ cần SSH lại và gõ `tmux a -t rpd` là trở lại ngay màn hình đang train).*

---

### Bước 2: Clone Repo & Cài đặt Môi trường Tự động

```bash
# 1. Clone repo của bạn
git clone https://github.com/ndhoang2310/RPD_Reproduce.git
cd RPD_Reproduce

# 2. Chạy script cài đặt tự động (tạo conda/venv + PyTorch CUDA 12.1 + dependencies)
bash setup_env.sh

# 3. Kích hoạt môi trường vừa tạo
conda activate rpd_env
# (hoặc: source rpd_env/bin/activate nếu không dùng conda)
```

---

### Bước 3: Chuẩn bị Dataset PhenoBench
Đảm bảo thư mục dataset có cấu trúc chuẩn:
```text
PhenoBench/
├── train/
│   ├── images/
│   └── semantics/
└── val/
    ├── images/
    └── semantics/
```

---

### Bước 4: Bắt đầu Huấn Luyện (Training)

#### Cách 1: Chạy bằng Shell Script (Được khuyến nghị)
```bash
bash run_train.sh /path/to/PhenoBench
```

#### Cách 2: Chạy trực tiếp với Python CLI (Tùy biến cao)
```bash
python multi_metric_train.py \
    --config ./config/config_paper.yaml \
    --export_dir ./log_dir \
    --dataset_dir /path/to/PhenoBench \
    --batch_size 4 \
    --max_epoch 300 \
    --num_workers 8 \
    --devices 1
```

*Một số tùy chọn hữu ích:*
- `--no_early_stopping`: Tắt tự động dừng để huấn luyện đủ số epoch đặt ra.
- `--resume`: Tự động tiếp tục từ file `last.ckpt` nếu trước đó bị dừng.
- `--check_val_every_n_epoch 5`: Tần suất đánh giá validation (mặc định mỗi 5 epoch).

---

### Bước 5: Giám sát Quá trình Huấn Luyện

Mở một cửa sổ terminal khác trên server:
```bash
# Giám sát GPU và nhiệt độ
watch -n 1 nvidia-smi

# Xem log Tensorboard từ xa
tensorboard --logdir ./log_dir --port 6006
```
*(Trên máy Mac, bạn có thể forward port: `ssh -N -L 6006:localhost:6006 username@server_ip` rồi mở trình duyệt vào `http://localhost:6006`)*.

---

## 3. Sau Khi Huấn Luyện Xong

### 1. Xuất bảng tổng hợp kết quả (CSV & Biểu đồ)
```bash
python summarize_results.py ./log_dir
```
Script sẽ tự động tìm checkpoint tốt nhất và xuất file `metrics_summary.csv` cùng ảnh đồ thị huấn luyện `training_metrics_plot.png`.

### 2. Re-parameterization (Convert mô hình sang Single-branch)
Sau khi huấn luyện xong mô hình đa nhánh, chạy script chuyển đổi để gộp toàn bộ các nhánh PDC về 1 nhánh tích chập duy nhất:
```bash
bash run_convert.sh
```
Mô hình đã chuyển đổi sẽ được lưu tại `./log_dir/deploy_testing/deploy_model.ckpt`, sẵn sàng cho inference thời gian thực với tốc độ >40 FPS.

### 3. Tải kết quả về máy tính cá nhân (từ máy Mac)
Từ terminal trên máy Mac của bạn:
```bash
scp -r username@server_ip:/path/to/RPD_Reproduce/log_dir/ ./result_server/
```
