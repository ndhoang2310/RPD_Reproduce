# Hướng Dẫn Thực Chiến Reproduce RPD-Net Trên Server Trường (SSH)

Tài liệu này hướng dẫn chi tiết từng bước từ lúc kết nối SSH vào server phòng lab/trường (cấu hình **NVIDIA RTX 4090 24GB VRAM**, **108GB RAM**) cho tới khi huấn luyện, chuyển đổi mô hình và xuất báo cáo hoàn chỉnh cho **Baseline B0**.

---

## 1. Thông Số Huấn Luyện Chuẩn Theo Paper (IEEE TGRS 2026)

Mô hình và cấu hình đã được freeze sẵn trong file [`configs/B0.yaml`](configs/B0.yaml):

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
| **num_workers** | `8` | Server có 108GB RAM, nạp dữ liệu song song cực nhanh |
| **max_epoch** | `4096` | Chuẩn paper; có thể tinh chỉnh nhanh qua cờ `--max_epoch` |

---

## 2. Quy Trình 5 Bước Thực Hiện Trên Server

### Bước 1: Kết nối SSH & Mở phiên `tmux` (BẮT BUỘC)
> **Lưu ý quan trọng:** Khi chạy qua SSH, nếu mạng bị ngắt hoặc bạn gập màn hình máy Mac, tiến trình train sẽ bị hủy nếu không dùng `tmux`.

```bash
# 1. SSH vào server trường
ssh username@server_ip -p port

# 2. Tạo một phiên làm việc độc lập với tmux (đặt tên là rpd_b0)
tmux new -s rpd_b0
```
*(Nếu rớt mạng, bạn chỉ cần SSH lại và gõ `tmux attach -t rpd_b0` là trở lại ngay màn hình đang train).*

---

### Bước 2: Clone Repo & Cài đặt Môi trường Tự động

```bash
# 1. Clone repo của bạn
git clone https://github.com/ndhoang2310/RPD_Reproduce.git
cd RPD_Reproduce

# 2. Chạy script cài đặt tự động (tạo conda/venv + PyTorch CUDA 12.1 + dependencies)
bash environments/setup_env.sh

# 3. Kích hoạt môi trường vừa tạo
conda activate rpd_env
# (hoặc: source .venv_server/bin/activate nếu không dùng conda)
```

---

### Bước 3: Chuẩn bị Dữ liệu PhenoBench (Tự động hóa)

Chạy script hỗ trợ chuẩn bị dữ liệu:
```bash
bash prepare_data.sh
```
Script sẽ cho bạn 3 tùy chọn:
1. **Tải tự động từ Kaggle**: Điền file `kaggle.json` vào `~/.kaggle/`, script sẽ tự tải file 7.6GB về và giải nén vào `data/PhenoBench` (chỉ mất ~2 phút).
2. **Giải nén file zip có sẵn**: Nếu bạn chuyển file `PhenoBench-v110.zip` lên server.
3. **Tạo symlink**: Nếu server đã có sẵn thư mục PhenoBench ở thư mục dùng chung (`/data/`).

---

### Bước 4: Bắt đầu Huấn Luyện (Training)

#### Cách 1: Chạy bằng Shell Script (Khuyên dùng)
```bash
bash run_train.sh --dataset_dir ./data/PhenoBench 2>&1 | tee results/logs/training.log
```

#### Cách 2: Chạy trực tiếp với Python CLI
```bash
python multi_metric_train.py \
    --config configs/B0.yaml \
    --dataset_dir ./data/PhenoBench \
    --export_dir results 2>&1 | tee results/logs/training.log
```

*Tách khỏi tmux để tắt máy đi ngủ:*
- Nhấn tổ hợp `Ctrl + B`, thả tay ra rồi nhấn `D`.
- Tiến trình vẫn tiếp tục chạy ngầm 100% trên server.

---

### Bước 5: Giám sát Quá trình Huấn Luyện

Mở một cửa sổ terminal khác trên server:
```bash
# 1. Giám sát GPU, VRAM và nhiệt độ (cập nhật mỗi giây)
watch -n 1 nvidia-smi

# 2. Theo dõi CPU & RAM hệ thống
htop

# 3. Xem tiến độ epoch và validation mIoU trực tiếp
tail -f results/logs/training.log
```

---

## 3. Sau Khi Huấn Luyện Xong

### 1. Xuất bảng tổng hợp kết quả (CSV & Biểu đồ)
```bash
python summarize_results.py results
```
Script sẽ tự động tìm checkpoint tốt nhất và xuất file `metrics_summary.csv` cùng ảnh đồ thị huấn luyện `training_metrics_plot.png`.

### 2. Re-parameterization (Convert mô hình sang Single-branch)
Sau khi huấn luyện xong mô hình đa nhánh, chạy script chuyển đổi để gộp toàn bộ các nhánh PDC về 1 nhánh tích chập duy nhất:
```bash
bash run_convert.sh results/checkpoints/last.ckpt
```
Mô hình đã chuyển đổi sẽ giảm tham số từ 0.189M xuống 0.063M, sẵn sàng cho inference thời gian thực với tốc độ >40 FPS.

### 3. Tải kết quả về máy tính cá nhân (từ máy Mac)
Từ terminal trên máy Mac của bạn:
```bash
rsync -avzP username@server_ip:~/RPD_Reproduce/results/ ./server_results/
```
