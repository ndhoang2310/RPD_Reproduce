#!/usr/bin/env python3
"""
RPD Model Training Results Summary & Visualization Tool
Tự động tổng hợp kết quả huấn luyện từ log_dir:
1. Trích xuất mIoU, Precision, Recall, F1 theo từng Epoch và theo từng Class (Background, Crop, Weed).
2. Tìm ra Epoch tốt nhất (Best Epoch).
3. Xuất kết quả ra file metrics_summary.csv.
4. Vẽ đồ thị tiến trình huấn luyện lưu thành training_metrics_plot.png (nếu có matplotlib).
"""

import os
import sys
import glob

def parse_simple_yaml(path):
    """Đọc file yaml đơn giản không phụ thuộc thư viện ngoài."""
    data = {}
    if not os.path.exists(path):
        return data
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' in line and not line.startswith('#'):
                k, v = line.split(':', 1)
                try:
                    data[k.strip()] = float(v.strip())
                except ValueError:
                    data[k.strip()] = v.strip()
    return data

def summarize(log_dir_path):
    # Tìm thư mục val/evaluation
    candidates = glob.glob(os.path.join(log_dir_path, "**/val/evaluation"), recursive=True)
    if not candidates:
        print(f"[LỖI] Không tìm thấy thư mục 'val/evaluation' trong '{log_dir_path}'!")
        return

    eval_dir = candidates[0]
    epoch_dirs = sorted([d for d in os.listdir(eval_dir) if d.startswith("epoch-")])
    if not epoch_dirs:
        print(f"[LỖI] Không có dữ liệu epoch nào trong '{eval_dir}'!")
        return

    print(f"\n" + "="*88)
    print(f"               BÁO CÁO TỔNG HỢP KẾT QUẢ HUẤN LUYỆN RPD (PhenoBench)")
    print(f" Thư mục log: {eval_dir}")
    print(f" Số lần đánh giá (eval points): {len(epoch_dirs)}")
    print("="*88)

    rows = []
    header = [
        "Epoch", "mIoU (%)", "IoU_BG (%)", "IoU_Crop (%)", "IoU_Weed (%)",
        "mF1 (%)", "mPrecision (%)", "mRecall (%)"
    ]

    print(f"{'Epoch':<7} | {'mIoU':<9} | {'IoU_BG':<9} | {'IoU_Crop':<10} | {'IoU_Weed':<10} | {'mF1':<9} | {'mPrec':<9} | {'mRecall':<9}")
    print("-" * 88)

    best_epoch = None
    best_miou = -1.0

    for ep_name in epoch_dirs:
        ep_path = os.path.join(eval_dir, ep_name)
        ep_num = int(ep_name.replace("epoch-", ""))

        iou = parse_simple_yaml(os.path.join(ep_path, "IoU.yaml"))
        f1 = parse_simple_yaml(os.path.join(ep_path, "F1.yaml"))
        prec = parse_simple_yaml(os.path.join(ep_path, "Precision.yaml"))
        rec = parse_simple_yaml(os.path.join(ep_path, "Recall.yaml"))

        miou = float(iou.get("mIoU", 0.0)) * 100
        iou_bg = float(iou.get("class_0", 0.0)) * 100
        iou_crop = float(iou.get("class_1", 0.0)) * 100
        iou_weed = float(iou.get("class_2", 0.0)) * 100

        mf1 = float(f1.get("mF1", 0.0)) * 100
        mprec = float(prec.get("mPrecision", 0.0)) * 100
        mrec = float(rec.get("mRecall", 0.0)) * 100

        if miou > best_miou:
            best_miou = miou
            best_epoch = ep_num

        row = [ep_num, miou, iou_bg, iou_crop, iou_weed, mf1, mprec, mrec]
        rows.append(row)

        print(f"{ep_num:<7} | {miou:<9.2f} | {iou_bg:<9.2f} | {iou_crop:<10.2f} | {iou_weed:<10.2f} | {mf1:<9.2f} | {mprec:<9.2f} | {mrec:<9.2f}")

    print("=" * 88)

    # Xuất ra file CSV
    csv_file = os.path.join(os.path.dirname(eval_dir), "metrics_summary.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join([f"{x:.2f}" if isinstance(x, float) else str(x) for x in r]) + "\n")

    print(f"\n[XUẤT FILE CSV] Đã lưu bảng kết quả chi tiết tại:")
    print(f" -> {csv_file}")

    # Tìm thông tin best epoch
    best_row = [r for r in rows if r[0] == best_epoch][0]
    print(f"\n" + "*"*88)
    print(f" ⭐ EPOCH ĐẠT KẾT QUẢ TỐT NHẤT: Epoch {best_epoch}")
    print(f"    - mIoU:         {best_row[1]:.2f}%")
    print(f"    - IoU Đất (BG): {best_row[2]:.2f}%")
    print(f"    - IoU Cây trồng:{best_row[3]:.2f}%")
    print(f"    - IoU Cỏ dại:   {best_row[4]:.2f}%")
    print(f"    - mF1-Score:    {best_row[5]:.2f}%")
    print(f"    - mPrecision:   {best_row[6]:.2f}%")
    print(f"    - mRecall:      {best_row[7]:.2f}%")
    print("*"*88)

    # Thử vẽ đồ thị nếu có matplotlib
    try:
        import matplotlib.pyplot as plt

        eps = [r[0] for r in rows]
        mious = [r[1] for r in rows]
        iou_crops = [r[3] for r in rows]
        iou_weeds = [r[4] for r in rows]
        mf1s = [r[5] for r in rows]

        plt.figure(figsize=(10, 6))
        plt.plot(eps, mious, 'b-o', label='mIoU (Trung bình)', linewidth=2)
        plt.plot(eps, iou_crops, 'g--s', label='IoU Cây trồng (Crop)', linewidth=1.5)
        plt.plot(eps, iou_weeds, 'r-.^', label='IoU Cỏ dại (Weed)', linewidth=1.5)
        plt.plot(eps, mf1s, 'm:', label='mF1 Score', linewidth=1.5)

        plt.axvline(x=best_epoch, color='gold', linestyle='--', label=f'Best Epoch ({best_epoch})')
        plt.title('Tiến trình Huấn luyện RPDNet trên PhenoBench', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Phần trăm (%)', fontsize=12)
        plt.ylim(0, 105)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right', fontsize=10)
        plt.tight_layout()

        plot_file = os.path.join(os.path.dirname(eval_dir), "training_metrics_plot.png")
        plt.savefig(plot_file, dpi=300)
        print(f"\n[VẼ ĐỒ THỊ] Đã xuất biểu đồ tiến trình huấn luyện tại:")
        print(f" -> {plot_file}")
    except ImportError:
        pass

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "./result/log_dir"
    summarize(target_dir)
