#!/usr/bin/env bash
# ==============================================================================
# Automated Dataset Preparation for PhenoBench
# Task: W1-T3 — Baseline B0 Training Readiness
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DATA_DIR="${SCRIPT_DIR}/data/PhenoBench"

echo "=============================================================="
echo " [PhenoBench] Chuẩn bị dữ liệu cho Baseline B0"
echo "=============================================================="

mkdir -p "${SCRIPT_DIR}/data"

# Kiểm tra nếu dataset đã tồn tại sẵn
if [ -d "${TARGET_DATA_DIR}/train/images" ] && [ -d "${TARGET_DATA_DIR}/val/images" ]; then
    echo " -> Phát hiện dataset PhenoBench đã tồn tại tại: ${TARGET_DATA_DIR}"
    NUM_TRAIN=$(ls -1 "${TARGET_DATA_DIR}/train/images"/*.png 2>/dev/null | wc -l || echo 0)
    NUM_VAL=$(ls -1 "${TARGET_DATA_DIR}/val/images"/*.png 2>/dev/null | wc -l || echo 0)
    echo "    - Tập Train: ${NUM_TRAIN} ảnh (Kỳ vọng: 1,407)"
    echo "    - Tập Val:   ${NUM_VAL} ảnh (Kỳ vọng: 772)"
    echo " -> Dữ liệu đã SẴN SÀNG! Bạn có thể bắt đầu huấn luyện ngay."
    exit 0
fi

echo "Chưa tìm thấy dataset hoàn chỉnh tại ${TARGET_DATA_DIR}."
echo "Vui lòng chọn một phương thức chuẩn bị dữ liệu bên dưới:"
echo " 1) Tải trực tiếp từ Kaggle bằng Kaggle API (Cần file kaggle.json)"
echo " 2) Giải nén từ file zip có sẵn (PhenoBench-v110.zip hoặc phenobench-dataset.zip)"
echo " 3) Tạo liên kết (symlink) từ thư mục PhenoBench có sẵn trên server"
echo " 4) Hủy thao tác"
echo ""
read -p "Nhập lựa chọn của bạn [1-4]: " CHOICE

case "$CHOICE" in
    1)
        echo -e "\n[1/3] Kiểm tra công cụ Kaggle CLI..."
        if ! command -v kaggle &> /dev/null; then
            echo "Đang cài đặt kaggle package..."
            pip install kaggle -q
        fi

        if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
            echo "[LỖI] Không tìm thấy file $HOME/.kaggle/kaggle.json!"
            echo "Hãy copy file kaggle.json vào $HOME/.kaggle/ và chạy lại script."
            exit 1
        fi
        chmod 600 "$HOME/.kaggle/kaggle.json"

        echo -e "\n[2/3] Đang tải dataset ndhoang2310/phenobench-dataset từ Kaggle..."
        cd "${SCRIPT_DIR}/data"
        kaggle datasets download -d ndhoang2310/phenobench-dataset

        echo -e "\n[3/3] Đang giải nén dữ liệu..."
        unzip -q phenobench-dataset.zip
        rm -f phenobench-dataset.zip

        # Chuẩn hóa tên thư mục nếu có thư mục lồng nhau
        if [ -d "${SCRIPT_DIR}/data/PhenoBench/PhenoBench" ]; then
            mv "${SCRIPT_DIR}/data/PhenoBench" "${SCRIPT_DIR}/data/PhenoBench_tmp"
            mv "${SCRIPT_DIR}/data/PhenoBench_tmp/PhenoBench" "${SCRIPT_DIR}/data/PhenoBench"
            rm -rf "${SCRIPT_DIR}/data/PhenoBench_tmp"
        fi
        ;;

    2)
        read -p "Nhập đường dẫn đến file .zip [mặc định: ./PhenoBench-v110.zip]: " ZIP_PATH
        ZIP_PATH=${ZIP_PATH:-"./PhenoBench-v110.zip"}
        if [ ! -f "$ZIP_PATH" ]; then
            echo "[LỖI] File không tồn tại: $ZIP_PATH"
            exit 1
        fi
        echo "Đang giải nén $ZIP_PATH vào ${SCRIPT_DIR}/data/..."
        unzip -q "$ZIP_PATH" -d "${SCRIPT_DIR}/data/"
        ;;

    3)
        read -p "Nhập đường dẫn thư mục PhenoBench trên server: " SOURCE_DIR
        if [ ! -d "$SOURCE_DIR/train/images" ]; then
            echo "[LỖI] Thư mục không hợp lệ hoặc thiếu train/images: $SOURCE_DIR"
            exit 1
        fi
        ln -sfn "$SOURCE_DIR" "${TARGET_DATA_DIR}"
        echo "Đã tạo symlink từ $SOURCE_DIR -> ${TARGET_DATA_DIR}"
        ;;

    4)
        echo "Đã hủy thao tác."
        exit 0
        ;;

    *)
        echo "Lựa chọn không hợp lệ."
        exit 1
        ;;
esac

# Kiểm tra lại tính toàn vẹn
echo -e "\n=============================================================="
echo " [Xác minh tính toàn vẹn Dữ liệu]"
echo "=============================================================="
if [ -d "${TARGET_DATA_DIR}/train/images" ] && [ -d "${TARGET_DATA_DIR}/val/images" ]; then
    NUM_TRAIN=$(ls -1 "${TARGET_DATA_DIR}/train/images"/*.png 2>/dev/null | wc -l || echo 0)
    NUM_VAL=$(ls -1 "${TARGET_DATA_DIR}/val/images"/*.png 2>/dev/null | wc -l || echo 0)
    echo "  - Thư mục: ${TARGET_DATA_DIR}"
    echo "  - Tập Train: ${NUM_TRAIN} ảnh (Kỳ vọng: 1,407)"
    echo "  - Tập Val:   ${NUM_VAL} ảnh (Kỳ vọng: 772)"
    echo " -> XÁC MINH THÀNH CÔNG! Pipeline sẵn sàng chạy."
else
    echo " [CẢNH BÁO] Chưa thấy đúng cấu trúc ${TARGET_DATA_DIR}/train/images."
    echo " Vui lòng kiểm tra lại đường dẫn."
    exit 1
fi
