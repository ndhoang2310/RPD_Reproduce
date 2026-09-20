#!/usr/bin/env bash
# ==============================================================================
# Script Huấn luyện RPDNet trên Server SSH (RTX 4090)
# Cách dùng:
#   1. Chạy trực tiếp:
#      bash run_train.sh /path/to/PhenoBench
#   2. Chạy ngầm bằng nohup (tắt terminal laptop vẫn chạy tiếp):
#      nohup bash run_train.sh /path/to/PhenoBench > train.log 2>&1 &
# ==============================================================================

set -e

# Đặt cấu hình chống phân mảnh VRAM
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONWARNINGS="ignore"

# Đường dẫn dataset (ưu tiên tham số thứ 1, nếu không có thì lấy mặc định)
DATASET_DIR=${1:-"/kaggle/input/datasets/ndhoang2310/phenobench-dataset/PhenoBench"}
CONFIG_FILE=${2:-"./config/config_paper.yaml"}
EXPORT_DIR=${3:-"./log_dir"}

echo "================================================================="
echo "   RPD-Net: Bắt đầu Huấn luyện (Semantic Segmentation)"
echo "================================================================="
echo " - Dataset:    ${DATASET_DIR}"
echo " - Config:     ${CONFIG_FILE}"
echo " - Export dir: ${EXPORT_DIR}"

# Kiểm tra đường dẫn dataset
if [ ! -d "${DATASET_DIR}/train/images" ]; then
    echo -e "\n[CẢNH BÁO] Không tìm thấy thư mục '${DATASET_DIR}/train/images'!"
    echo "Hãy truyền đường dẫn dataset chính xác: bash run_train.sh /path/to/PhenoBench"
    exit 1
fi

# Tự động tìm checkpoint để resume nếu có
RESUME_ARG=""
LAST_CKPT=$(find "${EXPORT_DIR}" -name "last.ckpt" 2>/dev/null | sort | tail -n 1 || true)
if [ -n "${LAST_CKPT}" ] && [ -f "${LAST_CKPT}" ]; then
    echo " -> Phát hiện checkpoint trước đó: ${LAST_CKPT}"
    echo " -> Tự động RESUME quá trình huấn luyện!"
    RESUME_ARG="--ckpt_path ${LAST_CKPT} --resume"
else
    echo " -> Huấn luyện mới từ đầu (Train from scratch)."
fi

# Chạy huấn luyện
python multi_metric_train.py \
    --config "${CONFIG_FILE}" \
    --export_dir "${EXPORT_DIR}" \
    --dataset_dir "${DATASET_DIR}" \
    ${RESUME_ARG}

echo -e "\n[HOÀN TẤT] Huấn luyện kết thúc! Chạy 'python summarize_results.py ${EXPORT_DIR}' để xem tổng kết."
