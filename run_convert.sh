#!/usr/bin/env bash
# ==============================================================================
# Script Convert & Deploy Model (Re-parameterization)
# Chuyển đổi mô hình đa nhánh RPD sang 1 nhánh duy nhất để inference siêu tốc
# Cách dùng:
#   bash run_convert.sh [path-to-best-checkpoint] [dataset-dir]
# ==============================================================================

set -e

EXPORT_DIR="./log_dir"
TARGET_CKPT=$1
DATASET_DIR=${2:-""}

# Tự động tìm checkpoint nếu người dùng không truyền vào
if [ -z "${TARGET_CKPT}" ]; then
    echo "Đang tự động tìm checkpoint mIoU cao nhất trong ${EXPORT_DIR}..."
    TARGET_CKPT=$(find "${EXPORT_DIR}" -name "*val_mIoU*.ckpt" 2>/dev/null | sort | tail -n 1 || true)
    if [ -z "${TARGET_CKPT}" ]; then
        TARGET_CKPT=$(find "${EXPORT_DIR}" -name "last.ckpt" 2>/dev/null | sort | tail -n 1 || true)
    fi
fi

if [ -z "${TARGET_CKPT}" ] || [ ! -f "${TARGET_CKPT}" ]; then
    echo "[LỖI] Không tìm thấy file checkpoint (.ckpt) nào để convert!"
    echo "Cách dùng: bash run_convert.sh /path/to/model.ckpt"
    exit 1
fi

echo "================================================================="
echo "   RPD-Net: Convert & Deploy (Re-parameterization)"
echo "================================================================="
echo " - Checkpoint nguồn: ${TARGET_CKPT}"

DATASET_ARG=""
if [ -n "${DATASET_DIR}" ]; then
    DATASET_ARG="--dataset_dir ${DATASET_DIR}"
fi

CONVERT_DIR="${EXPORT_DIR}/convert_testing"
DEPLOY_DIR="${EXPORT_DIR}/deploy_testing"

echo -e "\n[BƯỚC 1/2] Đang chuyển đổi đa nhánh (PDC) thành cấu trúc tương đương..."
python convert_multi_test.py \
    --config ./config/config_convert.yaml \
    --ckpt_path "${TARGET_CKPT}" \
    --export_dir "${CONVERT_DIR}" \
    ${DATASET_ARG}

echo -e "\n[BƯỚC 2/2] Đang gộp trọng số thành mô hình Deploy 1 nhánh duy nhất..."
python deploy_convert_multi_test.py \
    --config ./config/config_deploy_convert.yaml \
    --convert_ckpt_path "${TARGET_CKPT}" \
    --export_dir "${DEPLOY_DIR}" \
    ${DATASET_ARG}

echo -e "\n================================================================="
echo " ✅ HOÀN TẤT RE-PARAMETERIZATION!"
echo " - Mô hình triển khai đã được lưu tại: ${DEPLOY_DIR}/deploy_model.ckpt"
echo "================================================================="
