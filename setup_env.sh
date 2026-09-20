#!/usr/bin/env bash
set -e

echo "================================================================="
echo "   RPD-Net: Cài đặt Môi trường Huấn luyện (RTX 4090 / Linux Server)"
echo "================================================================="

# 1. Kiểm tra GPU
if command -v nvidia-smi &> /dev/null; then
    echo "[GPU Check] Đã phát hiện NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "[CẢNH BÁO] Không tìm thấy lệnh nvidia-smi! Vui lòng đảm bảo driver NVIDIA đã được cài."
fi

# 2. Tạo môi trường ảo (venv hoặc conda)
ENV_NAME="rpd_env"

if command -v conda &> /dev/null; then
    echo -e "\n[1/3] Phát hiện Conda. Đang tạo conda environment: ${ENV_NAME} (Python 3.10)..."
    conda create -y -n ${ENV_NAME} python=3.10
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ${ENV_NAME}
else
    echo -e "\n[1/3] Đang tạo Python Virtual Environment: ${ENV_NAME}..."
    python3 -m venv ${ENV_NAME}
    source ${ENV_NAME}/bin/activate
fi

# 3. Cài đặt PyTorch hỗ trợ CUDA (cho RTX 4090)
echo -e "\n[2/3] Đang cài đặt PyTorch với CUDA 12.1..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Cài đặt các thư viện phụ thuộc
echo -e "\n[3/3] Đang cài đặt các thư viện từ requirements.txt..."
pip install -r requirements.txt

echo -e "\n================================================================="
echo " ✅ CÀI ĐẶT THÀNH CÔNG!"
echo " Để bắt đầu sử dụng, hãy kích hoạt môi trường:"
if command -v conda &> /dev/null; then
    echo "    conda activate ${ENV_NAME}"
else
    echo "    source ${ENV_NAME}/bin/activate"
fi
echo "================================================================="
