#!/usr/bin/env bash
# ==============================================================================
# Execution Wrapper: Baseline B0 Training on School Server (RTX 4090 24GB)
# Task: W1-T3 — Prepare B0 Training Experiment
# ==============================================================================

set -e

# Anti-fragmentation configuration for PyTorch CUDA
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONWARNINGS="ignore"

# Defaults
DATASET_DIR="./data/PhenoBench"
CONFIG_FILE="configs/B0.yaml"
EXPORT_DIR="results"
EXTRA_ARGS=""

# Parse arguments (supports positional or flagged arguments)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset_dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --export_dir)
            EXPORT_DIR="$2"
            shift 2
            ;;
        *)
            # If not matching above, pass through as extra arg
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift 1
            ;;
    esac
done

echo "================================================================="
echo "   Baseline B0 (RPD-Net) Training Runner"
echo "================================================================="
echo " - Dataset Path: ${DATASET_DIR}"
echo " - Config File:  ${CONFIG_FILE}"
echo " - Export Dir:   ${EXPORT_DIR}"
echo " - Extra Flags:  ${EXTRA_ARGS}"

# Fallback dataset resolution
if [ ! -d "${DATASET_DIR}/train/images" ]; then
    if [ -d "../PhenoBench/train/images" ]; then
        DATASET_DIR="../PhenoBench"
        echo " -> Resolved dataset to relative path: ${DATASET_DIR}"
    fi
fi

# Automatic checkpoint resume detection
RESUME_ARG=""
LAST_CKPT=$(find "${EXPORT_DIR}" -name "last.ckpt" 2>/dev/null | sort | tail -n 1 || true)
if [ -n "${LAST_CKPT}" ] && [ -f "${LAST_CKPT}" ]; then
    echo " -> Found existing checkpoint: ${LAST_CKPT}"
    echo " -> Automatically RESUMING training session!"
    RESUME_ARG="--ckpt_path ${LAST_CKPT} --resume"
else
    echo " -> Starting clean run from scratch."
fi

mkdir -p "${EXPORT_DIR}/logs"

# Execute training
python multi_metric_train.py \
    --config "${CONFIG_FILE}" \
    --export_dir "${EXPORT_DIR}" \
    --dataset_dir "${DATASET_DIR}" \
    ${RESUME_ARG} \
    ${EXTRA_ARGS}

echo -e "\n[DONE] Training session finished!"
echo "Check summary with: python summarize_results.py ${EXPORT_DIR}"
