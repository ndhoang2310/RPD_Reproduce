#!/usr/bin/env bash
# ==============================================================================
# Execution Wrapper: B0 Compute Profiling on School Server (RTX 4090 24GB)
# Task: W1-T4 (Defined) & W1-T5 (Execution)
# Runs 10 epochs training + 1 validation pass with automated profiling
# ==============================================================================

set -e

# Anti-fragmentation configuration for PyTorch CUDA
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONWARNINGS="ignore"

# Profiling default arguments
DATASET_DIR="./data/PhenoBench"
CONFIG_FILE="configs/B0.yaml"
EXPORT_DIR="results/profiling"
MAX_EPOCH=10
VAL_INTERVAL=10
NUM_WORKERS=8
EXTRA_ARGS=""

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
        --max_epoch)
            MAX_EPOCH="$2"
            shift 2
            ;;
        --check_val_every_n_epoch)
            VAL_INTERVAL="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift 1
            ;;
    esac
done

echo "================================================================="
echo "   B0 Compute Profiling Runner (Task W1-T5 on School Server)     "
echo "================================================================="
echo " - Dataset Path:       ${DATASET_DIR}"
echo " - Config File:        ${CONFIG_FILE}"
echo " - Export Dir:         ${EXPORT_DIR}"
echo " - Profiling Epochs:   ${MAX_EPOCH} epochs"
echo " - Val Interval:       Every ${VAL_INTERVAL} epochs (1 val pass)"
echo " - Workers:            ${NUM_WORKERS}"
echo " - Extra Flags:        ${EXTRA_ARGS}"
echo "================================================================="

# Fallback dataset resolution
if [ ! -d "${DATASET_DIR}/train/images" ]; then
    if [ -d "../PhenoBench/train/images" ]; then
        DATASET_DIR="../PhenoBench"
        echo " -> Resolved dataset to relative path: ${DATASET_DIR}"
    fi
fi

mkdir -p "${EXPORT_DIR}/logs"
LOG_FILE="${EXPORT_DIR}/logs/profiling_run.log"

echo " -> Launching profiling run... Output is logged to: ${LOG_FILE}"

python multi_metric_train.py \
    --config "${CONFIG_FILE}" \
    --export_dir "${EXPORT_DIR}" \
    --dataset_dir "${DATASET_DIR}" \
    --max_epoch "${MAX_EPOCH}" \
    --check_val_every_n_epoch "${VAL_INTERVAL}" \
    --num_workers "${NUM_WORKERS}" \
    --profile \
    ${EXTRA_ARGS} 2>&1 | tee "${LOG_FILE}"

TRAIN_EXIT=${PIPESTATUS[0]}

if [ $TRAIN_EXIT -ne 0 ]; then
    echo -e "\n================================================================="
    echo " [ERROR] Profiling run failed with exit code $TRAIN_EXIT!"
    echo " -> Check the log file for tracebacks: ${LOG_FILE}"
    echo "================================================================="
    exit $TRAIN_EXIT
fi

echo -e "\n================================================================="
echo " [DONE] Profiling run completed successfully!"
echo " -> Profiling Summary JSON: ${EXPORT_DIR}/profiling_summary.json"
echo " -> Profiling Summary MD:   ${EXPORT_DIR}/profiling_summary.md"
echo " -> Log File:               ${LOG_FILE}"
echo " -> Now transfer these values into: experiments/B0_Compute_Profiling_Benchmark.md"
echo "================================================================="
