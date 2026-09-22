#!/usr/bin/env bash
# ==============================================================================
# One-Touch Sanity & Dry-Run Test for RPD-Net Baseline (B0)
# Checks: Dependencies, Forward pass, Loss, Backward, Convert, Checkpoint
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python test_dry_run.py
