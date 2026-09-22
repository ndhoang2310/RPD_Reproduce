# Automated Compute Profiling Summary

- **GPU**: NVIDIA GeForce RTX 4090 (23.51 GiB)
- **PyTorch / CUDA**: 2.5.1+cu121 / CUDA 12.1

## 1. Measured Steady-State Speed

- **Mean sec / iteration**: `0.2898 s`
- **Mean sec / epoch**: `125.33 s` (min: `121.45 s`, max: `156.48 s`)
- **Throughput**: `28.72 epochs/hr` (`13.8 samples/s`)

## 2. Measured Memory Footprint

- **Peak Allocated**: `12.196 GiB`
- **Peak Reserved**: `12.697 GiB`
- **VRAM Utilization on 24GB**: `51.9%`
- **Recommended Min VRAM (+20%)**: `14.64 GiB`

## 3. Full B0 Runtime & GPU-Hour Projections (4096 Epochs Upper-Bound)

- **Pure Training Time**: `142.60 hrs`
- **Validation Overhead (20 events)**: `0.19 hrs`
- **Safety Buffer (+15%)**: `21.42 hrs`
- **Total Full B0 Runtime**: `164.21 hrs`
- **Estimated B0 GPU-Hours (1x GPU)**: `164.21 GPU-hours`
- **Time per 200-epoch block**: `8.02 hrs`

## 4. Projected Total Project Compute Budget (B0 + B1 + P1 + Rerun buffer)

- **Total Planning Budget (Upper Bound)**: `582.94 GPU-hours`
- **Estimated Practical Budget (with Paper Early Stopping @ 400-800 epochs)**: `~120 - 150 GPU-hours`
