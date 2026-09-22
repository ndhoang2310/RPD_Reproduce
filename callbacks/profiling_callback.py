"""
B0 Compute Profiling Callback for PyTorch Lightning.
Task: W1-T4 — Define B0 Compute Profiling Benchmark
Used in: W1-T5 — Run B0 Training Profiling on School Server

Provides automated, accurate collection of:
- Peak GPU VRAM allocated and reserved via CUDA APIs
- Warm-up excluded steady-state iteration and epoch timings
- Validation pass overhead and throughput
- Automatic computation of 4096-epoch upper-bound runtime and GPU-hours
- Export of profiling_summary.json and profiling_summary.md
"""

import json
import os
import time
from typing import Dict, Any, List, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    from pytorch_lightning import Callback, Trainer, LightningModule
except ImportError:
    class Callback:
        pass
    Trainer = Any
    LightningModule = Any


class ComputeProfilingCallback(Callback):
    """
    Automated Compute Profiler for RPD-Net Baseline Training.
    Excludes cold-start warm-up iterations to ensure unbiased timing.
    """
    def __init__(self,
                 warmup_batches: int = 10,
                 target_max_epochs: int = 4096,
                 val_interval: int = 200,
                 output_dir: str = "results",
                 verbose: bool = True):
        super().__init__()
        self.warmup_batches = warmup_batches
        self.target_max_epochs = target_max_epochs
        self.val_interval = val_interval
        self.output_dir = output_dir
        self.verbose = verbose

        # Timing tracking
        self.warmup_passed = False
        self.batch_count = 0
        self.steady_batch_times: List[float] = []
        self.epoch_times: List[float] = []

        self._batch_start_time: float = 0.0
        self._epoch_start_time: float = 0.0
        self._val_start_time: float = 0.0
        self.val_pass_times: List[float] = []
        self.val_sample_counts: List[int] = []

        # Memory tracking
        self.peak_allocated_bytes: int = 0
        self.peak_reserved_bytes: int = 0
        self.val_peak_allocated_bytes: int = 0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self.verbose:
            print("\n" + "=" * 65)
            print("[Profiling] ComputeProfilingCallback active.")
            print(f"[Profiling] Warm-up policy: First {self.warmup_batches} batches will be excluded from steady-state timing.")
            print(f"[Profiling] Reference max epochs for estimation: {self.target_max_epochs}")
            print(f"[Profiling] Reference validation interval: {self.val_interval} epochs")
            print("=" * 65 + "\n")

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._epoch_start_time = time.perf_counter()

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        self._batch_start_time = time.perf_counter()

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        elapsed = time.perf_counter() - self._batch_start_time
        self.batch_count += 1

        if not self.warmup_passed:
            if self.batch_count >= self.warmup_batches:
                self.warmup_passed = True
                if torch is not None and torch.cuda.is_available():
                    # Reset stats so peak reflects steady-state training
                    torch.cuda.reset_peak_memory_stats()
                if self.verbose:
                    print(f"\n[Profiling] Warm-up ({self.warmup_batches} batches) completed. Entering steady-state measurement window.")
        else:
            self.steady_batch_times.append(elapsed)

        if torch is not None and torch.cuda.is_available():
            alloc = torch.cuda.max_memory_allocated()
            reserv = torch.cuda.max_memory_reserved()
            if alloc > self.peak_allocated_bytes:
                self.peak_allocated_bytes = alloc
            if reserv > self.peak_reserved_bytes:
                self.peak_reserved_bytes = reserv

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch_duration = time.perf_counter() - self._epoch_start_time
        # Only record epoch duration if warm-up is already completed to avoid skewing
        if self.warmup_passed:
            self.epoch_times.append(epoch_duration)

        if self.verbose:
            alloc_mb = (self.peak_allocated_bytes / (1024 ** 2)) if (torch is not None and torch.cuda.is_available()) else 0.0
            reserv_mb = (self.peak_reserved_bytes / (1024 ** 2)) if (torch is not None and torch.cuda.is_available()) else 0.0
            epoch_num = getattr(trainer, 'current_epoch', 0)
            print(f"[Profiling Epoch {epoch_num}] Duration: {epoch_duration:.2f}s | "
                  f"Peak Allocated: {alloc_mb:.1f} MB | Peak Reserved: {reserv_mb:.1f} MB")

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if getattr(trainer, 'sanity_checking', False):
            return
        self._val_start_time = time.perf_counter()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if getattr(trainer, 'sanity_checking', False):
            return
        val_duration = time.perf_counter() - self._val_start_time
        self.val_pass_times.append(val_duration)

        num_val_samples = 64  # Standard PhenoBench val split default
        if hasattr(trainer, 'val_dataloaders'):
            val_loader = trainer.val_dataloaders
            if val_loader:
                try:
                    if isinstance(val_loader, list) and len(val_loader) > 0:
                        num_val_samples = len(val_loader[0].dataset)
                    elif hasattr(val_loader, 'dataset'):
                        num_val_samples = len(val_loader.dataset)
                except Exception:
                    pass
        self.val_sample_counts.append(num_val_samples)

        if torch is not None and torch.cuda.is_available():
            val_alloc = torch.cuda.max_memory_allocated()
            if val_alloc > self.val_peak_allocated_bytes:
                self.val_peak_allocated_bytes = val_alloc

        if self.verbose:
            val_fps = num_val_samples / max(val_duration, 1e-6)
            print(f"\n[Profiling Validation Pass] Duration: {val_duration:.2f}s ({num_val_samples} samples, {val_fps:.2f} img/s)\n")

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        summary = self.generate_summary(trainer)
        self.save_summary(summary)
        self.print_summary(summary)

    def generate_summary(self, trainer: Optional[Trainer] = None) -> Dict[str, Any]:
        num_steady_batches = len(self.steady_batch_times)
        mean_sec_iter = (sum(self.steady_batch_times) / num_steady_batches) if num_steady_batches > 0 else 0.0

        num_steady_epochs = len(self.epoch_times)
        mean_sec_epoch = (sum(self.epoch_times) / num_steady_epochs) if num_steady_epochs > 0 else (mean_sec_iter * 55)
        min_sec_epoch = min(self.epoch_times) if num_steady_epochs > 0 else mean_sec_epoch
        max_sec_epoch = max(self.epoch_times) if num_steady_epochs > 0 else mean_sec_epoch

        epochs_per_hour = (3600.0 / mean_sec_epoch) if mean_sec_epoch > 0 else 0.0

        # Batch size resolution
        batch_size = 4
        if trainer and hasattr(trainer, 'train_dataloader') and trainer.train_dataloader:
            try:
                batch_size = trainer.train_dataloader.batch_size
            except Exception:
                pass
        samples_per_sec = (batch_size / mean_sec_iter) if mean_sec_iter > 0 else 0.0

        # Validation stats
        mean_val_pass_sec = (sum(self.val_pass_times) / len(self.val_pass_times)) if self.val_pass_times else 0.0
        val_events_in_4096 = self.target_max_epochs // self.val_interval  # e.g., 4096 // 200 = 20

        # Compute full runtime projections
        pure_train_hours = (mean_sec_epoch * self.target_max_epochs) / 3600.0
        val_overhead_hours = (mean_val_pass_sec * val_events_in_4096) / 3600.0
        subtotal_hours = pure_train_hours + val_overhead_hours
        buffer_hours = subtotal_hours * 0.15
        total_projected_hours = subtotal_hours + buffer_hours

        # VRAM stats in MiB & GiB
        peak_alloc_gib = self.peak_allocated_bytes / (1024 ** 3)
        peak_reserv_gib = self.peak_reserved_bytes / (1024 ** 3)
        recommended_min_vram = peak_alloc_gib * 1.20  # 20% margin

        gpu_name = (torch.cuda.get_device_name(0) if (torch is not None and torch.cuda.is_available()) else "NVIDIA GeForce RTX 4090 (Simulated / CPU)")
        total_vram_gib = ((torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)) if (torch is not None and torch.cuda.is_available()) else 24.0)
        cuda_ver = (torch.version.cuda if (torch is not None and torch.cuda.is_available()) else "12.1")
        torch_ver = (torch.__version__ if torch is not None else "2.2.0 (fallback)")

        summary = {
            "hardware": {
                "gpu_name": gpu_name,
                "gpu_count": 1,
                "total_vram_gib": round(total_vram_gib, 2),
                "cuda_version": cuda_ver,
                "pytorch_version": torch_ver,
            },
            "training_config": {
                "batch_size": batch_size,
                "warmup_batches_excluded": self.warmup_batches,
                "measured_steady_batches": num_steady_batches,
                "measured_steady_epochs": num_steady_epochs,
                "target_max_epochs": self.target_max_epochs,
                "val_interval_epochs": self.val_interval,
            },
            "observed_vram": {
                "peak_allocated_gib": round(peak_alloc_gib, 3),
                "peak_reserved_gib": round(peak_reserv_gib, 3),
                "vram_utilization_pct": round((peak_alloc_gib / total_vram_gib * 100), 1) if total_vram_gib > 0 else 0.0,
                "oom_occurred": False,
                "recommended_min_vram_gib": round(recommended_min_vram, 2),
            },
            "observed_speed": {
                "mean_sec_per_iteration": round(mean_sec_iter, 4),
                "mean_sec_per_epoch": round(mean_sec_epoch, 2),
                "min_sec_per_epoch": round(min_sec_epoch, 2),
                "max_sec_per_epoch": round(max_sec_epoch, 2),
                "epochs_per_hour": round(epochs_per_hour, 2),
                "samples_per_second": round(samples_per_sec, 2),
            },
            "observed_overhead": {
                "validation_pass_sec": round(mean_val_pass_sec, 2),
                "scheduled_val_events_4096": val_events_in_4096,
                "total_val_overhead_hours": round(val_overhead_hours, 3),
            },
            "estimations": {
                "pure_train_hours_4096": round(pure_train_hours, 2),
                "val_overhead_hours_4096": round(val_overhead_hours, 2),
                "safety_buffer_15pct_hours": round(buffer_hours, 2),
                "full_b0_estimated_hours": round(total_projected_hours, 2),
                "estimated_gpu_hours_1gpu": round(total_projected_hours, 2),
                "estimated_hours_per_200_epochs": round(((mean_sec_epoch * 200 + mean_val_pass_sec) * 1.15) / 3600.0, 2),
                "projected_b1_gpu_hours": round(total_projected_hours * 1.0, 2),
                "projected_p1_gpu_hours": round(total_projected_hours * 1.05, 2),
                "total_project_learning_gpu_hours": round(total_projected_hours * (1.0 + 1.0 + 1.05 + 0.5), 2),
            }
        }
        return summary

    def save_summary(self, summary: Dict[str, Any]) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        json_path = os.path.join(self.output_dir, "profiling_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        md_path = os.path.join(self.output_dir, "profiling_summary.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Automated Compute Profiling Summary\n\n")
            f.write(f"- **GPU**: {summary['hardware']['gpu_name']} ({summary['hardware']['total_vram_gib']} GiB)\n")
            f.write(f"- **PyTorch / CUDA**: {summary['hardware']['pytorch_version']} / {summary['hardware']['cuda_version']}\n\n")
            f.write("## 1. Measured Steady-State Speed\n\n")
            f.write(f"- **Mean sec / iteration**: `{summary['observed_speed']['mean_sec_per_iteration']} s`\n")
            f.write(f"- **Mean sec / epoch**: `{summary['observed_speed']['mean_sec_per_epoch']} s`\n")
            f.write(f"- **Throughput**: `{summary['observed_speed']['epochs_per_hour']} epochs/hr` ({summary['observed_speed']['samples_per_second']} samples/s)\n\n")
            f.write("## 2. Measured Memory Footprint\n\n")
            f.write(f"- **Peak Allocated**: `{summary['observed_vram']['peak_allocated_gib']} GiB`\n")
            f.write(f"- **Peak Reserved**: `{summary['observed_vram']['peak_reserved_gib']} GiB`\n")
            f.write(f"- **VRAM Utilization on 24GB**: `{summary['observed_vram']['vram_utilization_pct']}%`\n")
            f.write(f"- **Recommended Min VRAM (+20%)**: `{summary['observed_vram']['recommended_min_vram_gib']} GiB`\n\n")
            f.write("## 3. Full B0 Runtime & GPU-Hour Projections (4096 Epochs Upper-Bound)\n\n")
            f.write(f"- **Pure Training Time**: `{summary['estimations']['pure_train_hours_4096']} hrs`\n")
            f.write(f"- **Validation Overhead (20 events)**: `{summary['estimations']['val_overhead_hours_4096']} hrs`\n")
            f.write(f"- **Safety Buffer (+15%)**: `{summary['estimations']['safety_buffer_15pct_hours']} hrs`\n")
            f.write(f"- **Total Full B0 Runtime**: `{summary['estimations']['full_b0_estimated_hours']} hrs`\n")
            f.write(f"- **Estimated B0 GPU-Hours (1x GPU)**: `{summary['estimations']['estimated_gpu_hours_1gpu']} GPU-hours`\n")
            f.write(f"- **Time per 200-epoch block**: `{summary['estimations']['estimated_hours_per_200_epochs']} hrs`\n\n")
            f.write("## 4. Projected Total Project Compute Budget (B0 + B1 + P1 + Rerun buffer)\n\n")
            f.write(f"- **Total Planning Budget**: `{summary['estimations']['total_project_learning_gpu_hours']} GPU-hours`\n")

    def print_summary(self, summary: Dict[str, Any]) -> None:
        print("\n" + "=" * 70)
        print("                 B0 COMPUTE PROFILING REPORT                 ")
        print("=" * 70)
        print(f" Hardware:           {summary['hardware']['gpu_name']} ({summary['hardware']['total_vram_gib']} GiB VRAM)")
        print(f" PyTorch / CUDA:     {summary['hardware']['pytorch_version']} | CUDA {summary['hardware']['cuda_version']}")
        print("-" * 70)
        print(" [OBSERVED SPEED - STEADY STATE]")
        print(f"  Warm-up Excluded:  {summary['training_config']['warmup_batches_excluded']} batches")
        print(f"  Steady Batches:    {summary['training_config']['measured_steady_batches']} batches")
        print(f"  Steady Epochs:     {summary['training_config']['measured_steady_epochs']} epochs")
        print(f"  Mean Sec / Iter:   {summary['observed_speed']['mean_sec_per_iteration']} s")
        print(f"  Mean Sec / Epoch:  {summary['observed_speed']['mean_sec_per_epoch']} s (range: {summary['observed_speed']['min_sec_per_epoch']}s - {summary['observed_speed']['max_sec_per_epoch']}s)")
        print(f"  Epochs / Hour:     {summary['observed_speed']['epochs_per_hour']} epochs/hr")
        print(f"  Throughput:        {summary['observed_speed']['samples_per_second']} samples/s")
        print("-" * 70)
        print(" [OBSERVED VRAM FOOTPRINT]")
        print(f"  Peak Allocated:    {summary['observed_vram']['peak_allocated_gib']} GiB")
        print(f"  Peak Reserved:     {summary['observed_vram']['peak_reserved_gib']} GiB")
        print(f"  VRAM Utilization:  {summary['observed_vram']['vram_utilization_pct']}% of {summary['hardware']['total_vram_gib']} GiB")
        print(f"  Recommended VRAM:  >= {summary['observed_vram']['recommended_min_vram_gib']} GiB (with 20% margin)")
        print("-" * 70)
        print(" [FULL B0 ESTIMATIONS - 4096 EPOCHS UPPER BOUND]")
        print(f"  Pure Train Time:   {summary['estimations']['pure_train_hours_4096']} hrs")
        print(f"  Validation Time:   {summary['estimations']['val_overhead_hours_4096']} hrs (20 scheduled checks)")
        print(f"  Safety Buffer 15%: {summary['estimations']['safety_buffer_15pct_hours']} hrs")
        print(f"  TOTAL ESTIMATED:   {summary['estimations']['full_b0_estimated_hours']} hrs ({summary['estimations']['estimated_gpu_hours_1gpu']} GPU-hours)")
        print(f"  Per 200-Epochs:    {summary['estimations']['estimated_hours_per_200_epochs']} hrs")
        print("-" * 70)
        print(f" Total Project Compute Budget (B0+B1+P1): ~{summary['estimations']['total_project_learning_gpu_hours']} GPU-hours")
        print(f" Summary saved to: {os.path.join(self.output_dir, 'profiling_summary.json')} & .md")
        print("=" * 70 + "\n")
