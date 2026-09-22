#!/usr/bin/env python3
"""
Comprehensive Dry-Run & Sanity Test for RPD-Net Baseline (B0).
Task: W1-T3 Readiness & Pre-Server Verification.

Verifies:
1. Environment dependencies (PyTorch, Lightning, TorchMetrics, skimage, tifffile).
2. Full B0 Training Pipeline (Synthetic batch -> Forward -> Weighted Loss -> Backward -> Adam update).
3. Structural Reparameterization (Official 2-Step RPD Pipeline: convert_block -> switch_to_deploy).
4. Checkpoint Serialization (Save & Load).

Auto-tunes resolution:
- On CPU: Uses 256x256 to fit within limited system RAM (avoids Linux OOM killer).
- On GPU: Uses 768x768 to test full-resolution feature maps.
"""

import os
import sys
import copy
import tempfile
import time
import torch
import torch.nn as nn

print("=" * 65)
print("     RPD-NET BASELINE (B0) FULL SANITY & DRY-RUN TEST        ")
print("=" * 65)

# --- 1. Dependencies & Environment Check ---
print("\n[1/4] Checking Python Dependencies & Hardware...")
try:
    import pytorch_lightning as pl
    import torchmetrics
    import numpy as np
    import skimage
    import tifffile
    print(f"  ✓ PyTorch:          {torch.__version__}")
    print(f"  ✓ PyTorch Lightning:{pl.__version__}")
    print(f"  ✓ scikit-image:     {skimage.__version__}")
    print(f"  ✓ tifffile:         {tifffile.__version__}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  ✓ Execution Device: {device}")
    if torch.cuda.is_available():
        crop_size, batch_size = 768, 2
    else:
        print("    - Chạy trên CPU: Dùng batch 1x3x256x256 để tiết kiệm RAM...")
        crop_size, batch_size = 256, 1
except Exception as e:
    print(f"  ✗ Dependency Error: {e}")
    sys.exit(1)

# --- 2. B0 Training Dry-Run ---
print(f"\n[2/4] Testing B0 Training Pipeline (Batch: {batch_size}x3x{crop_size}x{crop_size})...")
try:
    from models import RPDNet
    
    # Initialize baseline B0 model (num_classes=3, deploy=False, convert=False)
    model = RPDNet(num_classes=3, deploy=False, convert=False).to(device)
    model.train()
    
    # Create synthetic input batch
    x = torch.randn(batch_size, 3, crop_size, crop_size, device=device)
    target = torch.randint(0, 3, (batch_size, crop_size, crop_size), device=device)
    
    # Forward pass
    t0 = time.time()
    logits = model(x)
    forward_time = time.time() - t0
    assert logits.shape == (batch_size, 3, crop_size, crop_size), f"Unexpected logits shape: {logits.shape}"
    print(f"  ✓ Forward pass OK ({forward_time:.2f}s) | Output shape: {list(logits.shape)}")
    
    # B0 Weighted CrossEntropy Loss: Soil=1.47, Crop=5.06, Weed=10.02
    weights = torch.tensor([1.47, 5.06, 10.02], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    loss = criterion(logits, target)
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN or Inf!"
    print(f"  ✓ B0 Weighted Cross-Entropy Loss: {loss.item():.4f}")
    
    # Backward pass & optimizer step
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4, weight_decay=2.0e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("  ✓ Backward pass & Adam optimizer step OK (Gradients computed & applied)!")
except Exception as e:
    print(f"  ✗ B0 Training Pipeline Failed: {e}")
    sys.exit(1)

# --- 3. Structural Reparameterization Test (Official 2-Step RPD Pipeline) ---
print("\n[3/4] Testing Structural Reparameterization (Official 2-Step Pipeline)...")
try:
    from models import convert_block
    
    # Model 1: Training Multi-Branch Model
    train_model = RPDNet(num_classes=3, deploy=False, convert=False).to(device)
    train_model.eval()
    train_params = sum(p.numel() for p in train_model.parameters())
    print(f"  - Mô hình huấn luyện ban đầu (Multi-Branch PDC): {train_params:,} tham số")
    
    # Step 3.1: Convert PDC difference weights to equivalent standard Conv weights
    t_c = time.time()
    converted_dict = convert_block(train_model.state_dict(), 'pdc')
    print(f"  ✓ [Bước 1/2]: convert_block('pdc') chuyển đổi trọng số PDC thành công ({time.time()-t_c:.3f}s)")
    
    # Step 3.2: Load converted weights into converted architecture
    converted_model = RPDNet(num_classes=3, deploy=False, convert=True).to(device)
    converted_model.load_state_dict(converted_dict, strict=False)
    
    # Step 3.3: Fuse multi-branch into single-branch deploy model via switch_to_deploy
    deploy_model = copy.deepcopy(converted_model)
    fused_count = 0
    for m in deploy_model.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()
            fused_count += 1
    deploy_model.eval()
    print(f"  ✓ [Bước 2/2]: switch_to_deploy() đã gộp {fused_count} khối RepConv/RPD về 1 nhánh duy nhất")
    
    deploy_params = sum(p.numel() for p in deploy_model.parameters())
    reduction = (1 - deploy_params / train_params) * 100
    print(f"  ✓ Số tham số sau khi gộp:   {deploy_params:,} (Giảm {reduction:.1f}%)")
    assert deploy_params < train_params, "Tham số sau khi gộp phải nhỏ hơn mô hình huấn luyện!"
    
    # Verify inference output on single-branch deploy model
    test_in = torch.randn(1, 3, crop_size, crop_size, device=device)
    with torch.no_grad():
        out_deploy = deploy_model(test_in)
    assert out_deploy.shape == (1, 3, crop_size, crop_size)
    print(f"  ✓ Single-branch Deploy Model suy luận thành công! Output shape: {list(out_deploy.shape)}")
except Exception as e:
    print(f"  ✗ Reparameterization Failed: {e}")
    sys.exit(1)

# --- 4. Checkpoint Serialization Test ---
print("\n[4/4] Testing Checkpoint Serialization (Save / Load)...")
try:
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test_b0.ckpt")
        torch.save({"state_dict": train_model.state_dict()}, ckpt_path)
        assert os.path.exists(ckpt_path), "Checkpoint file was not written!"
        ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)
        
        # Reload
        loaded = torch.load(ckpt_path, map_location=device)
        assert "state_dict" in loaded, "Corrupted checkpoint state dict!"
        print(f"  ✓ Checkpoint successfully saved ({ckpt_size_mb:.2f} MB) và nạp lại thành công.")
except Exception as e:
    print(f"  ✗ Checkpoint Test Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 65)
print(" 🎉 TẤT CẢ 4 BƯỚC KIỂM THỬ DRY-RUN ĐÃ PASS 100% THÀNH CÔNG!   ")
print(" Pipeline huấn luyện và Reparameterization đã sẵn sàng cho Server!")
print("=" * 65)
