"""Train semantic segmentation model."""

import argparse
from calendar import c
import os
import time
from typing import Dict, Any, List, Optional, Tuple
import warnings

# Tắt tất cả warnings rác từ thư viện cũ
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

# NumPy 2.0 compatibility patch for older PyTorch Lightning / TorchMetrics
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _alias, _target in [('Inf', np.inf), ('Infinity', np.inf), ('infty', np.inf),
                            ('NaN', np.nan), ('bool', bool), ('int', int), ('float', float)]:
        try:
            if not hasattr(np, _alias):
                setattr(np, _alias, _target)
        except Exception:
            pass

import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from callbacks import (ConfigCallback, PostprocessorCallback, VisualizerCallback, get_postprocessors, get_visualizers)
from datasets import get_data_module
from models import get_backbone, get_criterion, model_multimetrics

import torch

def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description='Train RPD Semantic Segmentation Model')
    parser.add_argument('--export_dir', default='log_dir/', help='Path to export dir which saves logs, metrics, etc.')
    parser.add_argument('--config', default='./config/config_deeplearn.yaml',
                        help="Path to configuration file (*.yaml)")
    parser.add_argument('--ckpt_path', default=None, help='Provide *.ckpt file to continue training.')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume training state from checkpoint')
    
    # Các tham số tiện lợi khi chạy trên server SSH
    parser.add_argument('--dataset_dir', default=None, type=str,
                        help='Override path_to_dataset in config (e.g. /data/PhenoBench)')
    parser.add_argument('--batch_size', default=None, type=int,
                        help='Override training batch_size (default: from config, e.g. 4)')
    parser.add_argument('--max_epoch', default=None, type=int,
                        help='Override training max_epoch (default: from config)')
    parser.add_argument('--devices', default=None, type=str,
                        help='Override GPU devices (e.g. 1, "0,", "auto")')
    parser.add_argument('--num_workers', default=None, type=int,
                        help='Override DataLoader num_workers (e.g. 8 for high-RAM server)')
    parser.add_argument('--check_val_every_n_epoch', default=None, type=int,
                        help='Override validation frequency')
    parser.add_argument('--early_stopping_patience', default=None, type=int,
                        help='Patience for EarlyStopping (set 0 or negative to disable EarlyStopping)')
    parser.add_argument('--no_early_stopping', default=False, action='store_true',
                        help='Completely disable EarlyStopping to train for full epochs')

    args = vars(parser.parse_args())
    return args


def load_config(path_to_config_file: str) -> Dict:
    assert os.path.exists(path_to_config_file)

    with open(path_to_config_file) as istream:
        config = yaml.safe_load(istream)

    # Tự động dò tìm dataset PhenoBench trên Kaggle nếu đường dẫn trong YAML không tồn tại
    current_path = config.get('data', {}).get('path_to_dataset', '')
    if not os.path.exists(current_path):
        candidates = [
            "/kaggle/input/datasets/ndhoang2310/phenobench-dataset/PhenoBench",
            "/kaggle/input/phenobench-dataset/PhenoBench",
            "/kaggle/input/datasets/ndhoang2310/phenobench-dataset",
            "/kaggle/input/phenobench-dataset",
        ]
        for cand in candidates:
            if os.path.exists(os.path.join(cand, "train", "images")):
                print(f"[Auto-Detect] Thay thế đường dẫn dataset thành: {cand}")
                config['data']['path_to_dataset'] = cand
                break

    return config


def main():
    args = parse_args()

    cfg = load_config(args['config'])

    # Áp dụng CLI overrides (nếu người dùng truyền vào từ dòng lệnh)
    if args['dataset_dir']:
        cfg['data']['path_to_dataset'] = args['dataset_dir']
        print(f"[Config Override] path_to_dataset -> {args['dataset_dir']}")

    if args['batch_size'] is not None:
        cfg['train']['batch_size'] = args['batch_size']
        print(f"[Config Override] batch_size -> {args['batch_size']}")

    if args['max_epoch'] is not None:
        cfg['train']['max_epoch'] = args['max_epoch']
        print(f"[Config Override] max_epoch -> {args['max_epoch']}")

    if args['devices'] is not None:
        dev_val = int(args['devices']) if args['devices'].isdigit() else args['devices']
        cfg['train']['devices'] = dev_val
        cfg['val']['devices'] = 1 if isinstance(dev_val, int) else dev_val
        print(f"[Config Override] devices -> {dev_val}")

    if args['num_workers'] is not None:
        cfg['data']['num_workers'] = args['num_workers']
        print(f"[Config Override] num_workers -> {args['num_workers']}")

    if args['check_val_every_n_epoch'] is not None:
        cfg['val']['check_val_every_n_epoch'] = args['check_val_every_n_epoch']
        print(f"[Config Override] check_val_every_n_epoch -> {args['check_val_every_n_epoch']}")
    
    if cfg.get('seed') is None:
        seed_val = int(time.time())
        cfg['seed'] = seed_val
    else:
        seed_val = cfg['seed']
    seed_everything(seed_val) 

    datasetmodule = get_data_module(cfg) 
    criterion = get_criterion(cfg)

    # define backbone
    network = get_backbone(cfg)

    if (args['ckpt_path'] is not None) and (not args['resume']):
        seg_module = model_multimetrics.SegmentationNetwork(network,
                                                criterion,
                                                cfg['train']['learning_rate'],
                                                cfg['train']['weight_decay'],
                                                train_step_settings=cfg['train']['step_settings'],
                                                val_step_settings=cfg['val']['step_settings'],
                                                ckpt_path=args['ckpt_path'])
    else:
        seg_module = model_multimetrics.SegmentationNetwork(network,
                                                            criterion,
                                                            cfg['train']['learning_rate'],
                                                            cfg['train']['weight_decay'],
                                                            train_step_settings=cfg['train']['step_settings'],
                                                            val_step_settings=cfg['val']['step_settings'])

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_saver_val_loss = ModelCheckpoint(
        monitor='val_loss',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{val_loss:.4f}',
        mode='min',
        save_last=True)
    checkpoint_saver_val_mIoU = ModelCheckpoint(
        monitor='val_mIoU',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{val_mIoU:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_train_loss = ModelCheckpoint(
        monitor='train_loss',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{train_loss:.4f}',
        mode='min',
        save_last=False)
    checkpoint_saver_train_mIoU = ModelCheckpoint(
        monitor='train_mIoU',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{train_mIoU:.4f}',
        mode='max',
        save_last=False)

    checkpoint_saver_val_mPrecision = ModelCheckpoint(
        monitor='val_mPrecision',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{val_mPrecision:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_train_mPrecision = ModelCheckpoint(
        monitor='train_mPrecision',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{train_mPrecision:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_val_mF1 = ModelCheckpoint(
        monitor='val_mF1',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{val_mF1:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_train_mF1 = ModelCheckpoint(
        monitor='train_mF1',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{train_mF1:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_val_mAcc = ModelCheckpoint(
        monitor='val_mAcc',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{val_mAcc:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_train_mAcc = ModelCheckpoint(
        monitor='train_mAcc',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{train_mAcc:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_val_OverallAcc = ModelCheckpoint(
        monitor='val_OverallAcc',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{val_OverallAcc:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_train_OverallAcc = ModelCheckpoint(
        monitor='train_OverallAcc',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{train_OverallAcc:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_val_mRecall = ModelCheckpoint(
        monitor='val_mRecall',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{val_mRecall:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_train_mRecall = ModelCheckpoint(
        monitor='train_mRecall',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{train_mRecall:.4f}',
        mode='max',
        save_last=False)


    my_checkpoint_savers = [var_value for var_name, var_value in locals().items() if
                            var_name.startswith('checkpoint_saver')]

    visualizer_callback = VisualizerCallback(get_visualizers(cfg),
                                             cfg['train']['vis_train_every_x_epochs'],
                                             cfg['val']['vis_val_every_x_epochs'])

    postprocessor_callback = PostprocessorCallback(get_postprocessors(cfg),
                                                   cfg['train']['postprocess_train_every_x_epochs'],
                                                   cfg['val']['postprocess_val_every_x_epochs'])
    all_callbacks = [
        *my_checkpoint_savers,
        lr_monitor,
        visualizer_callback,
        postprocessor_callback,
        config_callback,
    ]

    # Cấu hình EarlyStopping linh hoạt
    use_early_stopping = not args['no_early_stopping']
    patience = args['early_stopping_patience']
    if patience is not None and patience <= 0:
        use_early_stopping = False

    if use_early_stopping:
        actual_patience = patience if patience is not None else cfg['train'].get('early_stopping_patience', 10)
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=actual_patience,
            min_delta=0.001,
            mode='min',
            verbose=True
        )
        all_callbacks.append(early_stopping)
        print(f"[EarlyStopping] Kích hoạt với patience = {actual_patience}")
    else:
        print("[EarlyStopping] Đã tắt, mô hình sẽ huấn luyện đủ max_epoch theo cấu hình paper.")

    # Đảm bảo export_dir tồn tại
    os.makedirs(args['export_dir'], exist_ok=True)

    # Setup strategy
    train_devices = cfg['train'].get('devices', 'auto')
    train_strategy = cfg['train'].get('strategy', 'auto')
    if train_strategy == 'auto' and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        from pytorch_lightning.strategies import DDPStrategy
        train_strategy = DDPStrategy(find_unused_parameters=False)

    # Setup trainer
    trainer = Trainer(
        accelerator=cfg['train'].get('accelerator', 'auto'),
        devices=train_devices,
        strategy=train_strategy,
        benchmark=cfg['train'].get('benchmark', True),
        default_root_dir=args['export_dir'],
        max_epochs=cfg['train']['max_epoch'],
        check_val_every_n_epoch=cfg['val']['check_val_every_n_epoch'],
        callbacks=all_callbacks)

    if args['ckpt_path'] is None:
        print('Train from scratch.')
        trainer.fit(seg_module, datasetmodule)
    elif (args['ckpt_path'] is not None) and (not args['resume']):
        print('Load pretrained model weights but other params (e.g. learning rate) start from scratch.')
        trainer.fit(seg_module, datasetmodule)
    elif (args['ckpt_path'] is not None) and args['resume']:
        print("Load pretrained model weights and resume training.")
        trainer.fit(seg_module, datasetmodule, ckpt_path=args['ckpt_path'])
    else:
        raise RuntimeError("Can't train any model since the settings are invalid.")


if __name__ == '__main__':
    main()




