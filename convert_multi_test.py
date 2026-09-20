import argparse

import numpy as np
for _alias, _target in [('Inf', np.inf), ('Infinity', np.inf), ('infty', np.inf),
                        ('NaN', np.nan), ('bool', bool), ('int', int), ('float', float)]:
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)

import pytorch_lightning
from pytorch_lightning import LightningModule, Trainer, Callback
import torch.backends.cudnn
import yaml
from typing import Dict

from models import get_backbone, get_criterion, module, convert_block
from datasets import get_data_module
from callbacks import *
import os


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Convert_Multi_Testing !')

    parser.add_argument('--export_dir', default='log_dir/convert_multi_testing',
                        help='Path to save files related to converting multi_branch into a single branch !')
    parser.add_argument('--config', default='./config/config_convert.yaml',
                        help="Path to configuration file (*.yaml)")
    parser.add_argument('--ckpt_path', type=str,
                        default='./log_dir/.../checkpoints/XXX.ckpt',
                        help='Provide *.ckpt file to convert.')
    parser.add_argument('--deploy_model',
                        default='./log_dir/deploy_convert_multi_testing/deploy_model.ckpt',
                        help='Path to deploy file (*.ckpt/pth)')
    parser.add_argument('--dataset_dir', default=None, type=str,
                        help='Override path_to_dataset in config')

    args = vars(parser.parse_args())

    return args


def load_config(path_to_config_file: str) -> Dict:
    assert os.path.exists(path_to_config_file)

    with open(path_to_config_file) as istream:
        config = yaml.safe_load(istream)

    current_path = config.get('data', {}).get('path_to_dataset', '')
    if not os.path.exists(current_path):
        for cand in [
            "/kaggle/input/datasets/ndhoang2310/phenobench-dataset/PhenoBench",
            "/kaggle/input/phenobench-dataset/PhenoBench",
            "/kaggle/input/datasets/ndhoang2310/phenobench-dataset",
            "/kaggle/input/phenobench-dataset",
        ]:
            if os.path.exists(os.path.join(cand, "train", "images")):
                config['data']['path_to_dataset'] = cand
                break

    return config


def main():
    args = parse_args()

    cfg = load_config(args['config'])
    if args['dataset_dir']:
        cfg['data']['path_to_dataset'] = args['dataset_dir']

    datasetmodule = get_data_module(cfg)
    criterion = get_criterion(cfg)

    print('deploy', cfg['backbone']['deploy'], 'convert', cfg['backbone']['convert'])

    # define backbone
    network = get_backbone(cfg)

    os.makedirs(args['export_dir'], exist_ok=True)

    if cfg['backbone']['deploy']:
        print("Deploy model !")
        convert_path = torch.load(args['deploy_model'])
    else:
        if cfg['backbone']['convert']:
            print('This is just the process of re-parameterized weights!')
            ckpt_dict = torch.load(args['ckpt_path'])
            network.load_state_dict(convert_block(ckpt_dict['state_dict'], 'pdc'), strict=False)
            ckpt_dict['state_dict'] = convert_block(ckpt_dict['state_dict'], 'pdc')

            convert_path = os.path.join(args['export_dir'], 'convert_pdc_weights.pth')
            torch.save(ckpt_dict, convert_path)
            print(f'Saved converted weights to: {convert_path}')
        else:
            print('Using original model and weights !')
            convert_path = torch.load(args['ckpt_path'])

    seg_module = module.SegmentationNetwork(network,
                                            criterion,
                                            cfg['train']['learning_rate'], 
                                            cfg['train']['weight_decay'],
                                            train_step_settings=cfg['train']['step_settings'],
                                            val_step_settings=cfg['val']['step_settings'],
                                            )

    # # Add callbacks
    visualizer_callback = VisualizerCallback(get_visualizers(cfg))
    postprocessor_callback = PostprocessorCallback(get_postprocessors(cfg))
    config_callback = ConfigCallback(cfg)

    # Setup trainer
    trainer = Trainer(default_root_dir=args['export_dir'],
                      accelerator=cfg['val'].get('accelerator', 'auto'),
                      max_epochs=cfg['train']['max_epoch'],
                      devices=cfg['val'].get('devices', 'auto'),
                      num_nodes=cfg['val'].get('num_nodes', 1),
                      callbacks=[visualizer_callback,
                                 postprocessor_callback,
                                 config_callback])

    trainer.validate(seg_module,
                     datasetmodule,
                     ckpt_path=convert_path)


if __name__ == '__main__':

    main()

