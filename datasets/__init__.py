from typing import Dict

import pytorch_lightning as pl

from .pdc import PDCModule


def get_data_module(cfg: Dict) -> pl.LightningDataModule:
    dataset_name = cfg['data']['name']
    if dataset_name == 'phenobench':
        return PDCModule(cfg)

    elif dataset_name in ('CoFly-WeedDB', 'cofly'):
        from .cofly import CoFlyModule  # lazy import: chỉ load khi thực sự cần
        return CoFlyModule(cfg)

    else:
        assert False, f'There is no parser for: {dataset_name}.'
