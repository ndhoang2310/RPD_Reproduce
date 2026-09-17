from typing import Dict

import pytorch_lightning as pl

from .pdc import PDCModule
from .cofly import CoFlyModule

def get_data_module(cfg: Dict) -> pl.LightningDataModule:
    dataset_name = cfg['data']['name']
    if dataset_name == 'phenobench':  # phenobench'CWFID'
        return PDCModule(cfg)

    elif dataset_name == 'CoFly-WeedDB' or dataset_name == 'cofly':
        return CoFlyModule(cfg)

    else:
        assert False, f'There is no parser for: {dataset_name}.'
