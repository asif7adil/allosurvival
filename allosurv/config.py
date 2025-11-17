import os
from dataclasses import dataclass
from typing import Union

import yaml
from pyprojroot import here

PathLike = Union[str, bytes, os.PathLike]


@dataclass
class DefaultConfig:
    data: PathLike
    data_raw: PathLike
    data_processed: PathLike
    data_interim: PathLike
    data_external: PathLike
    plots: PathLike
    model_dl: PathLike
    model_ml: PathLike



def get_config():
    """
    Get the configuration from the 'config.yml' file.

    Returns:
        DefaultConfig: The configuration object.
    """
    with open(here('config.yml'), encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return DefaultConfig(**data['default'])