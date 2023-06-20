"""Config setup for blimp"""
from copy import deepcopy
from typing import Any, Mapping, MutableMapping
import os
import logging
import collections.abc

logger = logging.getLogger(__name__)


def load_config(config_file: str) -> Any:
    """
    Load configuration file and return configuration object.
    Parameters
    ----------
    config_file
        Full path to ``config.py`` file.
    Returns
    -------
    Python module.
    """
    import importlib.util
    import importlib.machinery

    loader = importlib.machinery.SourceFileLoader(os.path.basename(config_file), config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is not None:
        config = importlib.util.module_from_spec(spec)
        loader.exec_module(config)
        return config
    return None


def merged_config(config1: MutableMapping[str, Any], config2: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """
    Update config1 with config2.
    Work with arbitrary nested levels.
    Parameters
    ----------
    config1
        Base configuration dict.
    config2
        Configuration dict containing values that should be updated.
    Returns
    -------
    Updated configuration (copy).
    """
    res_config = deepcopy(config1)
    for k, v in config2.items():
        if isinstance(v, collections.abc.Mapping):
            res_config[k] = merged_config(config1.get(k, {}), v)
        else:
            res_config[k] = v
    return res_config
