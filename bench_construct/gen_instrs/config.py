"""Load portable construction settings and optional Slurm sharding."""

import os

from src.runtime import load_config, resolve_config_paths


def load_construction_config(path, mode):
    from omegaconf import OmegaConf

    settings = load_config(path)
    cfg = OmegaConf.merge(settings.common, settings[mode])
    return resolve_config_paths(cfg)


def worker_shard(rank=None, world_size=None):
    rank = int(os.environ.get("SLURM_PROCID", "0")) if rank is None else rank
    world_size = (
        int(os.environ.get("SLURM_NTASKS", "1")) if world_size is None else world_size
    )
    if not 0 <= rank < world_size:
        raise ValueError("Worker rank must satisfy 0 <= rank < world_size")
    return rank, world_size
