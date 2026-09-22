# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

from pprint import pprint

import ray
from hydra.utils import instantiate
from omegaconf import OmegaConf

from verl_vla.train_cluster import TrainCluster
from verl_vla.trainer.grfpo import GRFPORayTrainer
from verl_vla.utils.ray_utils import ensure_ray_initialized, get_controller_remote_options


def run_grfpo(config):
    ensure_ray_initialized(config)
    remote_options = get_controller_remote_options(config)
    return ray.get(_run_grfpo_remote.options(**remote_options).remote(config))


@ray.remote
def _run_grfpo_remote(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.set_struct(config, False)
    OmegaConf.resolve(config)
    cluster = TrainCluster(instantiate(config.cluster, _recursive_=False))
    cluster.start()
    try:
        trainer = GRFPORayTrainer(
            trainer_config=config.trainer,
            cluster=cluster,
            tracking_config=OmegaConf.to_container(config, resolve=True),
        )
        trainer.fit()
    finally:
        cluster.shutdown()
