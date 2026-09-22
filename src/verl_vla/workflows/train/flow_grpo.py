from pprint import pprint

import ray
from hydra.utils import instantiate
from omegaconf import OmegaConf

from verl_vla.train_cluster import TrainCluster
from verl_vla.trainer.flow_grpo import FlowGRPORayTrainer
from verl_vla.utils.ray_utils import ensure_ray_initialized, get_controller_remote_options


def run_flow_grpo(config):
    ensure_ray_initialized(config)
    remote_options = get_controller_remote_options(config)
    return ray.get(_run_flow_grpo_remote.options(**remote_options).remote(config))


@ray.remote
def _run_flow_grpo_remote(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.set_struct(config, False)
    OmegaConf.resolve(config)
    cluster = TrainCluster(instantiate(config.cluster, _recursive_=False))
    cluster.start()
    try:
        FlowGRPORayTrainer(
            trainer_config=config.trainer,
            cluster=cluster,
            tracking_config=OmegaConf.to_container(config, resolve=True),
        ).fit()
    finally:
        cluster.shutdown()
