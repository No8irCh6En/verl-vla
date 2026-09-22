# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

import hydra


@hydra.main(
    config_path="../../workflows/config",
    config_name="train/flow_grpo_spooled",
    version_base=None,
)
def main(config):
    from verl_vla.workflows.train.flow_grpo_spooled import run_spooled_flow_grpo

    run_spooled_flow_grpo(config)


if __name__ == "__main__":
    main()
