# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

import hydra


@hydra.main(config_path="../../workflows/config", config_name="train/grfpo", version_base=None)
def main(config):
    from verl_vla.workflows.train.grfpo import run_grfpo

    run_grfpo(config)


if __name__ == "__main__":
    main()
