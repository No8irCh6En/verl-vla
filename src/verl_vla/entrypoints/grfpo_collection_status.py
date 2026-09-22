# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Hydra entrypoint for CPU-only fragmented collection coordination."""

import hydra


@hydra.main(
    config_path="../workflows/config",
    config_name="grfpo_fragmented_collection",
    version_base=None,
)
def main(config) -> None:
    from verl_vla.workflows.grfpo_fragmented_collection import run_grfpo_collection_status

    run_grfpo_collection_status(config)


if __name__ == "__main__":
    main()
