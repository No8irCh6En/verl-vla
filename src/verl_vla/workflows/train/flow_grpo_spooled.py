# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Flow-GRPO wrapper around the shared durable group-RL update driver."""

from .spooled_group_rl import run_spooled_group_rl


def run_spooled_flow_grpo(config):
    return run_spooled_group_rl(config, actor_objective="flow_grpo")


__all__ = ["run_spooled_flow_grpo"]
