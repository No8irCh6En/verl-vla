# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Vanilla/FPO++ wrapper around the shared durable group-RL update driver."""

from .spooled_group_rl import run_spooled_group_rl


def run_spooled_grfpo(config):
    return run_spooled_group_rl(config, actor_objective="fpo")
