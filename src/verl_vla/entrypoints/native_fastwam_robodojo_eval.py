# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Hydra entrypoint for single-GPU native Fast-WAM + RoboDojo evaluation."""

import hydra


@hydra.main(
    config_path="../workflows/config",
    config_name="native_fastwam_robodojo_eval",
    version_base=None,
)
def main(config) -> None:
    from verl_vla.workflows.native_fastwam_robodojo_eval import run_native_fastwam_robodojo_eval

    result = run_native_fastwam_robodojo_eval(config)
    print(f"success={result['successes']}/{result['num_cases']} mean_score={result['mean_score']:.4f}")


if __name__ == "__main__":
    main()
