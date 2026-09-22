# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Command-line export of a verl-vla actor checkpoint to native Fast-WAM."""

from __future__ import annotations

import argparse
import json

from verl_vla.models.fastwam.native_checkpoint import export_verl_fastwam_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="global_step_N dir, actor dir, or rank-0 actor state")
    parser.add_argument("--base-model-root", required=True, help="immutable native Fast-WAM root providing stats")
    parser.add_argument("--output-root", required=True, help="new native Fast-WAM checkpoint directory")
    parser.add_argument("--step", type=int, default=None, help="native metadata step (default: infer global_step_N)")
    parser.add_argument("--torch-dtype", default="torch.bfloat16")
    args = parser.parse_args()
    manifest = export_verl_fastwam_checkpoint(
        source=args.source,
        base_model_root=args.base_model_root,
        output_root=args.output_root,
        step=args.step,
        torch_dtype=args.torch_dtype,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
