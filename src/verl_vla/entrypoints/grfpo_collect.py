# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Hydra entrypoint for one fragmented GRFPO collector lane."""

import json
import os
import time
from pathlib import Path

import hydra


@hydra.main(
    config_path="../workflows/config",
    config_name="grfpo_fragmented_collection",
    version_base=None,
)
def main(config) -> None:
    from verl_vla.workflows.grfpo_fragmented_collection import run_grfpo_candidate

    result = run_grfpo_candidate(config)

    # RoboDojo/Isaac has failure paths that call ``sys.exit(0)`` while launching
    # the simulator (for example an invalid viewport prim).  Slurm would mistake
    # that for a successfully completed collector.  This marker is intentionally
    # written only after the workflow returns normally, so the shell supervisor
    # can distinguish real completion from a false-zero simulator exit.
    marker = os.environ.get("GRFPO_COLLECTOR_COMPLETION_MARKER")
    if marker:
        marker_path = Path(marker)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = marker_path.with_name(f".{marker_path.name}.tmp-{os.getpid()}-{time.time_ns()}")
        temporary.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, marker_path)


if __name__ == "__main__":
    main()
