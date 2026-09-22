#!/usr/bin/env python3
"""Fail-fast import check for the supported Fast-WAM/RoboDojo workflows.

This deliberately imports the modules used while Fast-WAM is constructed.
Checking only package metadata is insufficient for overlay environments: a
package may be installed in the base interpreter while one of its transitive
imports is missing from the overlay selected by ``bin/python``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def existing_directory(value: str) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=existing_directory)
    parser.add_argument("--robodojo-root", required=True, type=existing_directory)
    parser.add_argument("--policy-root", required=True, type=existing_directory)
    args = parser.parse_args()

    search_paths = (
        args.repo_root / "src",
        args.robodojo_root,
        args.policy_root / "FastWAM/src",
    )
    for path in reversed(search_paths):
        if not path.is_dir():
            raise FileNotFoundError(f"required Python source directory is missing: {path}")
        sys.path.insert(0, str(path))

    modules = ("git", "accelerate", "torch", "verl", "verl_vla", "fastwam.runtime")
    imported: dict[str, str] = {}
    failures: dict[str, str] = {}
    for name in modules:
        try:
            module = importlib.import_module(name)
        except BaseException as exc:  # Import-time native failures matter too.
            failures[name] = f"{type(exc).__name__}: {exc}"
        else:
            imported[name] = str(getattr(module, "__file__", "built-in"))

    result = {
        "python": sys.executable,
        "prefix": sys.prefix,
        "search_paths": [str(path) for path in search_paths],
        "imported": imported,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        joined = ", ".join(f"{name}: {reason}" for name, reason in failures.items())
        raise SystemExit(
            "Fast-WAM/RoboDojo runtime preflight failed. Install the project "
            f"dependencies into this exact Python entrypoint ({sys.executable}); {joined}"
        )


if __name__ == "__main__":
    main()
