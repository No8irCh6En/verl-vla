"""Append site-configured pure-Python dependency fallbacks safely.

The primary interpreter remains authoritative for binary packages.  This is a
portable replacement for the historical Phase-6 runtime shim: machine paths
come only from ``VVLA_PYTHON_EXTRA_PATHS`` in the selected site config.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


for raw_path in os.environ.get("VVLA_PYTHON_EXTRA_PATHS", "").split(os.pathsep):
    if not raw_path:
        continue
    candidate = Path(raw_path).expanduser()
    value = str(candidate)
    if candidate.is_dir() and value not in sys.path:
        sys.path.append(value)
