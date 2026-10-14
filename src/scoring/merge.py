"""
src/scoring/merge.py
─────────────────────
Thin wrapper around utils/merge_scores.py that accepts economic_mode
as a parameter instead of reading it from config at import time.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)


def merge_scores(root_dir: str, output_file: str, economic_mode: bool) -> None:
    """Merge all classification*.json files under root_dir into output_file."""
    from utils.merge_scores import merge  # noqa
    merge(root_dir, output_file, economic_mode)
