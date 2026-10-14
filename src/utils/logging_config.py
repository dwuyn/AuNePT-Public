"""
src/utils/logging_config.py
────────────────────────────
Unified logging setup. Every module calls setup_logging() at startup
instead of configuring its own FileHandler.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def setup_logging(
    name: str = "pentest-agent",
    log_dir: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return the root logger.

    - Console handler: INFO and above, coloured.
    - File handler: DEBUG and above, rotating (10 MB × 3 files).
    """
    log_dir = log_dir or os.environ.get("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    if root.handlers:          # already configured — don't add duplicates
        return logging.getLogger(name)

    root.setLevel(logging.DEBUG)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File (rotating)
    fh = RotatingFileHandler(
        os.path.join(log_dir, f"{name}.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return logging.getLogger(name)
