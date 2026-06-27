"""Dedicated logger for the Summary Agent API.

Writes to `summary_api.log` at the repo root and mirrors to stdout.
Pattern mirrors the per-agent loggers (e.g. agents/summary_agent/logger.py).
"""

import logging
import sys
from pathlib import Path

# api/sum_api/ -> api/ -> repo root
_LOG_PATH = Path(__file__).resolve().parents[2] / "summary_api.log"


def _setup_summary_api_logger() -> logging.Logger:
    logger = logging.getLogger("summary_api")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


logger = _setup_summary_api_logger()

