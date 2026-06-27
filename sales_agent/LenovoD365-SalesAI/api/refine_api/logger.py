"""Dedicated logger for the Email Refine Agent API.

Writes to `refine_api.log` at the repo root and mirrors to stdout.
Pattern mirrors `api/sum_api/logger.py`.
"""

import logging
import sys
from pathlib import Path

# api/refine_api/ -> api/ -> repo root
_LOG_PATH = Path(__file__).resolve().parents[2] / "refine_api.log"


def _setup_refine_api_logger() -> logging.Logger:
    logger = logging.getLogger("refine_api")
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


logger = _setup_refine_api_logger()
