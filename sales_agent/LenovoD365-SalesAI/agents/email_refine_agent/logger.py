"""Dedicated logger for the EmailRefineAgent.

Writes to `email_refine_agent.log` at the repo root and mirrors to stdout.
Pattern mirrors `agents/summary_agent/logger.py`.
"""

import logging
import sys
from pathlib import Path

_LOG_PATH = Path(__file__).resolve().parents[2] / "email_refine_agent.log"


def _setup_email_refine_logger() -> logging.Logger:
    logger = logging.getLogger("email_refine_agent")
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


logger = _setup_email_refine_logger()
