"""Application logging utilities for the obstetric emergency decision support system."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(name: str = "obstetric_decision_support", log_file: Optional[str] = None) -> logging.Logger:
    """Create and configure a logger for the application.

    Parameters
    ----------
    name:
        Logger name.
    log_file:
        Optional log file path. If not provided, logs are emitted to the console.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
