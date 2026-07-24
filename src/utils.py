"""Shared logging configuration for the churn prediction pipeline."""

import logging
from datetime import datetime, timezone
from pathlib import Path


def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    """Configure a logger that writes to both console and a timestamped file.

    Args:
        name: logger name, typically __name__ of the calling script
        log_dir: directory to store log files

    Returns:
        Configured logger instance
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{name.replace('.', '_')}_{timestamp}.log"

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_fmt)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers if called twice

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
