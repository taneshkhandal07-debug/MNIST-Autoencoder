"""
logger.py
---------
Centralized logging configuration for the MNIST Autoencoder project.

Provides a factory function to create consistent loggers across all modules,
writing to both console (with color) and a rotating file handler.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


# ANSI color codes for console output
_COLORS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",
}


class _ColorFormatter(logging.Formatter):
    """Custom formatter that adds ANSI color codes to console log records."""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, _COLORS["RESET"])
        reset = _COLORS["RESET"]
        record.levelname = f"{color}{record.levelname:<8}{reset}"
        return super().format(record)


def get_logger(
    name: str,
    log_dir: str = "outputs/logs",
    log_file: str = "training.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Create and return a configured logger instance.

    Creates a logger with two handlers:
      - A color-formatted StreamHandler writing to stdout.
      - A RotatingFileHandler writing plain-text logs to disk.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        log_dir: Directory where log files are stored.
        log_file: Base filename for the rotating log file.
        level: Minimum logging level (default ``logging.INFO``).

    Returns:
        A configured :class:`logging.Logger` instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt_str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # ── Console handler ───────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(_ColorFormatter(fmt=fmt_str, datefmt=date_fmt))

    # ── File handler ──────────────────────────────────────────────────────
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt=fmt_str, datefmt=date_fmt))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
