#!/usr/bin/env python3
"""
Logging setup for Armenian Video Dubbing AI.

Uses loguru for structured, colorful logging with file rotation.
Usage:
    from src.utils.logger import setup_logger, logger
    setup_logger()
    logger.info("Starting pipeline...")
"""

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def setup_logger(
    log_dir: str | Path = "logs",
    level: str = "INFO",
    rotation: str = "50 MB",
    retention: str = "30 days",
) -> None:
    """Configure loguru logger with console and file sinks.

    Args:
        log_dir: Directory for log files.
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        rotation: Log file rotation size.
        retention: How long to keep old log files.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler with rich formatting
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler — all levels
    logger.add(
        str(log_dir / "armtts_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation=rotation,
        retention=retention,
        compression="zip",
        enqueue=True,  # Thread-safe
    )

    # Error-only file
    logger.add(
        str(log_dir / "errors_{time:YYYY-MM-DD}.log"),
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
        rotation=rotation,
        retention=retention,
        compression="zip",
        enqueue=True,
    )

    _CONFIGURED = True
    logger.info("Logger initialized — log dir: {}", log_dir.resolve())
