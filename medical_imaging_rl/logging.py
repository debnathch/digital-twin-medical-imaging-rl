"""Structured JSON logging for the medical imaging RL application.

NEVER logs patient-identifiable information.
SYNTHETIC / RESEARCH DATA ONLY
"""

import logging
import sys
from pathlib import Path


def setup_logging(config: dict = None):
    """Configure logging."""
    level = logging.INFO
    if config and config.get("level"):
        level = getattr(logging, config["level"].upper(), logging.INFO)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout)

    # File handler
    log_dir = Path(config.get("log_dir", "logs")) if config else Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "experiment.log")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)


def get_logger(name: str):
    return logging.getLogger(name)
