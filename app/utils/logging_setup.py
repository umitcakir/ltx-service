"""Rotating service log plus one log file per job."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator

from app.config import LoggingConfig

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_configured = False


def configure_logging(config: LoggingConfig) -> None:
    """Attach a rotating file handler + console handler to the root logger."""
    global _configured
    if _configured:
        return

    config.dir.mkdir(parents=True, exist_ok=True)
    if config.per_job_files:
        (config.dir / "jobs").mkdir(parents=True, exist_ok=True)

    level = getattr(logging, config.level.upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT)

    file_handler = RotatingFileHandler(
        config.dir / "ltx-service.log",
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [file_handler, console_handler]
    _configured = True


@contextmanager
def job_log_context(config: LoggingConfig, job_id: str) -> Iterator[logging.Logger]:
    """Logger that also writes to ``logs/jobs/<job_id>.log`` for the job's lifetime."""
    logger = logging.getLogger(f"ltx.job.{job_id}")
    handler: logging.Handler | None = None

    if config.per_job_files:
        job_dir = config.dir / "jobs"
        job_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            job_dir / f"{job_id}.log",
            maxBytes=config.max_bytes,
            backupCount=1,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)

    try:
        yield logger
    finally:
        if handler is not None:
            logger.removeHandler(handler)
            handler.close()


def job_log_path(config: LoggingConfig, job_id: str) -> Path:
    return config.dir / "jobs" / f"{job_id}.log"
