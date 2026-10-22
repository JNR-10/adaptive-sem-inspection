"""Simple structured run logging without notebook-only state."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from seminspect.utils.io import utc_now


def configure_logging(run_dir: str | Path, name: str = "seminspect") -> logging.Logger:
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(destination / "stdout.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def log_event(path: str | Path, event: str, **fields: Any) -> None:
    record = {"timestamp": utc_now(), "event": event, **fields}
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
