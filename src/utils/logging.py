import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict


def get_logger(name: str = "yaas", log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class JsonlLogger:
    """Append-only JSONL metrics logger."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def log(self, record: Dict[str, Any]) -> None:
        record = {"ts": time.time(), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
