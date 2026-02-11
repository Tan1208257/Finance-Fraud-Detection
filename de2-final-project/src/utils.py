from __future__ import annotations
import json
import logging
from pathlib import Path

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def get_logger(name: str, logfile: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # reset handlers every run (prevents "empty file" issues)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logfile.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def from_json(s: str):
    return json.loads(s)
