from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pipeline")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def main() -> None:
    root = _project_root()
    logger = _setup_logger(root / "logs" / "pipeline.log")

    in_file = root / "data" / "clean" / "ticks_clean.csv"
    out_dir = root / "data" / "analytics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hourly_analytics.csv"

    if not in_file.exists():
        raise FileNotFoundError(f"Missing input: {in_file}. Run clean_validate first.")

    ticks = pd.read_csv(in_file)
    ticks["ts"] = pd.to_datetime(ticks["ts"], utc=True)

    ticks["hour"] = ticks["ts"].dt.floor("h")


    hourly = (
        ticks.groupby(["symbol", "hour"], as_index=False)
        .agg(
            avg_price=("price", "mean"),
            total_volume=("volume", "sum"),
            trades=("event_id", "count"),
        )
        .sort_values(["symbol", "hour"])
        .reset_index(drop=True)
    )

    hourly.to_csv(out_file, index=False)
    logger.info(f"[aggregate] Input: {in_file}")
    logger.info(f"[aggregate] Output: {out_file} (rows={len(hourly)})")


if __name__ == "__main__":
    main()
