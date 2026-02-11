from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    # .../de2-final-project/batch/clean_validate.py -> root is parents[1]
    return Path(__file__).resolve().parents[1]


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pipeline")
    if logger.handlers:
        return logger  # don't double-add handlers
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

    in_dir = root / "data" / "raw" / "batch_input"
    out_dir = root / "data" / "clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "ticks_clean.csv"

    files = sorted(in_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {in_dir}")

    logger.info(f"[clean_validate] Reading {len(files)} raw CSV files from {in_dir}")

    dfs: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    ticks = pd.concat(dfs, ignore_index=True)

    # Basic schema checks
    required = {"event_id", "ts", "symbol", "price", "volume"}
    missing = required - set(ticks.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean + validate
    ticks["ts"] = pd.to_datetime(ticks["ts"], utc=True, errors="coerce")
    ticks["price"] = pd.to_numeric(ticks["price"], errors="coerce")
    ticks["volume"] = pd.to_numeric(ticks["volume"], errors="coerce")

    before = len(ticks)
    ticks = ticks.dropna(subset=["event_id", "ts", "symbol", "price", "volume"])
    ticks = ticks.drop_duplicates(subset=["event_id"])
    ticks = ticks[(ticks["price"] > 0) & (ticks["volume"] >= 0)]
    after = len(ticks)

    ticks = ticks.sort_values(["symbol", "ts"]).reset_index(drop=True)

    ticks.to_csv(out_file, index=False)
    logger.info(f"[clean_validate] Clean rows: {after}/{before} (dropped {before-after})")
    logger.info(f"[clean_validate] Wrote: {out_file}")


if __name__ == "__main__":
    main()
