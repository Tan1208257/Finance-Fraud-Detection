from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
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

    in_file = root / "data" / "analytics" / "hourly_analytics.csv"
    out_dir = root / "data" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "features.csv"

    if not in_file.exists():
        raise FileNotFoundError(f"Missing input: {in_file}. Run aggregate first.")

    df = pd.read_csv(in_file)
    df["hour"] = pd.to_datetime(df["hour"], utc=True, errors="coerce")
    df = df.sort_values(["symbol", "hour"]).reset_index(drop=True)

    # Next-hour movement label (based on next avg_price)
    df["avg_price_next"] = df.groupby("symbol")["avg_price"].shift(-1)
    df["return_next_1h"] = (df["avg_price_next"] / df["avg_price"]) - 1.0
    df["label_up_next"] = (df["return_next_1h"] > 0).astype(int)

    # Features (small windows so your 6 hours per symbol is enough)
    df["return_1h"] = df.groupby("symbol")["avg_price"].pct_change()
    df["ma_3"] = df.groupby("symbol")["avg_price"].transform(lambda s: s.rolling(3, min_periods=3).mean())
    df["vol_3"] = df.groupby("symbol")["return_1h"].transform(lambda s: s.rolling(3, min_periods=3).std())

    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    # drop rows lacking features or label (last hour per symbol has no label)
    df = df.dropna(subset=["hour", "avg_price", "total_volume", "trades", "return_1h", "ma_3", "vol_3", "label_up_next"])
    after = len(df)

    df.to_csv(out_file, index=False)
    logger.info(f"[features] Input: {in_file}")
    logger.info(f"[features] Output: {out_file} (rows={after} kept from {before})")


if __name__ == "__main__":
    main()
