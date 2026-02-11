from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
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

    features_file = root / "data" / "features" / "features.csv"
    model_path = root / "data" / "models" / "rf_model.joblib"
    meta_path = root / "data" / "models" / "rf_model_meta.json"

    out_dir = root / "data" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not features_file.exists():
        raise FileNotFoundError("Missing features.csv. Run features first.")
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Missing model files. Run train_model first.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_cols = meta["feature_cols"]

    df = pd.read_csv(features_file)

    # Predict on the latest hour per symbol (most recent row per symbol)
    if "hour" in df.columns:
        df["hour"] = pd.to_datetime(df["hour"], utc=True, errors="coerce")

    df = df.sort_values(["symbol", "hour"] if "hour" in df.columns else ["symbol"]).reset_index(drop=True)
    latest = df.groupby("symbol", as_index=False).tail(1).copy()

    X = latest[feature_cols]

    model = joblib.load(model_path)
    proba = model.predict_proba(X)[:, 1]  # probability of class 1

    latest["pred_up_next_proba"] = proba
    latest["pred_up_next"] = (latest["pred_up_next_proba"] >= 0.5).astype(int)

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"predictions_{stamp}.csv"
    latest.to_csv(out_file, index=False)

    logger.info(f"[batch_predict] Wrote: {out_file} (rows={len(latest)})")


if __name__ == "__main__":
    main()
