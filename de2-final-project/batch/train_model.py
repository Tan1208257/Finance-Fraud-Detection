from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


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

    in_file = root / "data" / "features" / "features.csv"
    out_dir = root / "data" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "rf_model.joblib"
    meta_path = out_dir / "rf_model_meta.json"

    if not in_file.exists():
        raise FileNotFoundError(f"Missing input: {in_file}. Run features first.")

    df = pd.read_csv(in_file)
    if df.empty:
        raise ValueError("features.csv is empty. Fix features step or generate more data.")

    feature_cols = ["avg_price", "total_volume", "trades", "return_1h", "ma_3", "vol_3"]
    target_col = "label_up_next"

    X = df[feature_cols]
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(model, model_path)
    meta = {"model_type": "RandomForestClassifier", "feature_cols": feature_cols, "target_col": target_col, "accuracy": float(acc)}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info(f"[train_model] Accuracy: {acc:.4f}")
    logger.info(f"[train_model] Saved model: {model_path}")
    logger.info(f"[train_model] Saved meta:  {meta_path}")


if __name__ == "__main__":
    main()
