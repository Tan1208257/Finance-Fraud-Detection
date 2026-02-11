from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
from flask import Flask, jsonify, request


# ------------------------------------------------------
# Helper: project root
# ------------------------------------------------------
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = project_root()

# Model paths
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "data" / "models" / "rf_model.joblib")))
META_PATH = Path(os.getenv("META_PATH", str(ROOT / "data" / "models" / "rf_model_meta.json")))

# ------------------------------------------------------
# Flask app
# ------------------------------------------------------
app = Flask(__name__)


# ------------------------------------------------------
# Load model + meta at startup (fail fast if missing)
# ------------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}\n"
        f"Run this first:\n"
        f"  python -m batch.train_model"
    )

model = joblib.load(MODEL_PATH)

if not META_PATH.exists():
    raise FileNotFoundError(
        f"Meta file not found at: {META_PATH}\n"
        f"Run this first:\n"
        f"  python -m batch.train_model"
    )

meta = json.loads(META_PATH.read_text(encoding="utf-8"))

if "feature_cols" not in meta or not meta["feature_cols"]:
    raise ValueError("rf_model_meta.json must contain a non-empty 'feature_cols' list.")

FEATURES = meta["feature_cols"]


# ------------------------------------------------------
# Routes
# ------------------------------------------------------

@app.get("/")
def home():
    return """
    <h2>DE2 Finance ML API</h2>
    <p>Endpoints:</p>
    <ul>
      <li><a href="/health">/health</a> (GET)</li>
      <li>/predict (POST)</li>
    </ul>
    """


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
        "meta_path": str(META_PATH),
        "features": FEATURES
    })


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}

    # validate presence of all required features
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return jsonify({
            "error": "Missing required feature(s)",
            "missing": missing,
            "required_features": FEATURES
        }), 400

    # build DataFrame WITH feature names (prevents sklearn warning)
    try:
        import pandas as pd
        row = {f: float(payload[f]) for f in FEATURES}
        X = pd.DataFrame([row], columns=FEATURES)
    except Exception:
        return jsonify({
            "error": "Invalid feature values. All features must be numeric."
        }), 400

    # predict
    try:
        pred = int(model.predict(X)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0][1])

        return jsonify({
            "prediction": pred,
            "prob_up_next": proba
        })
    except Exception as e:
        # return readable error
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


# ------------------------------------------------------
# Run app (stable mode for Windows)
# ------------------------------------------------------
if __name__ == "__main__":
    # IMPORTANT: debug=False avoids Windows watchdog/reloader issues ("connection reset")
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
