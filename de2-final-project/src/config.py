from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    # Kafka
    kafka_bootstrap: str = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    topic: str = os.getenv("KAFKA_TOPIC", "finance_ticks")
    consumer_group: str = os.getenv("KAFKA_GROUP", "finance_consumer_v2")


    # Paths
    data_raw_batch: Path = BASE_DIR / "data" / "raw" / "batch_input"
    data_clean: Path = BASE_DIR / "data" / "clean"
    data_analytics: Path = BASE_DIR / "data" / "analytics"
    data_features: Path = BASE_DIR / "data" / "features"
    data_models: Path = BASE_DIR / "data" / "models"
    data_predictions: Path = BASE_DIR / "data" / "predictions"
    logs_dir: Path = BASE_DIR / "logs"

    # Finance settings
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    micro_batch_seconds: int = int(os.getenv("MICRO_BATCH_SECONDS", "30"))
    spike_threshold_pct: float = float(os.getenv("SPIKE_THRESHOLD_PCT", "1.0"))  # 1% in micro-batch

settings = Settings()
