from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
from kafka import KafkaProducer

from src.config import settings
from src.utils import ensure_dir, get_logger, to_json

# Converts the generated data into csv files 
def latest_file(folder: Path) -> Path:
    files = sorted(folder.glob("ticks_*.csv"))
    if not files:
        raise FileNotFoundError(f"No files found in {folder}. Run generator first.")
    return files[-1]

def main(sleep_ms: int = 10) -> None:
    ensure_dir(settings.logs_dir)
    logger = get_logger("producer", settings.logs_dir / "pipeline.log")

    # NOTE: If running inside Docker container, set KAFKA_BOOTSTRAP=kafka:9092
    producer = KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap,
        value_serializer=lambda v: v.encode("utf-8"),
        acks="all",
        retries=5,
    )
# ack=all

    fp = latest_file(settings.data_raw_batch)
    df = pd.read_csv(fp)

    logger.info("Producing %s messages from %s to topic=%s bootstrap=%s",
                len(df), fp.name, settings.topic, settings.kafka_bootstrap)

    sent = 0
    for _, row in df.iterrows():
        msg = {
            "event_id": str(row["event_id"]),
            "ts": str(row["ts"]),
            "symbol": str(row["symbol"]),
            "price": None if pd.isna(row["price"]) else float(row["price"]),
            "volume": float(row["volume"]),
        }
        producer.send(settings.topic, value=to_json(msg))
        sent += 1
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    producer.flush()
    logger.info("Done producing: %s messages", sent)

if __name__ == "__main__":
    main(sleep_ms=5)
