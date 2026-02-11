from __future__ import annotations
from datetime import datetime, timedelta, timezone
import argparse
import random

import numpy as np
import pandas as pd
from faker import Faker

from src.config import settings
from src.utils import ensure_dir, get_logger

fake = Faker()

def generate_hour(symbols: list[str], start_ts: datetime, n: int) -> pd.DataFrame:
    rows = []
    base_prices = {s: random.uniform(50, 70000) for s in symbols}

    # spread evenly across the hour
    for i in range(n):
        sym = random.choice(symbols)
        ts = start_ts + timedelta(seconds=int((3600 / n) * i))

        price = base_prices[sym] * (1 + random.uniform(-0.0008, 0.0008))
        vol = max(0.0, random.gauss(2.0, 1.0))

        rows.append({
            "event_id": fake.uuid4(),
            "ts": ts.isoformat(),
            "symbol": sym,
            "price": round(float(price), 6),
            "volume": round(float(vol), 6),
        })

        base_prices[sym] = price

    df = pd.DataFrame(rows)

    # --- Required quality challenges (PDF) ---
    # 1) Missing values (~1%)
    if len(df) > 0:
        miss_idx = df.sample(frac=0.01, random_state=42).index
        df.loc[miss_idx, "price"] = np.nan

    # 2) Duplicates (~0.5%)
    dup_count = max(1, int(0.005 * len(df)))
    df = pd.concat([df, df.sample(n=dup_count, random_state=7)], ignore_index=True)

    # 3) Outliers (~0.5%)
    out_count = max(1, int(0.005 * len(df)))
    out_idx = df.sample(n=out_count, random_state=99).index
    df.loc[out_idx, "price"] = df.loc[out_idx, "price"] * 10

    # shuffle final
    return df.sample(frac=1.0, random_state=1).reset_index(drop=True)

def main(hours: int, records_per_hour: int) -> None:
    # Enforce assignment requirement:
    # "at least 1,000 records/hour for 6+ hours" :contentReference[oaicite:2]{index=2}
    hours = max(hours, 6)
    records_per_hour = max(records_per_hour, 1000)

    ensure_dir(settings.data_raw_batch)
    ensure_dir(settings.logs_dir)

    logger = get_logger("generator", settings.logs_dir / "pipeline.log")
    logger.info("Generating finance tick data...")
    logger.info("Target: %s records/hour for %s hours", records_per_hour, hours)

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=hours)

    symbols = list(settings.symbols)

    for h in range(hours):
        hour_start = start + timedelta(hours=h)
        df = generate_hour(symbols, hour_start, records_per_hour)
        fname = settings.data_raw_batch / f"ticks_{hour_start.strftime('%Y%m%d_%H')}.csv"
        df.to_csv(fname, index=False)
        logger.info("Wrote %s rows to %s", len(df), fname)

    logger.info("Done. Raw batch files are in %s", settings.data_raw_batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--records-per-hour", type=int, default=1000)
    args = parser.parse_args()
    main(hours=args.hours, records_per_hour=args.records_per_hour)
