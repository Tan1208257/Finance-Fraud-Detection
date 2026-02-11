from __future__ import annotations
import time
from collections import defaultdict
from kafka import KafkaConsumer

TOPIC = "finance_ticks"
BOOTSTRAP = "localhost:9092"

from src.config import settings
MICRO_BATCH_SECONDS = settings.micro_batch_seconds  # default 30
  # keep 10 for fast demo; change to 30 later

def main():
    print(f"[BOOT] consumer starting | topic={TOPIC} | bootstrap={BOOTSTRAP} | micro_batch={MICRO_BATCH_SECONDS}s")

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id=None,  # IMPORTANT: avoids offset confusion
        value_deserializer=lambda b: b.decode("utf-8"),
    )

    buffer = []
    batch_start = time.time()

    while True:
        # poll for messages
        msg_pack = consumer.poll(timeout_ms=1000)
        polled = 0

        for _, records in msg_pack.items():
            for r in records:
                buffer.append(r.value)
                polled += 1

        if polled > 0:
            print(f"[POLL] received {polled} msgs | buffer={len(buffer)}")

        now = time.time()
        if now - batch_start >= MICRO_BATCH_SECONDS:
            if not buffer:
                print(f"[BATCH] no messages in last {MICRO_BATCH_SECONDS}s")
                batch_start = now
                continue

            # simple analytics: count per symbol
            counts = defaultdict(int)
            for raw in buffer:
                # raw is JSON string, extract symbol without heavy parsing
                # safe enough for demo: look for "symbol":"XYZ"
                try:
                    sym = raw.split('"symbol":"')[1].split('"')[0]
                except Exception:
                    sym = "BAD_MSG"
                counts[sym] += 1

            print("[BATCH] counts:", dict(counts))
            buffer.clear()
            batch_start = now

if __name__ == "__main__":
    main()
