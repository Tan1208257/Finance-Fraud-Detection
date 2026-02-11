from kafka import KafkaConsumer
import os, json, time

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "finance_ticks")

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BOOTSTRAP,
    auto_offset_reset="earliest",
    enable_auto_commit=False,
    group_id=None,  # IMPORTANT: no group, so no offset issues
    consumer_timeout_ms=5000,
    value_deserializer=lambda b: b.decode("utf-8"),
)

print(f"Peeking topic={TOPIC} bootstrap={BOOTSTRAP} ...")
count = 0
start = time.time()
for msg in consumer:
    if count < 10:
        print(msg.value[:300])
    count += 1

print(f"Read {count} messages in {time.time()-start:.2f}s")
consumer.close()
