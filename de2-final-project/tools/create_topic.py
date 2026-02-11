from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
import time
import os

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "finance_ticks")

def main():
    # small retry loop so it works even if Kafka is still starting
    for attempt in range(1, 21):
        try:
            admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP, client_id="topic_creator")
            topic = NewTopic(name=TOPIC, num_partitions=1, replication_factor=1)
            try:
                admin.create_topics([topic], validate_only=False)
                print(f"✅ Created topic: {TOPIC}")
            except TopicAlreadyExistsError:
                print(f"ℹ️ Topic already exists: {TOPIC}")
            finally:
                admin.close()
            return
        except Exception as e:
            print(f"Waiting for Kafka... attempt {attempt}/20 | {e}")
            time.sleep(1)

    raise RuntimeError("Kafka did not become ready in time.")

if __name__ == "__main__":
    main()
