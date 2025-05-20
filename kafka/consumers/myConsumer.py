from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'quickstart-events',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='fraud-detector',
    value_deserializer=lambda m: m.decode('utf-8')  # Only decode, don't parse yet
)

print("Listening for messages on 'quickstart-events'...")

for message in consumer:
    try:
        data = json.loads(message.value)
        print(f"✅ Received: {data}")
    except json.JSONDecodeError:
        print(f"⚠️ Skipping non-JSON message: {message.value}")
