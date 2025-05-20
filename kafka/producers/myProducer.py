from kafka import KafkaProducer
import json
import time
import random

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_transaction():
    return {
        "transaction_id": random.randint(1000, 9999),
        "user_id": random.randint(1, 100),
        "amount": round(random.uniform(5.0, 5000.0), 2),
        "timestamp": time.time()
    }

while True:
    transaction = generate_transaction()
    producer.send('quickstart-events', transaction)
    print(f"Sent: {transaction}")
    time.sleep(1)
