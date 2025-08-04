import time
import random
import json
from kafka import KafkaProducer, KafkaConsumer
from joblib import load
import threading
import os


# Load ML model and preprocessor
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "nbModel.joblib")
PREPROC_PATH = os.path.join(BASE_DIR, "model", "paysim_preprocessor.joblib")

model = load(MODEL_PATH)
preprocessor = load(PREPROC_PATH)

# Kafka config
TOPIC = 'transaction-input'
BOOTSTRAP_SERVERS = 'localhost:9092'

# 1. Data Simulator and Kafka Producer
def simulate_transaction():
    amount = random.randint(1000, 1000000)
    oldbalanceOrg = random.randint(amount, amount + 1000000)
    newbalanceOrig = oldbalanceOrg - amount
    oldbalanceDest = 0
    newbalanceDest = amount

    return {
        'type': random.choice(['CASH_OUT', 'TRANSFER']),
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'nameOrig': f'C{random.randint(10000000, 99999999)}',
        'nameDest': f'M{random.randint(10000000, 99999999)}',
        'isFlaggedFraud': 0
    }

def start_producer():
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    try:
        while True:
            txn = simulate_transaction()
            producer.send(TOPIC, txn)
            print(f"[PRODUCED] {txn}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Producer stopped.")
    finally:
        producer.flush()
        producer.close()

# 2. Kafka Consumer and ML Prediction
def start_consumer():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='transaction-group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    print(f"[CONSUMER] Listening on topic '{TOPIC}'")

    for message in consumer:
        txn = message.value
        print(f"[CONSUMED] {txn}")

        try:
            # Preprocess input
            input_features = preprocessor.transform([[
                txn['type'],
                txn['amount'],
                txn['oldbalanceOrg'],
                txn['newbalanceOrig'],
                txn['oldbalanceDest'],
                txn['newbalanceDest']
            ]])

            # Make prediction
            prediction = model.predict(input_features)
            print(f"[PREDICTED] Is Fraud? => {prediction[0]}")

        except Exception as e:
            print(f"[ERROR] During prediction: {e}")

# 3. Orchestrator
if __name__ == "__main__":
    # Run producer and consumer in separate threads
    producer_thread = threading.Thread(target=start_producer)
    consumer_thread = threading.Thread(target=start_consumer)

    producer_thread.start()
    consumer_thread.start()

    # Keep the main thread alive
    producer_thread.join()
    consumer_thread.join()
