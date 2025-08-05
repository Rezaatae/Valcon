import asyncio
from datetime import datetime, timedelta
import random
from database import transactions_collection

LOCATIONS = ["New York, USA", "London, UK", "Delhi, India", "Berlin, Germany", "Lagos, Nigeria"]
DEVICES = ["iPhone 13", "Samsung S21", "MacBook Pro", "Windows Laptop", "Pixel 6"]
CURRENCIES = ["USD", "GBP", "EUR", "INR", "NGN"]

def generate_transaction(i):
    amount = round(random.uniform(5, 5000), 2)
    is_fraud = random.random() < 0.2  # 20% fraud
    return {
        "transaction_id": f"txn_{i}",
        "user_id": f"user_{random.randint(1, 50)}",
        "timestamp": datetime.utcnow() - timedelta(minutes=i),
        "amount": amount,
        "currency": random.choice(CURRENCIES),
        "location": random.choice(LOCATIONS),
        "device": random.choice(DEVICES),
        "is_fraud": is_fraud,
        "risk_score": round(random.uniform(0.1, 0.99), 2),
        "fraud_reason": "Anomalous behavior" if is_fraud else None,
    }

async def seed_data(n=100):
    transactions = [generate_transaction(i) for i in range(n)]
    await transactions_collection.insert_many(transactions)
    print(f"Seeded {n} transactions.")

if __name__ == "__main__":
    asyncio.run(seed_data(100))
