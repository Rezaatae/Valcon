from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_DETAILS = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_DETAILS)

db = client["fraud_db"]
transactions_collection = db.get_collection("transactions")


