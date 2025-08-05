from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import transactions_collection
from schemas import Transaction
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or limit to frontend host
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/transactions", response_model=List[Transaction])
async def get_transactions():
    transactions = []
    async for tx in transactions_collection.find().limit(20):
        tx["_id"] = str(tx["_id"])  # convert ObjectId to string
        transactions.append(tx)
    return transactions
