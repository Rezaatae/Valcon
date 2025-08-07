from fastapi import APIRouter, Query, HTTPException
from bson import ObjectId
from database import transactions_collection
from models import Transaction
from typing import List, Optional

router = APIRouter()

@router.get("/transactions", response_model=List[Transaction])
async def get_transactions(
    status: Optional[str] = Query(None),
    min_amount: Optional[float] = Query(None),
    max_amount: Optional[float] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = 100
):
    query = {}

    if status:
        query["status"] = status
    if min_amount is not None:
        query["amount"] = query.get("amount", {})
        query["amount"]["$gte"] = min_amount
    if max_amount is not None:
        query["amount"] = query.get("amount", {})
        query["amount"]["$lte"] = max_amount
    if search:
        query["description"] = {"$regex": search, "$options": "i"}

    cursor = transactions_collection.find(query).limit(limit)
    results = await cursor.to_list(length=limit)
    return results


@router.get("/transaction/{id}", response_model=Transaction)
async def get_transaction(id: str):
    if not ObjectId.is_valid(id):
        raise HTTPException(status_code=400, detail="Invalid ID format")

    result = await transactions_collection.find_one({"_id": ObjectId(id)})
    if not result:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return result
