from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class Transaction(BaseModel):
    transaction_id: str
    user_id: str
    timestamp: datetime
    amount: float
    currency: str
    location: str
    device: str
    is_fraud: bool
    risk_score: float
    fraud_reason: Optional[str]
