from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class TransactionRecord(BaseModel):
    TransactionId: Optional[str]
    BatchId: Optional[str]
    AccountId: Optional[str]
    SubscriptionId: Optional[str]
    CustomerId: str
    CurrencyCode: Optional[str]
    CountryCode: Optional[int]
    ProviderId: Optional[int]
    ProductId: Optional[str]
    ProductCategory: Optional[str]
    ChannelId: Optional[str]
    Amount: float
    Value: float
    TransactionStartTime: datetime
    PricingStrategy: Optional[str]
    FraudResult: Optional[int] = 0


class PredictionRequest(BaseModel):
    transactions: List[TransactionRecord]


class PredictionResponse(BaseModel):
    customer_id: str
    risk_probability: float
    risk_label: int
