"""
Pydantic Schemas — database/schemas.py

Serialization schemas for API responses.
Separate from SQLAlchemy models — models talk to the DB,
schemas talk to the API layer.
"""
from decimal import Decimal
from datetime import datetime
from pydantic import BaseModel


class CandleSchema(BaseModel):
    id: int
    symbol: str
    timeframe: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    model_config = {"from_attributes": True}


class SignalSchema(BaseModel):
    id: int
    symbol: str
    direction: str
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    position_size: Decimal
    risk_amount: Decimal
    risk_reward_ratio: Decimal
    status: str
    technical_summary: str | None
    coordinator_summary: str | None
    created_at: datetime
    expires_at: datetime

    model_config = {"from_attributes": True}


class TradeSchema(BaseModel):
    id: int
    signal_id: int
    mode: str
    symbol: str
    direction: str
    entry_price: Decimal
    exit_price: Decimal | None
    position_size: Decimal
    pnl: Decimal | None
    pnl_pct: Decimal | None
    status: str
    opened_at: datetime
    closed_at: datetime | None

    model_config = {"from_attributes": True}
