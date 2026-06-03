"""
Shared data transfer objects — notifications/types.py

Defines TradeSignal and ApprovalResult, which are the boundary types
between the coordinator/risk gate and the HITL notification layer.
Both are frozen dataclasses — immutable once created.
"""
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class TradeSignal:
    """
    Everything the TelegramNotifier needs to push a HITL approval request.
    Populated by the Coordinator from a risk-gate-approved SignalModel.
    """
    signal_id: int              # SignalModel.id (Integer PK)
    symbol: str                 # e.g. "BTC/USD"
    direction: str              # "long" | "short"
    confidence: Decimal         # 0.0000 – 1.0000
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    position_size: Decimal      # from RiskGate.validate()
    risk_amount: Decimal        # from RiskGate.validate()
    risk_reward_ratio: Decimal
    agent_summary: str | None = None   # coordinator_summary snippet
    ewma_vol: Decimal | None = None    # from seam metadata['ewma_vol'], Phase 3+


@dataclass(frozen=True)
class ApprovalResult:
    """
    Returned by TelegramNotifier.request_approval().
    Passed to ReviewerStateLogger.log_decision() for persistence.
    """
    signal_id: int
    approved: bool
    timed_out: bool
    decision_latency_seconds: float
    reviewer_notes: str | None = None
