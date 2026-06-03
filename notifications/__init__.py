"""
notifications package — HITL interface between the AI pipeline and the human approver.

Public surface:
    TelegramNotifier        — push approval requests, await human decision
    ReviewerStateLogger     — persist decisions, detect override rate drift
    DiscretionarySignalIngestor — ingest human-sourced trade ideas
    TradeSignal, ApprovalResult — shared DTOs
"""
from notifications.discretionary import DiscretionarySignalIngestor
from notifications.reviewer_state import ReviewerStateLogger
from notifications.telegram import TelegramNotifier
from notifications.types import ApprovalResult, TradeSignal

__all__ = [
    "TelegramNotifier",
    "ReviewerStateLogger",
    "DiscretionarySignalIngestor",
    "TradeSignal",
    "ApprovalResult",
]