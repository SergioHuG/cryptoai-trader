"""
Signal Message Formatter — notifications/formatter.py

Builds the Telegram message text and inline keyboard for a HITL approval
request. Kept separate from telegram.py so formatting can be tested without
a live bot connection.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from agents.risk import SIGNAL_TIMEOUT_SECONDS

if TYPE_CHECKING:
    from notifications.types import TradeSignal


def format_signal_message(signal: TradeSignal) -> tuple[str, InlineKeyboardMarkup]:
    """
    Returns (message_text, keyboard) for a pending trade signal.

    message_text is HTML-formatted for Telegram (parse_mode='HTML').
    keyboard contains Approve / Cancel inline buttons keyed to signal_id
    so the polling loop can match responses to the correct signal.
    """
    direction_label = "🟢 LONG" if signal.direction == "long" else "🔴 SHORT"
    confidence_pct = float(signal.confidence) * 100

    text = (
        f"<b>📊 {signal.symbol} — {direction_label}</b>\n\n"
        f"💰 Entry:  <b>${signal.entry_price:,.2f}</b>\n"
        f"🛑 Stop:   ${signal.stop_loss:,.2f}\n"
        f"🎯 Target: ${signal.take_profit:,.2f}\n"
        f"📐 R/R:    1:{float(signal.risk_reward_ratio):.1f}\n"
        f"⚖️ Risk:   ${signal.risk_amount:.2f}  (1% rule)\n"
        f"🤖 Confidence: {confidence_pct:.0f}%\n"
    )

    if signal.agent_summary:
        text += f"\n💭 <i>{signal.agent_summary}</i>\n"

    if signal.ewma_vol is not None:
        text += f"📈 EWMA Vol: {float(signal.ewma_vol):.4f}\n"

    text += f"\n⏱️ Auto-cancels in {SIGNAL_TIMEOUT_SECONDS}s — tap to decide."

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "✅ Approve",
                callback_data=f"approve:{signal.signal_id}",
            ),
            InlineKeyboardButton(
                "❌ Cancel",
                callback_data=f"cancel:{signal.signal_id}",
            ),
        ]
    ])

    return text, keyboard


def format_alert(message: str) -> str:
    """Wraps a one-way system alert in standard HTML envelope."""
    return f"🚨 <b>SYSTEM ALERT</b>\n\n{message}"
