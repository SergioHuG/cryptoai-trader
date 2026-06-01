"""
Telegram HITL Notifier — notifications/telegram.py

Pushes pending trade signals to the authorized Telegram chat and awaits
human approval. Auto-cancels after SIGNAL_TIMEOUT_SECONDS (90s, ADR-002).

Every trade reaching this module has already passed the RiskGate.
No trade proceeds to execution without an explicit approve response.

Environment variables required (ADR-005 — never hardcoded):
    TELEGRAM_BOT_TOKEN  — bot token from @BotFather
    TELEGRAM_CHAT_ID    — authorized chat / user ID
"""
import logging
import os
import time

from telegram import Bot
from telegram.error import TelegramError

from agents.risk import SIGNAL_TIMEOUT_SECONDS
from notifications.formatter import format_alert, format_signal_message
from notifications.types import ApprovalResult, TradeSignal

logger = logging.getLogger(__name__)

# How long each getUpdates long-poll call blocks (seconds).
# Kept short enough that the timeout deadline is checked frequently.
_POLL_CHUNK_SECONDS = 10


class TelegramNotifier:
    """
    HITL approval gate via Telegram.

    Coordinator usage:
        notifier = TelegramNotifier.from_env()
        result   = await notifier.request_approval(signal)
        if not result.approved:
            return   # trade cancelled or timed out
        # → hand off to execution layer
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        approval_timeout: int = SIGNAL_TIMEOUT_SECONDS,
        _bot: Bot | None = None,  # injectable for tests
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._approval_timeout = approval_timeout
        self._bot: Bot = _bot or Bot(token=bot_token)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "TelegramNotifier":
        """
        Construct from environment variables (ADR-005).
        Raises EnvironmentError — not None — if either variable is missing
        so callers can distinguish config failures from runtime errors.
        """
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise EnvironmentError(
                "TELEGRAM_BOT_TOKEN is not set. "
                "Add it to your .env file or environment."
            )
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not chat_id:
            raise EnvironmentError(
                "TELEGRAM_CHAT_ID is not set. "
                "Add it to your .env file or environment."
            )
        return cls(bot_token=token, chat_id=chat_id)

    # ── Public API ────────────────────────────────────────────────────────────

    async def request_approval(self, signal: TradeSignal) -> ApprovalResult:
        """
        Push the signal to Telegram and wait for a human response.

        Returns ApprovalResult with:
          - approved=True  on /approve within timeout
          - approved=False, timed_out=False  on /cancel
          - approved=False, timed_out=True   on timeout (auto-cancel)

        Never raises on timeout — always returns a result.
        """
        text, keyboard = format_signal_message(signal)
        sent_at = time.monotonic()

        sent_message = await self._bot.send_message(
            chat_id=self._chat_id,
            text=text,
            reply_markup=keyboard,
            parse_mode="HTML",
        )

        decision, notes = await self._poll_response(
            message_id=sent_message.message_id,
            signal_id=signal.signal_id,
            sent_at=sent_at,
        )

        latency = time.monotonic() - sent_at

        # Edit the original message to reflect the final outcome.
        # Non-critical: swallow errors (message may have been deleted, etc.)
        outcome_suffix = {
            "approve": "\n\n✅ <b>APPROVED</b> — sending to execution.",
            "cancel":  "\n\n❌ <b>CANCELLED</b> by reviewer.",
            "timeout": "\n\n⏰ <b>AUTO-CANCELLED</b> — no response within timeout.",
        }[decision]

        try:
            await self._bot.edit_message_text(
                chat_id=self._chat_id,
                message_id=sent_message.message_id,
                text=text + outcome_suffix,
                parse_mode="HTML",
            )
        except TelegramError:
            pass

        return ApprovalResult(
            signal_id=signal.signal_id,
            approved=(decision == "approve"),
            timed_out=(decision == "timeout"),
            decision_latency_seconds=latency,
            reviewer_notes=notes,
        )

    async def send_alert(self, message: str) -> None:
        """
        One-way system alert (kill switch triggered, drawdown warning, etc.).
        Bypasses any rate limiting — always delivered immediately.
        """
        await self._bot.send_message(
            chat_id=self._chat_id,
            text=format_alert(message),
            parse_mode="HTML",
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _poll_response(
        self,
        message_id: int,
        signal_id: int,
        sent_at: float,
    ) -> tuple[str, str | None]:
        """
        Long-poll getUpdates until the reviewer taps Approve or Cancel,
        or the approval_timeout deadline is reached.

        Returns ("approve"|"cancel"|"timeout", notes_or_None).
        """
        offset: int | None = None

        while True:
            elapsed = time.monotonic() - sent_at
            remaining = self._approval_timeout - elapsed

            if remaining <= 0:
                logger.info(
                    "Signal %s timed out after %.1fs — auto-cancelling.",
                    signal_id,
                    elapsed,
                )
                return ("timeout", None)

            poll_timeout = min(int(remaining), _POLL_CHUNK_SECONDS)

            try:
                updates = await self._bot.get_updates(
                    offset=offset,
                    timeout=poll_timeout,
                    allowed_updates=["callback_query"],
                )
            except TelegramError as exc:
                logger.warning("getUpdates error: %s — retrying.", exc)
                updates = []

            for update in updates:
                offset = update.update_id + 1

                cq = update.callback_query
                if cq is None:
                    continue
                # Guard: only act on callbacks for this specific message
                if cq.message.message_id != message_id:
                    continue

                await cq.answer()  # clears the "loading" spinner on the button

                if cq.data == f"approve:{signal_id}":
                    logger.info("Signal %s APPROVED by reviewer.", signal_id)
                    return ("approve", None)

                if cq.data == f"cancel:{signal_id}":
                    logger.info("Signal %s CANCELLED by reviewer.", signal_id)
                    return ("cancel", None)
