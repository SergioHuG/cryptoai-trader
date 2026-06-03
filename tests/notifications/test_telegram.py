"""
Tests for notifications/telegram.py

TelegramNotifier uses an injectable _bot parameter so tests never touch
the real Telegram API. The Bot object is replaced with an AsyncMock whose
send_message / get_updates / edit_message_text are fully controllable.

_poll_response is patched at the instance level in most tests so we can
exercise request_approval's result-assembly logic without replaying the
full polling loop.
"""
import asyncio
import os
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from notifications.telegram import TelegramNotifier
from notifications.types import TradeSignal


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_signal() -> TradeSignal:
    return TradeSignal(
        signal_id=42,
        symbol="BTC/USD",
        direction="long",
        confidence=Decimal("0.82"),
        entry_price=Decimal("65000.00"),
        stop_loss=Decimal("63500.00"),
        take_profit=Decimal("68000.00"),
        position_size=Decimal("0.00153846"),
        risk_amount=Decimal("100.00"),
        risk_reward_ratio=Decimal("2.0"),
        agent_summary="EMA crossover confirmed, volume above SMA.",
        ewma_vol=Decimal("0.0152"),
    )


@pytest.fixture
def mock_bot() -> AsyncMock:
    """Pre-configured async mock of telegram.Bot."""
    bot = AsyncMock()
    sent_msg = MagicMock()
    sent_msg.message_id = 99
    bot.send_message.return_value = sent_msg
    bot.get_updates.return_value = []
    bot.edit_message_text = AsyncMock()
    return bot


@pytest.fixture
def notifier(mock_bot: AsyncMock) -> TelegramNotifier:
    return TelegramNotifier(
        bot_token="test-token",
        chat_id="12345",
        approval_timeout=5,  # fast for tests — real default is 90 (ADR-002)
        _bot=mock_bot,
    )


# ── Happy path ────────────────────────────────────────────────────────────────

async def test_request_approval_approved(
    notifier: TelegramNotifier,
    sample_signal: TradeSignal,
) -> None:
    """Human responds approve → approved=True, timed_out=False."""
    with patch.object(notifier, "_poll_response", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = ("approve", None)
        result = await notifier.request_approval(sample_signal)

    assert result.approved is True
    assert result.timed_out is False
    assert result.signal_id == 42
    assert result.reviewer_notes is None
    assert result.decision_latency_seconds >= 0.0


# ── Timeout path ──────────────────────────────────────────────────────────────

async def test_request_approval_timeout(
    notifier: TelegramNotifier,
    sample_signal: TradeSignal,
) -> None:
    """No response within timeout → approved=False, timed_out=True."""
    with patch.object(notifier, "_poll_response", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = ("timeout", None)
        result = await notifier.request_approval(sample_signal)

    assert result.approved is False
    assert result.timed_out is True
    assert result.signal_id == 42


# ── Cancel path ───────────────────────────────────────────────────────────────

async def test_request_approval_cancel(
    notifier: TelegramNotifier,
    sample_signal: TradeSignal,
) -> None:
    """Human cancels → approved=False, timed_out=False."""
    with patch.object(notifier, "_poll_response", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = ("cancel", None)
        result = await notifier.request_approval(sample_signal)

    assert result.approved is False
    assert result.timed_out is False


# ── Message content ───────────────────────────────────────────────────────────

async def test_notification_message_contains_required_fields(
    notifier: TelegramNotifier,
    sample_signal: TradeSignal,
    mock_bot: AsyncMock,
) -> None:
    """Pushed message must include symbol, direction, entry, confidence, risk."""
    with patch.object(notifier, "_poll_response", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = ("approve", None)
        await notifier.request_approval(sample_signal)

    mock_bot.send_message.assert_awaited_once()
    text = mock_bot.send_message.call_args.kwargs["text"]

    assert "BTC/USD" in text
    assert "LONG" in text
    assert "65,000.00" in text    # entry price
    assert "82%" in text          # confidence (0.82 → 82%)
    assert "100.00" in text       # risk_amount


async def test_notification_includes_agent_summary(
    notifier: TelegramNotifier,
    sample_signal: TradeSignal,
    mock_bot: AsyncMock,
) -> None:
    """Agent summary appears in message when provided."""
    with patch.object(notifier, "_poll_response", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = ("approve", None)
        await notifier.request_approval(sample_signal)

    text = mock_bot.send_message.call_args.kwargs["text"]
    assert "EMA crossover confirmed" in text


async def test_notification_includes_ewma_vol(
    notifier: TelegramNotifier,
    sample_signal: TradeSignal,
    mock_bot: AsyncMock,
) -> None:
    """EWMA vol appears in message when provided."""
    with patch.object(notifier, "_poll_response", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = ("approve", None)
        await notifier.request_approval(sample_signal)

    text = mock_bot.send_message.call_args.kwargs["text"]
    assert "0.0152" in text


async def test_notification_no_ewma_vol_when_none(
    notifier: TelegramNotifier,
    mock_bot: AsyncMock,
) -> None:
    """No EWMA vol line when ewma_vol=None."""
    signal = TradeSignal(
        signal_id=1,
        symbol="ETH/USD",
        direction="short",
        confidence=Decimal("0.70"),
        entry_price=Decimal("3000.00"),
        stop_loss=Decimal("3100.00"),
        take_profit=Decimal("2800.00"),
        position_size=Decimal("0.01"),
        risk_amount=Decimal("100.00"),
        risk_reward_ratio=Decimal("2.0"),
        ewma_vol=None,
    )
    with patch.object(notifier, "_poll_response", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = ("approve", None)
        await notifier.request_approval(signal)

    text = mock_bot.send_message.call_args.kwargs["text"]
    assert "EWMA" not in text


# ── send_alert ────────────────────────────────────────────────────────────────

async def test_send_alert_sends_one_message_no_poll(
    notifier: TelegramNotifier,
    mock_bot: AsyncMock,
) -> None:
    """send_alert sends exactly one message and never polls for a response."""
    with patch.object(notifier, "_poll_response", new_callable=AsyncMock) as mock_poll:
        await notifier.send_alert("Kill switch triggered: daily drawdown limit reached.")
        mock_poll.assert_not_awaited()

    mock_bot.send_message.assert_awaited_once()
    text = mock_bot.send_message.call_args.kwargs["text"]
    assert "Kill switch" in text
    assert "SYSTEM ALERT" in text


# ── ADR-005 — env factory ─────────────────────────────────────────────────────

def test_from_env_reads_token_and_chat_id() -> None:
    with patch.dict(os.environ, {
        "TELEGRAM_BOT_TOKEN": "env-token-abc",
        "TELEGRAM_CHAT_ID": "env-chat-789",
    }):
        n = TelegramNotifier.from_env()

    assert n._bot_token == "env-token-abc"
    assert n._chat_id == "env-chat-789"
    assert n._approval_timeout == 90   # ADR-002 default


def test_from_env_raises_on_missing_token() -> None:
    env = {"TELEGRAM_CHAT_ID": "12345"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(EnvironmentError, match="TELEGRAM_BOT_TOKEN"):
            TelegramNotifier.from_env()


def test_from_env_raises_on_missing_chat_id() -> None:
    env = {"TELEGRAM_BOT_TOKEN": "some-token"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(EnvironmentError, match="TELEGRAM_CHAT_ID"):
            TelegramNotifier.from_env()


# ── Latency measurement ───────────────────────────────────────────────────────

async def test_decision_latency_reflects_elapsed_time(
    notifier: TelegramNotifier,
    sample_signal: TradeSignal,
) -> None:
    """decision_latency_seconds is >= the real polling delay."""
    async def slow_poll(*_args, **_kwargs):
        await asyncio.sleep(0.06)
        return ("approve", None)

    with patch.object(notifier, "_poll_response", side_effect=slow_poll):
        result = await notifier.request_approval(sample_signal)

    assert result.decision_latency_seconds >= 0.05


# ── _poll_response unit tests ─────────────────────────────────────────────────

async def test_poll_response_returns_timeout_when_deadline_exceeded(
    notifier: TelegramNotifier,
    mock_bot: AsyncMock,
) -> None:
    """
    When the timeout deadline is already elapsed, _poll_response must
    return ("timeout", None) on the very first iteration without making
    a getUpdates call.
    """
    import time

    # Force remaining time to be negative by setting sent_at far in the past
    past = time.monotonic() - (notifier._approval_timeout + 10)
    decision, notes = await notifier._poll_response(
        message_id=99,
        signal_id=42,
        sent_at=past,
    )

    assert decision == "timeout"
    assert notes is None
    mock_bot.get_updates.assert_not_awaited()


async def test_poll_response_returns_approve_on_matching_callback(
    notifier: TelegramNotifier,
    mock_bot: AsyncMock,
) -> None:
    """
    _poll_response returns ("approve", None) when a matching approve
    callback_query arrives in the first batch of updates.
    """
    import time

    # Build a fake update with the correct callback data
    fake_cq = MagicMock()
    fake_cq.data = "approve:42"
    fake_cq.message.message_id = 99
    fake_cq.answer = AsyncMock()

    fake_update = MagicMock()
    fake_update.update_id = 1001
    fake_update.callback_query = fake_cq

    mock_bot.get_updates.return_value = [fake_update]

    decision, notes = await notifier._poll_response(
        message_id=99,
        signal_id=42,
        sent_at=time.monotonic(),
    )

    assert decision == "approve"
    assert notes is None
    fake_cq.answer.assert_awaited_once()


async def test_poll_response_ignores_callback_for_different_message(
    notifier: TelegramNotifier,
    mock_bot: AsyncMock,
) -> None:
    """
    Callbacks for a different message_id must be skipped; polling continues
    until timeout rather than returning a wrong result.
    """
    import time

    # Callback for a different message — should be ignored
    fake_cq = MagicMock()
    fake_cq.data = "approve:42"
    fake_cq.message.message_id = 9999   # wrong message
    fake_cq.answer = AsyncMock()

    fake_update = MagicMock()
    fake_update.update_id = 2001
    fake_update.callback_query = fake_cq

    # First call returns the wrong-message update; subsequent calls return [].
    # Use an async function so the mock never raises StopAsyncIteration
    # when the polling loop keeps calling after the list would be exhausted.
    call_count = 0

    async def get_updates_effect(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        return [fake_update] if call_count == 1 else []

    mock_bot.get_updates.side_effect = get_updates_effect

    # Set sent_at far enough in the past that we time out quickly
    past = time.monotonic() - (notifier._approval_timeout - 1)
    decision, _ = await notifier._poll_response(
        message_id=99,
        signal_id=42,
        sent_at=past,
    )

    assert decision == "timeout"
    fake_cq.answer.assert_not_awaited()


# ── TelegramError handling ────────────────────────────────────────────────────

async def test_edit_message_text_error_is_swallowed(
    notifier: TelegramNotifier,
    sample_signal: TradeSignal,
    mock_bot: AsyncMock,
) -> None:
    """TelegramError on edit_message_text must not propagate — non-critical."""
    from telegram.error import TelegramError

    mock_bot.edit_message_text.side_effect = TelegramError("message not modified")

    with patch.object(notifier, "_poll_response", new_callable=AsyncMock) as mock_poll:
        mock_poll.return_value = ("approve", None)
        result = await notifier.request_approval(sample_signal)   # must not raise

    assert result.approved is True


async def test_poll_response_continues_after_get_updates_error(
    notifier: TelegramNotifier,
    mock_bot: AsyncMock,
) -> None:
    """TelegramError on get_updates → updates treated as [] and loop continues."""
    import time
    from telegram.error import TelegramError

    fake_cq = MagicMock()
    fake_cq.data = "approve:42"
    fake_cq.message.message_id = 99
    fake_cq.answer = AsyncMock()
    fake_update = MagicMock()
    fake_update.update_id = 3001
    fake_update.callback_query = fake_cq

    call_count = 0

    async def get_updates_with_error(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise TelegramError("network hiccup")
        return [fake_update]

    mock_bot.get_updates.side_effect = get_updates_with_error

    decision, _ = await notifier._poll_response(
        message_id=99,
        signal_id=42,
        sent_at=time.monotonic(),
    )

    assert decision == "approve"   # recovered on second call


async def test_poll_response_skips_update_with_no_callback_query(
    notifier: TelegramNotifier,
    mock_bot: AsyncMock,
) -> None:
    """Update with callback_query=None is ignored (non-button update)."""
    import time

    # First update: no callback_query (e.g. a plain message)
    null_cq_update = MagicMock()
    null_cq_update.update_id = 4001
    null_cq_update.callback_query = None

    # Second update: the real approval
    real_cq = MagicMock()
    real_cq.data = "approve:42"
    real_cq.message.message_id = 99
    real_cq.answer = AsyncMock()
    real_update = MagicMock()
    real_update.update_id = 4002
    real_update.callback_query = real_cq

    call_count = 0

    async def updates_fn(*_a, **_kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [null_cq_update, real_update]
        return []

    mock_bot.get_updates.side_effect = updates_fn

    decision, _ = await notifier._poll_response(
        message_id=99,
        signal_id=42,
        sent_at=time.monotonic(),
    )

    assert decision == "approve"


async def test_poll_response_returns_cancel_on_cancel_callback(
    notifier: TelegramNotifier,
    mock_bot: AsyncMock,
) -> None:
    """_poll_response returns ('cancel', None) on the cancel button."""
    import time

    fake_cq = MagicMock()
    fake_cq.data = "cancel:42"
    fake_cq.message.message_id = 99
    fake_cq.answer = AsyncMock()
    fake_update = MagicMock()
    fake_update.update_id = 5001
    fake_update.callback_query = fake_cq

    mock_bot.get_updates.return_value = [fake_update]

    decision, notes = await notifier._poll_response(
        message_id=99,
        signal_id=42,
        sent_at=time.monotonic(),
    )

    assert decision == "cancel"
    assert notes is None
    fake_cq.answer.assert_awaited_once()
