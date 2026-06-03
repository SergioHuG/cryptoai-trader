"""
Tests for notifications/reviewer_state.py

The SQLAlchemy Session is a plain MagicMock — no real DB required.
Where the test focuses on log_decision field correctness, the helper
methods (should_escalate, current_session_override_rate, etc.) are
patched so we don't need to set up complex mock execute() chains.
"""
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from notifications.reviewer_state import (
    MIN_HISTORY_FOR_ESCALATION,
    OVERRIDE_CONFIDENCE_THRESHOLD,
    ReviewerStateLogger,
)
from notifications.types import ApprovalResult, TradeSignal


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock()
    session.add = MagicMock()
    session.commit = MagicMock()
    # execute().fetchall() used by current_session_override_rate and _daily_rates
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    mock_result.scalar.return_value = 0
    session.execute.return_value = mock_result
    return session


@pytest.fixture
def logger_obj(mock_session: MagicMock) -> ReviewerStateLogger:
    return ReviewerStateLogger(session=mock_session)


@pytest.fixture
def high_conf_signal() -> TradeSignal:
    return TradeSignal(
        signal_id=10,
        symbol="ETH/USD",
        direction="short",
        confidence=Decimal("0.75"),    # >= OVERRIDE_CONFIDENCE_THRESHOLD
        entry_price=Decimal("3000.00"),
        stop_loss=Decimal("3100.00"),
        take_profit=Decimal("2800.00"),
        position_size=Decimal("0.01"),
        risk_amount=Decimal("100.00"),
        risk_reward_ratio=Decimal("2.0"),
    )


@pytest.fixture
def low_conf_signal() -> TradeSignal:
    return TradeSignal(
        signal_id=11,
        symbol="BTC/USD",
        direction="long",
        confidence=Decimal("0.62"),    # < OVERRIDE_CONFIDENCE_THRESHOLD
        entry_price=Decimal("65000.00"),
        stop_loss=Decimal("63500.00"),
        take_profit=Decimal("68000.00"),
        position_size=Decimal("0.00153846"),
        risk_amount=Decimal("100.00"),
        risk_reward_ratio=Decimal("2.0"),
    )


@pytest.fixture
def approved_result(high_conf_signal: TradeSignal) -> ApprovalResult:
    return ApprovalResult(
        signal_id=high_conf_signal.signal_id,
        approved=True,
        timed_out=False,
        decision_latency_seconds=12.4,
    )


@pytest.fixture
def cancelled_result(high_conf_signal: TradeSignal) -> ApprovalResult:
    return ApprovalResult(
        signal_id=high_conf_signal.signal_id,
        approved=False,
        timed_out=False,
        decision_latency_seconds=44.1,
    )


@pytest.fixture
def timed_out_result(low_conf_signal: TradeSignal) -> ApprovalResult:
    return ApprovalResult(
        signal_id=low_conf_signal.signal_id,
        approved=False,
        timed_out=True,
        decision_latency_seconds=90.0,
    )


# ── log_decision — correct field values ───────────────────────────────────────

def test_log_decision_approve_writes_correct_fields(
    logger_obj: ReviewerStateLogger,
    mock_session: MagicMock,
    high_conf_signal: TradeSignal,
    approved_result: ApprovalResult,
) -> None:
    with patch.object(logger_obj, "should_escalate", return_value=False), \
         patch.object(logger_obj, "historical_mean_override_rate", return_value=Decimal("0")), \
         patch.object(logger_obj, "_compute_z_score", return_value=None):

        logger_obj.log_decision(high_conf_signal, approved_result)

    mock_session.add.assert_called_once()
    row = mock_session.add.call_args[0][0]

    assert row.decision == "approve"
    assert row.signal_id == 10
    assert row.symbol == "ETH/USD"
    assert row.direction == "short"
    assert row.confidence == Decimal("0.75")
    assert float(row.decision_latency_seconds) == pytest.approx(12.4)
    assert row.is_override is False   # approved → never an override


def test_log_decision_cancel_high_conf_sets_override_true(
    logger_obj: ReviewerStateLogger,
    mock_session: MagicMock,
    high_conf_signal: TradeSignal,
    cancelled_result: ApprovalResult,
) -> None:
    """Cancel on confidence >= OVERRIDE_CONFIDENCE_THRESHOLD → is_override=True."""
    with patch.object(logger_obj, "should_escalate", return_value=False), \
         patch.object(logger_obj, "historical_mean_override_rate", return_value=Decimal("0")), \
         patch.object(logger_obj, "_compute_z_score", return_value=None):

        logger_obj.log_decision(high_conf_signal, cancelled_result)

    row = mock_session.add.call_args[0][0]
    assert row.decision == "cancel"
    assert row.is_override is True


def test_log_decision_timeout_low_conf_is_override_false(
    logger_obj: ReviewerStateLogger,
    mock_session: MagicMock,
    low_conf_signal: TradeSignal,
    timed_out_result: ApprovalResult,
) -> None:
    """Timeout on confidence < OVERRIDE_CONFIDENCE_THRESHOLD → is_override=False."""
    with patch.object(logger_obj, "should_escalate", return_value=False), \
         patch.object(logger_obj, "historical_mean_override_rate", return_value=Decimal("0")), \
         patch.object(logger_obj, "_compute_z_score", return_value=None):

        logger_obj.log_decision(low_conf_signal, timed_out_result)

    row = mock_session.add.call_args[0][0]
    assert row.decision == "timeout"
    assert row.is_override is False


def test_log_decision_timeout_decided_at_is_none(
    logger_obj: ReviewerStateLogger,
    mock_session: MagicMock,
    low_conf_signal: TradeSignal,
    timed_out_result: ApprovalResult,
) -> None:
    """Timeouts have no decided_at — the reviewer never responded."""
    with patch.object(logger_obj, "should_escalate", return_value=False), \
         patch.object(logger_obj, "historical_mean_override_rate", return_value=Decimal("0")), \
         patch.object(logger_obj, "_compute_z_score", return_value=None):

        logger_obj.log_decision(low_conf_signal, timed_out_result)

    row = mock_session.add.call_args[0][0]
    assert row.decided_at is None


def test_log_decision_hour_of_day_in_valid_range(
    logger_obj: ReviewerStateLogger,
    mock_session: MagicMock,
    high_conf_signal: TradeSignal,
    approved_result: ApprovalResult,
) -> None:
    with patch.object(logger_obj, "should_escalate", return_value=False), \
         patch.object(logger_obj, "historical_mean_override_rate", return_value=Decimal("0")), \
         patch.object(logger_obj, "_compute_z_score", return_value=None):

        logger_obj.log_decision(high_conf_signal, approved_result)

    row = mock_session.add.call_args[0][0]
    assert isinstance(row.hour_of_day, int)
    assert 0 <= row.hour_of_day <= 23


def test_log_decision_commits_session(
    logger_obj: ReviewerStateLogger,
    mock_session: MagicMock,
    high_conf_signal: TradeSignal,
    approved_result: ApprovalResult,
) -> None:
    with patch.object(logger_obj, "should_escalate", return_value=False), \
         patch.object(logger_obj, "historical_mean_override_rate", return_value=Decimal("0")), \
         patch.object(logger_obj, "_compute_z_score", return_value=None):

        logger_obj.log_decision(high_conf_signal, approved_result)

    mock_session.commit.assert_called_once()


# ── Escalation flag ───────────────────────────────────────────────────────────

def test_escalation_flag_set_when_should_escalate_true(
    logger_obj: ReviewerStateLogger,
    mock_session: MagicMock,
    high_conf_signal: TradeSignal,
    cancelled_result: ApprovalResult,
) -> None:
    with patch.object(logger_obj, "should_escalate", return_value=True), \
         patch.object(logger_obj, "historical_mean_override_rate", return_value=Decimal("0.2")), \
         patch.object(logger_obj, "_compute_z_score", return_value=Decimal("2.5")):

        logger_obj.log_decision(high_conf_signal, cancelled_result)

    row = mock_session.add.call_args[0][0]
    assert row.escalation_flag is True


def test_escalation_flag_false_when_normal_rate(
    logger_obj: ReviewerStateLogger,
    mock_session: MagicMock,
    high_conf_signal: TradeSignal,
    approved_result: ApprovalResult,
) -> None:
    with patch.object(logger_obj, "should_escalate", return_value=False), \
         patch.object(logger_obj, "historical_mean_override_rate", return_value=Decimal("0.2")), \
         patch.object(logger_obj, "_compute_z_score", return_value=Decimal("0.8")):

        logger_obj.log_decision(high_conf_signal, approved_result)

    row = mock_session.add.call_args[0][0]
    assert row.escalation_flag is False


# ── should_escalate ───────────────────────────────────────────────────────────

def test_should_escalate_false_when_insufficient_history(
    logger_obj: ReviewerStateLogger,
) -> None:
    """Below MIN_HISTORY_FOR_ESCALATION rows → no escalation, no std dev."""
    with patch.object(logger_obj, "_history_row_count", return_value=MIN_HISTORY_FOR_ESCALATION - 1):
        assert logger_obj.should_escalate() is False


def test_should_escalate_true_when_z_above_two(
    logger_obj: ReviewerStateLogger,
) -> None:
    """z-score > 2 → escalation fires."""
    with patch.object(logger_obj, "_history_row_count", return_value=25), \
         patch.object(logger_obj, "current_session_override_rate", return_value=Decimal("0.90")), \
         patch.object(logger_obj, "historical_mean_override_rate", return_value=Decimal("0.20")), \
         patch.object(logger_obj, "_historical_stddev", return_value=Decimal("0.10")):

        # z = (0.90 - 0.20) / 0.10 = 7.0 → escalate
        assert logger_obj.should_escalate() is True


def test_should_escalate_false_when_z_within_two(
    logger_obj: ReviewerStateLogger,
) -> None:
    """z-score <= 2 → no escalation."""
    with patch.object(logger_obj, "_history_row_count", return_value=25), \
         patch.object(logger_obj, "current_session_override_rate", return_value=Decimal("0.30")), \
         patch.object(logger_obj, "historical_mean_override_rate", return_value=Decimal("0.20")), \
         patch.object(logger_obj, "_historical_stddev", return_value=Decimal("0.10")):

        # z = (0.30 - 0.20) / 0.10 = 1.0 → no escalation
        assert logger_obj.should_escalate() is False


def test_should_escalate_false_when_stddev_zero(
    logger_obj: ReviewerStateLogger,
) -> None:
    """stddev=0 (all rates identical) → no escalation (avoid division by zero)."""
    with patch.object(logger_obj, "_history_row_count", return_value=25), \
         patch.object(logger_obj, "current_session_override_rate", return_value=Decimal("0.50")), \
         patch.object(logger_obj, "historical_mean_override_rate", return_value=Decimal("0.50")), \
         patch.object(logger_obj, "_historical_stddev", return_value=Decimal("0")):

        assert logger_obj.should_escalate() is False


# ── Constants ─────────────────────────────────────────────────────────────────

def test_override_confidence_threshold_is_point_seven() -> None:
    assert OVERRIDE_CONFIDENCE_THRESHOLD == Decimal("0.70")


def test_min_history_for_escalation_is_ten() -> None:
    assert MIN_HISTORY_FOR_ESCALATION == 10


# ── SQL helper methods — direct coverage ──────────────────────────────────────

def test_current_session_override_rate_empty_returns_zero(
    logger_obj: ReviewerStateLogger,
    mock_session: MagicMock,
) -> None:
    """No rows in window → rate is Decimal('0')."""
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    mock_session.execute.return_value = mock_result

    rate = logger_obj.current_session_override_rate()
    assert rate == Decimal("0")


def test_current_session_override_rate_computes_fraction(
    logger_obj: ReviewerStateLogger,
    mock_session: MagicMock,
) -> None:
    """1 override out of 4 decisions → rate = 0.25."""
    rows = []
    for is_override in [True, False, False, False]:
        r = MagicMock()
        r.is_override = is_override
        rows.append(r)

    mock_result = MagicMock()
    mock_result.fetchall.return_value = rows
    mock_session.execute.return_value = mock_result

    rate = logger_obj.current_session_override_rate()
    assert rate == Decimal("0.25")


def test_history_row_count_returns_integer(
    logger_obj: ReviewerStateLogger,
    mock_session: MagicMock,
) -> None:
    mock_result = MagicMock()
    mock_result.scalar.return_value = 17
    mock_session.execute.return_value = mock_result

    assert logger_obj._history_row_count() == 17


def test_history_row_count_returns_zero_when_none(
    logger_obj: ReviewerStateLogger,
    mock_session: MagicMock,
) -> None:
    mock_result = MagicMock()
    mock_result.scalar.return_value = None
    mock_session.execute.return_value = mock_result

    assert logger_obj._history_row_count() == 0


def test_daily_rates_returns_empty_below_min_history(
    logger_obj: ReviewerStateLogger,
    mock_session: MagicMock,
) -> None:
    """Fewer than MIN_HISTORY_FOR_ESCALATION rows → empty list."""
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [MagicMock()] * (MIN_HISTORY_FOR_ESCALATION - 1)
    mock_session.execute.return_value = mock_result

    assert logger_obj._daily_rates() == []


def test_daily_rates_groups_by_day(
    logger_obj: ReviewerStateLogger,
    mock_session: MagicMock,
) -> None:
    """With enough rows, daily_rates returns one float per day."""
    from datetime import datetime, timezone

    rows = []
    for i in range(MIN_HISTORY_FOR_ESCALATION):
        r = MagicMock()
        r.is_override = (i % 2 == 0)  # alternating
        # Put all rows on the same day so we get one rate bucket
        r.decided_at = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
        rows.append(r)

    mock_result = MagicMock()
    mock_result.fetchall.return_value = rows
    mock_session.execute.return_value = mock_result

    rates = logger_obj._daily_rates()
    # All 10 rows on the same day → one rate bucket = 5/10 = 0.5
    assert len(rates) == 1
    assert rates[0] == pytest.approx(0.5)


def test_historical_stddev_returns_zero_with_one_day(
    logger_obj: ReviewerStateLogger,
) -> None:
    """Only one day's data → stddev cannot be computed → Decimal('0')."""
    with patch.object(logger_obj, "_daily_rates", return_value=[0.5]):
        result = logger_obj._historical_stddev()
    assert result == Decimal("0")


def test_historical_stddev_computes_with_multiple_days(
    logger_obj: ReviewerStateLogger,
) -> None:
    """Two or more days → real stddev."""
    with patch.object(logger_obj, "_daily_rates", return_value=[0.0, 1.0]):
        result = logger_obj._historical_stddev()
    # stdev([0.0, 1.0]) = ~0.7071
    assert float(result) == pytest.approx(0.7071, abs=0.001)


def test_compute_z_score_returns_none_with_insufficient_history(
    logger_obj: ReviewerStateLogger,
) -> None:
    with patch.object(logger_obj, "_history_row_count", return_value=MIN_HISTORY_FOR_ESCALATION - 1):
        assert logger_obj._compute_z_score() is None


def test_compute_z_score_returns_none_when_stddev_zero(
    logger_obj: ReviewerStateLogger,
) -> None:
    with patch.object(logger_obj, "_history_row_count", return_value=25), \
         patch.object(logger_obj, "current_session_override_rate", return_value=Decimal("0.5")), \
         patch.object(logger_obj, "historical_mean_override_rate", return_value=Decimal("0.5")), \
         patch.object(logger_obj, "_historical_stddev", return_value=Decimal("0")):

        assert logger_obj._compute_z_score() is None


def test_compute_z_score_returns_decimal(
    logger_obj: ReviewerStateLogger,
) -> None:
    with patch.object(logger_obj, "_history_row_count", return_value=25), \
         patch.object(logger_obj, "current_session_override_rate", return_value=Decimal("0.8")), \
         patch.object(logger_obj, "historical_mean_override_rate", return_value=Decimal("0.3")), \
         patch.object(logger_obj, "_historical_stddev", return_value=Decimal("0.1")):

        z = logger_obj._compute_z_score()

    assert isinstance(z, Decimal)
    assert z == pytest.approx(Decimal("5.0"))


def test_historical_mean_override_rate_empty_returns_zero(
    logger_obj: ReviewerStateLogger,
) -> None:
    with patch.object(logger_obj, "_daily_rates", return_value=[]):
        result = logger_obj.historical_mean_override_rate()
    assert result == Decimal("0")


def test_historical_mean_override_rate_with_data(
    logger_obj: ReviewerStateLogger,
) -> None:
    with patch.object(logger_obj, "_daily_rates", return_value=[0.2, 0.4, 0.6]):
        result = logger_obj.historical_mean_override_rate()
    assert float(result) == pytest.approx(0.4, abs=0.001)
