"""
Tests for agents/risk.py

This module requires 100% test coverage before any other execution code is built.
Every risk rule has both a passing and failing test case.
"""
import pytest
from decimal import Decimal
from agents.risk import (
    RiskGate,
    RiskValidationResult,
    RejectionReason,
    SystemState,
    MAX_RISK_PER_TRADE_PCT,
    MAX_CONCURRENT_POSITIONS,
    DAILY_DRAWDOWN_KILL_SWITCH_PCT,
    SIGNAL_TIMEOUT_SECONDS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def risk_gate():
    return RiskGate()


@pytest.fixture
def healthy_state():
    """A system state with no issues — all checks should pass."""
    return SystemState(
        total_capital=Decimal("10000.00"),
        available_capital=Decimal("10000.00"),
        open_positions=0,
        daily_pnl_pct=0.0,
        kill_switch_active=False,
    )


@pytest.fixture
def valid_long_signal():
    """A valid long signal with 1:2 R/R."""
    return {
        "symbol": "BTC/USD",
        "direction": "long",
        "entry_price": Decimal("42000.00"),
        "stop_loss": Decimal("41500.00"),    # $500 risk
        "take_profit": Decimal("43000.00"),  # $1000 reward → 1:2
    }


@pytest.fixture
def valid_short_signal():
    """A valid short signal with 1:2 R/R."""
    return {
        "symbol": "ETH/USD",
        "direction": "short",
        "entry_price": Decimal("2200.00"),
        "stop_loss": Decimal("2300.00"),   # $100 risk
        "take_profit": Decimal("2000.00"), # $200 reward → 1:2
    }


# ── Constant value tests ──────────────────────────────────────────────────────

class TestRiskConstants:
    """Risk constants must never change without explicit instruction."""

    def test_max_risk_per_trade_is_one_percent(self):
        assert MAX_RISK_PER_TRADE_PCT == 0.01

    def test_max_concurrent_positions_is_three(self):
        assert MAX_CONCURRENT_POSITIONS == 3

    def test_daily_drawdown_kill_switch_is_five_percent(self):
        assert DAILY_DRAWDOWN_KILL_SWITCH_PCT == 0.05

    def test_signal_timeout_is_ninety_seconds(self):
        assert SIGNAL_TIMEOUT_SECONDS == 90


# ── Kill switch tests ─────────────────────────────────────────────────────────

class TestKillSwitch:
    def test_rejects_when_kill_switch_active(self, risk_gate, valid_long_signal):
        state = SystemState(
            total_capital=Decimal("10000"),
            available_capital=Decimal("10000"),
            open_positions=0,
            daily_pnl_pct=0.0,
            kill_switch_active=True,
        )
        result = risk_gate.validate(valid_long_signal, state)
        assert result.approved is False
        assert result.rejection_reason == RejectionReason.KILL_SWITCH_ACTIVE

    def test_rejects_when_daily_drawdown_breached(self, risk_gate, valid_long_signal):
        state = SystemState(
            total_capital=Decimal("10000"),
            available_capital=Decimal("9500"),
            open_positions=0,
            daily_pnl_pct=-0.05,  # Exactly at limit
            kill_switch_active=False,
        )
        result = risk_gate.validate(valid_long_signal, state)
        assert result.approved is False
        assert result.rejection_reason == RejectionReason.KILL_SWITCH_ACTIVE

    def test_rejects_when_daily_drawdown_exceeded(self, risk_gate, valid_long_signal):
        state = SystemState(
            total_capital=Decimal("10000"),
            available_capital=Decimal("9400"),
            open_positions=0,
            daily_pnl_pct=-0.06,  # Exceeds limit
            kill_switch_active=False,
        )
        result = risk_gate.validate(valid_long_signal, state)
        assert result.approved is False

    def test_approves_below_drawdown_limit(self, risk_gate, valid_long_signal):
        state = SystemState(
            total_capital=Decimal("10000"),
            available_capital=Decimal("9600"),
            open_positions=0,
            daily_pnl_pct=-0.04,  # Below limit
            kill_switch_active=False,
        )
        result = risk_gate.validate(valid_long_signal, state)
        assert result.approved is True

    def test_check_kill_switch_triggers_at_five_percent(self, risk_gate):
        assert risk_gate.check_kill_switch(-0.05) is True

    def test_check_kill_switch_triggers_above_five_percent(self, risk_gate):
        assert risk_gate.check_kill_switch(-0.06) is True

    def test_check_kill_switch_does_not_trigger_below_limit(self, risk_gate):
        assert risk_gate.check_kill_switch(-0.04) is False

    def test_check_kill_switch_does_not_trigger_at_zero(self, risk_gate):
        assert risk_gate.check_kill_switch(0.0) is False

    def test_check_kill_switch_does_not_trigger_positive_pnl(self, risk_gate):
        assert risk_gate.check_kill_switch(0.03) is False


# ── Max positions tests ───────────────────────────────────────────────────────

class TestMaxPositions:
    def test_rejects_when_max_positions_reached(self, risk_gate, valid_long_signal):
        state = SystemState(
            total_capital=Decimal("10000"),
            available_capital=Decimal("7000"),
            open_positions=3,  # At maximum
            daily_pnl_pct=0.0,
            kill_switch_active=False,
        )
        result = risk_gate.validate(valid_long_signal, state)
        assert result.approved is False
        assert result.rejection_reason == RejectionReason.MAX_POSITIONS_REACHED

    def test_approves_below_max_positions(self, risk_gate, valid_long_signal, healthy_state):
        healthy_state.open_positions = 2  # One below maximum
        result = risk_gate.validate(valid_long_signal, healthy_state)
        assert result.approved is True

    def test_approves_at_zero_positions(self, risk_gate, valid_long_signal, healthy_state):
        result = risk_gate.validate(valid_long_signal, healthy_state)
        assert result.approved is True


# ── Signal validity tests ─────────────────────────────────────────────────────

class TestSignalValidity:
    def test_rejects_missing_entry_price(self, risk_gate, healthy_state):
        signal = {"direction": "long", "stop_loss": Decimal("41500"), "take_profit": Decimal("43000")}
        result = risk_gate.validate(signal, healthy_state)
        assert result.approved is False
        assert result.rejection_reason == RejectionReason.INVALID_SIGNAL

    def test_rejects_missing_stop_loss(self, risk_gate, healthy_state):
        signal = {"direction": "long", "entry_price": Decimal("42000"), "take_profit": Decimal("43000")}
        result = risk_gate.validate(signal, healthy_state)
        assert result.approved is False

    def test_rejects_missing_take_profit(self, risk_gate, healthy_state):
        signal = {"direction": "long", "entry_price": Decimal("42000"), "stop_loss": Decimal("41500")}
        result = risk_gate.validate(signal, healthy_state)
        assert result.approved is False

    def test_rejects_entry_equals_stop_loss(self, risk_gate, healthy_state):
        signal = {
            "direction": "long",
            "entry_price": Decimal("42000"),
            "stop_loss": Decimal("42000"),  # Same as entry
            "take_profit": Decimal("43000"),
        }
        result = risk_gate.validate(signal, healthy_state)
        assert result.approved is False
        assert result.rejection_reason == RejectionReason.INVALID_SIGNAL


# ── Risk/reward tests ─────────────────────────────────────────────────────────

class TestRiskReward:
    def test_rejects_rr_below_minimum(self, risk_gate, healthy_state):
        signal = {
            "direction": "long",
            "entry_price": Decimal("42000"),
            "stop_loss": Decimal("41500"),   # $500 risk
            "take_profit": Decimal("42600"), # $600 reward → 1:1.2 — below minimum
        }
        result = risk_gate.validate(signal, healthy_state)
        assert result.approved is False
        assert result.rejection_reason == RejectionReason.INVALID_RISK_REWARD

    def test_approves_rr_at_one_point_five(self, risk_gate, healthy_state):
        signal = {
            "direction": "long",
            "entry_price": Decimal("42000"),
            "stop_loss": Decimal("41500"),   # $500 risk
            "take_profit": Decimal("42750"), # $750 reward → 1:1.5 — at minimum
        }
        result = risk_gate.validate(signal, healthy_state)
        assert result.approved is True

    def test_approves_rr_at_two(self, risk_gate, valid_long_signal, healthy_state):
        result = risk_gate.validate(valid_long_signal, healthy_state)
        assert result.approved is True


# ── Position sizing tests ─────────────────────────────────────────────────────

class TestPositionSizing:
    def test_position_size_respects_one_percent_rule(self, risk_gate, valid_long_signal, healthy_state):
        """With $10,000 capital and $500 risk distance, position should be 0.2 BTC."""
        result = risk_gate.validate(valid_long_signal, healthy_state)
        assert result.approved is True
        # Max risk = $10,000 * 1% = $100
        # Risk distance = $42,000 - $41,500 = $500
        # Position size = $100 / $500 = 0.2 BTC
        assert result.position_size == Decimal("0.20000000")
        assert result.risk_amount == Decimal("100.00")

    def test_rejects_insufficient_capital(self, risk_gate, healthy_state):
        signal = {
            "direction": "long",
            "entry_price": Decimal("42000"),
            "stop_loss": Decimal("41999"),   # $1 risk distance → huge position size
            "take_profit": Decimal("42002"), # 1:2 R/R
        }
        state = SystemState(
            total_capital=Decimal("10000"),
            available_capital=Decimal("100"),  # Very low available capital
            open_positions=0,
            daily_pnl_pct=0.0,
        )
        result = risk_gate.validate(signal, state)
        assert result.approved is False
        assert result.rejection_reason == RejectionReason.INSUFFICIENT_CAPITAL


# ── Short signal tests ────────────────────────────────────────────────────────

class TestShortSignals:
    def test_validates_valid_short_signal(self, risk_gate, valid_short_signal, healthy_state):
        result = risk_gate.validate(valid_short_signal, healthy_state)
        assert result.approved is True

    def test_position_size_correct_for_short(self, risk_gate, valid_short_signal, healthy_state):
        result = risk_gate.validate(valid_short_signal, healthy_state)
        # Max risk = $10,000 * 1% = $100
        # Risk distance = |$2,200 - $2,300| = $100
        # Position size = $100 / $100 = 1.0 ETH
        assert result.approved is True
        assert result.risk_amount == Decimal("100.00")
