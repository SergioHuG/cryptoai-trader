"""
Tests for agents/coordinator.py

All dependencies are mocked — no real API calls, no database, no exchange.
Tests verify the pipeline logic and audit trail, not the underlying modules
(those are tested in their own suites).
"""
import pytest
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from data.kraken import Candle
from agents.technical import TechnicalSignal
from agents.risk import (
    RiskValidationResult,
    RejectionReason,
    SystemState,
)
from agents.coordinator import SignalCoordinator, CoordinatorResult, CANDLES_TO_FETCH


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def system_state():
    return SystemState(
        total_capital=Decimal("10000"),
        available_capital=Decimal("10000"),
        open_positions=0,
        daily_pnl_pct=0.0,
        kill_switch_active=False,
    )


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.fetch_candles.return_value = [
        Candle(
            symbol="BTC/USD",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i),
            open=Decimal("42000"),
            high=Decimal("42100"),
            low=Decimal("41900"),
            close=Decimal("42000"),
            volume=Decimal("100"),
            timeframe="1h",
        )
        for i in range(CANDLES_TO_FETCH)
    ]
    return client


@pytest.fixture
def mock_analyst():
    return MagicMock()


@pytest.fixture
def mock_risk():
    return MagicMock()


@pytest.fixture
def sample_signal():
    return TechnicalSignal(
        symbol="BTC/USD",
        timeframe="1h",
        direction="long",
        entry_price=Decimal("42000"),
        stop_loss=Decimal("41000"),
        take_profit=Decimal("44000"),
        risk_reward=Decimal("2.0"),
        ema_fast=Decimal("42100"),
        ema_slow=Decimal("41900"),
        rsi=Decimal("52.00"),
        volume_ratio=Decimal("1.5"),
        confidence="high",
    )


@pytest.fixture
def approved_risk_result():
    return RiskValidationResult(
        approved=True,
        position_size=Decimal("0.09000000"),
        risk_amount=Decimal("100.00"),
    )


@pytest.fixture
def rejected_risk_result():
    return RiskValidationResult(
        approved=False,
        rejection_reason=RejectionReason.MAX_POSITIONS_REACHED,
        rejection_message="Maximum concurrent positions reached: 3/3.",
    )


@pytest.fixture
def coordinator(system_state, mock_client, mock_analyst, mock_risk):
    return SignalCoordinator(
        system_state=system_state,
        client=mock_client,
        analyst=mock_analyst,
        risk=mock_risk,
    )


# ── CoordinatorResult ─────────────────────────────────────────────────────────

class TestCoordinatorResult:
    def test_has_signal_true_when_signal_present(self, sample_signal):
        result = CoordinatorResult(
            symbol="BTC/USD",
            timeframe="1h",
            signal=sample_signal,
            risk_approved=True,
        )
        assert result.has_signal is True

    def test_has_signal_false_when_no_signal(self):
        result = CoordinatorResult(
            symbol="BTC/USD",
            timeframe="1h",
            signal=None,
            risk_approved=False,
        )
        assert result.has_signal is False

    def test_repr_approved(self, sample_signal):
        result = CoordinatorResult(
            symbol="BTC/USD",
            timeframe="1h",
            signal=sample_signal,
            risk_approved=True,
        )
        assert "APPROVED" in repr(result)

    def test_repr_no_signal(self):
        result = CoordinatorResult(
            symbol="BTC/USD",
            timeframe="1h",
            signal=None,
            risk_approved=False,
        )
        assert "NO_SIGNAL" in repr(result)

    def test_repr_error(self):
        result = CoordinatorResult(
            symbol="BTC/USD",
            timeframe="1h",
            signal=None,
            risk_approved=False,
            error="Connection refused",
        )
        assert "ERROR" in repr(result)


# ── evaluate_symbol — happy path ──────────────────────────────────────────────

class TestEvaluateSymbolApproved:
    def test_returns_approved_result(
        self, coordinator, mock_analyst, mock_risk, sample_signal, approved_risk_result
    ):
        mock_analyst.analyze.return_value = sample_signal
        mock_risk.validate.return_value = approved_risk_result

        result = coordinator.evaluate_symbol("BTC/USD")

        assert result.risk_approved is True
        assert result.signal is sample_signal
        assert result.position_size == approved_risk_result.position_size
        assert result.risk_amount == approved_risk_result.risk_amount
        assert result.error is None

    def test_passes_correct_dict_to_risk_gate(
        self, coordinator, mock_analyst, mock_risk, sample_signal, approved_risk_result
    ):
        mock_analyst.analyze.return_value = sample_signal
        mock_risk.validate.return_value = approved_risk_result

        coordinator.evaluate_symbol("BTC/USD")

        call_kwargs = mock_risk.validate.call_args
        signal_dict = call_kwargs.kwargs["signal"]
        assert signal_dict["entry_price"] == sample_signal.entry_price
        assert signal_dict["stop_loss"] == sample_signal.stop_loss
        assert signal_dict["take_profit"] == sample_signal.take_profit
        assert signal_dict["direction"] == sample_signal.direction

    def test_fetches_candles_with_correct_symbol(
        self, coordinator, mock_client, mock_analyst, mock_risk,
        sample_signal, approved_risk_result
    ):
        mock_analyst.analyze.return_value = sample_signal
        mock_risk.validate.return_value = approved_risk_result

        coordinator.evaluate_symbol("ETH/USD")

        call_kwargs = mock_client.fetch_candles.call_args
        assert call_kwargs.kwargs["symbol"] == "ETH/USD"


# ── evaluate_symbol — rejection paths ────────────────────────────────────────

class TestEvaluateSymbolRejected:
    def test_no_signal_returns_unapproved(
        self, coordinator, mock_analyst, mock_risk
    ):
        mock_analyst.analyze.return_value = None

        result = coordinator.evaluate_symbol("BTC/USD")

        assert result.risk_approved is False
        assert result.signal is None
        assert result.error is None
        mock_risk.validate.assert_not_called()

    def test_risk_rejected_returns_unapproved(
        self, coordinator, mock_analyst, mock_risk, sample_signal, rejected_risk_result
    ):
        mock_analyst.analyze.return_value = sample_signal
        mock_risk.validate.return_value = rejected_risk_result

        result = coordinator.evaluate_symbol("BTC/USD")

        assert result.risk_approved is False
        assert result.signal is sample_signal
        assert result.rejection_reason == RejectionReason.MAX_POSITIONS_REACHED.value
        assert result.rejection_message is not None

    def test_candle_fetch_error_returns_error_result(
        self, coordinator, mock_client, mock_analyst
    ):
        mock_client.fetch_candles.side_effect = Exception("Connection refused")

        result = coordinator.evaluate_symbol("BTC/USD")

        assert result.risk_approved is False
        assert result.error is not None
        assert "Connection refused" in result.error
        mock_analyst.analyze.assert_not_called()

    def test_analysis_error_returns_error_result(
        self, coordinator, mock_analyst, mock_risk
    ):
        mock_analyst.analyze.side_effect = Exception("Insufficient candles")

        result = coordinator.evaluate_symbol("BTC/USD")

        assert result.risk_approved is False
        assert result.error is not None
        mock_risk.validate.assert_not_called()

    def test_risk_gate_error_returns_error_result(
        self, coordinator, mock_analyst, mock_risk, sample_signal
    ):
        mock_analyst.analyze.return_value = sample_signal
        mock_risk.validate.side_effect = Exception("Unexpected risk error")

        result = coordinator.evaluate_symbol("BTC/USD")

        assert result.risk_approved is False
        assert result.signal is sample_signal
        assert result.error is not None


# ── run_pipeline ──────────────────────────────────────────────────────────────

class TestRunPipeline:
    def test_evaluates_all_supported_symbols(
        self, coordinator, mock_analyst, mock_risk
    ):
        mock_analyst.analyze.return_value = None

        results = coordinator.run_pipeline()

        assert len(results) == 2
        symbols = {r.symbol for r in results}
        assert "BTC/USD" in symbols
        assert "ETH/USD" in symbols

    def test_returns_all_results_not_just_approved(
        self, coordinator, mock_analyst, mock_risk, sample_signal, approved_risk_result
    ):
        mock_analyst.analyze.return_value = sample_signal
        mock_risk.validate.return_value = approved_risk_result

        results = coordinator.run_pipeline()

        assert len(results) == 2

    def test_one_failure_does_not_stop_pipeline(
        self, coordinator, mock_client, mock_analyst
    ):
        def fetch_side_effect(**kwargs):
            if kwargs.get("symbol") == "BTC/USD":
                raise Exception("BTC fetch failed")
            return mock_client.fetch_candles.return_value

        mock_client.fetch_candles.side_effect = fetch_side_effect
        mock_analyst.analyze.return_value = None

        results = coordinator.run_pipeline()

        assert len(results) == 2
        btc = next(r for r in results if r.symbol == "BTC/USD")
        eth = next(r for r in results if r.symbol == "ETH/USD")
        assert btc.error is not None
        assert eth.error is None


# ── update_system_state ───────────────────────────────────────────────────────

class TestUpdateSystemState:
    def test_new_state_is_used_on_next_evaluation(
        self, coordinator, mock_analyst, mock_risk, sample_signal
    ):
        mock_analyst.analyze.return_value = sample_signal
        mock_risk.validate.return_value = RiskValidationResult(approved=True,
            position_size=Decimal("0.1"), risk_amount=Decimal("100"))

        new_state = SystemState(
            total_capital=Decimal("20000"),
            available_capital=Decimal("20000"),
            open_positions=1,
            daily_pnl_pct=0.01,
            kill_switch_active=False,
        )
        coordinator.update_system_state(new_state)
        coordinator.evaluate_symbol("BTC/USD")

        call_kwargs = mock_risk.validate.call_args
        assert call_kwargs.kwargs["system_state"].total_capital == Decimal("20000")


# ── signal_to_dict ────────────────────────────────────────────────────────────

class TestSignalToDict:
    def test_all_required_keys_present(self, sample_signal):
        result = SignalCoordinator._signal_to_dict(sample_signal)
        assert "entry_price" in result
        assert "stop_loss" in result
        assert "take_profit" in result
        assert "direction" in result

    def test_values_match_signal(self, sample_signal):
        result = SignalCoordinator._signal_to_dict(sample_signal)
        assert result["entry_price"] == sample_signal.entry_price
        assert result["direction"] == sample_signal.direction
