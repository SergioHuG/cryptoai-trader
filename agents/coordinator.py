"""
Signal Coordinator — agents/coordinator.py

Wires KrakenClient → TechnicalAnalyst → RiskGate into a single pipeline.
This is the central orchestrator for signal generation.

Flow per symbol:
  1. Fetch latest candles from Kraken
  2. Run technical analysis → TechnicalSignal | None
  3. If signal exists, validate through RiskGate
  4. Return CoordinatorResult with full audit trail

Usage:
    coordinator = SignalCoordinator(system_state=state)
    results = coordinator.run_pipeline()
    approved = [r for r in results if r.risk_approved]
"""
import logging
from dataclasses import dataclass
from decimal import Decimal

from data.kraken import KrakenClient, SUPPORTED_SYMBOLS, DEFAULT_TIMEFRAME
from agents.technical import TechnicalAnalyst, TechnicalSignal
from agents.risk import RiskGate, RiskValidationResult, SystemState, risk_gate

logger = logging.getLogger(__name__)

# Candles fetched per evaluation — enough for all indicators + buffer
# MIN_CANDLES=35, we fetch 100 to give indicators warm-up room
CANDLES_TO_FETCH = 100


# ── Output model ──────────────────────────────────────────────────────────────

@dataclass
class CoordinatorResult:
    """
    Full audit trail for one symbol evaluation.
    Whether approved or rejected, every field is populated for logging.
    """
    symbol: str
    timeframe: str
    signal: TechnicalSignal | None
    risk_approved: bool
    position_size: Decimal | None = None
    risk_amount: Decimal | None = None
    rejection_reason: str | None = None
    rejection_message: str | None = None
    error: str | None = None

    @property
    def has_signal(self) -> bool:
        return self.signal is not None

    def __repr__(self) -> str:
        if self.error:
            return f"CoordinatorResult({self.symbol} ERROR: {self.error})"
        if not self.has_signal:
            return f"CoordinatorResult({self.symbol} NO_SIGNAL)"
        status = "APPROVED" if self.risk_approved else f"REJECTED({self.rejection_reason})"
        return f"CoordinatorResult({self.symbol} {status})"


# ── Coordinator ───────────────────────────────────────────────────────────────

class SignalCoordinator:
    """
    Orchestrates the full signal pipeline for all supported symbols.
    All dependencies are injected — fully testable without real API calls.
    """

    def __init__(
        self,
        system_state: SystemState,
        client: KrakenClient | None = None,
        analyst: TechnicalAnalyst | None = None,
        risk: RiskGate | None = None,
        timeframe: str = DEFAULT_TIMEFRAME,
        candles_to_fetch: int = CANDLES_TO_FETCH,
    ):
        self._client = client or KrakenClient()
        self._analyst = analyst or TechnicalAnalyst()
        self._risk = risk or risk_gate
        self._system_state = system_state
        self._timeframe = timeframe
        self._candles_to_fetch = candles_to_fetch

    def evaluate_symbol(self, symbol: str) -> CoordinatorResult:
        """
        Run the full pipeline for a single symbol.

        Returns a CoordinatorResult with the full audit trail regardless
        of outcome — no signal, rejected, or approved.
        """
        base = dict(symbol=symbol, timeframe=self._timeframe)

        # ── Step 1: Fetch candles ─────────────────────────────────────────────
        try:
            candles = self._client.fetch_candles(
                symbol=symbol,
                timeframe=self._timeframe,
                limit=self._candles_to_fetch,
            )
        except Exception as e:
            logger.error("Failed to fetch candles for %s: %s", symbol, e)
            return CoordinatorResult(
                **base,
                signal=None,
                risk_approved=False,
                error=f"Candle fetch failed: {e}",
            )

        # ── Step 2: Technical analysis ────────────────────────────────────────
        try:
            signal = self._analyst.analyze(candles)
        except Exception as e:
            logger.error("Technical analysis failed for %s: %s", symbol, e)
            return CoordinatorResult(
                **base,
                signal=None,
                risk_approved=False,
                error=f"Technical analysis failed: {e}",
            )

        if signal is None:
            logger.debug("No signal for %s", symbol)
            return CoordinatorResult(
                **base,
                signal=None,
                risk_approved=False,
            )

        # ── Step 3: Risk gate ─────────────────────────────────────────────────
        try:
            risk_result = self._risk.validate(
                signal=self._signal_to_dict(signal),
                system_state=self._system_state,
            )
        except Exception as e:
            logger.error("Risk gate failed for %s: %s", symbol, e)
            return CoordinatorResult(
                **base,
                signal=signal,
                risk_approved=False,
                error=f"Risk gate failed: {e}",
            )

        if risk_result.approved:
            logger.info(
                "Signal APPROVED for %s: size=%s risk=%s",
                symbol, risk_result.position_size, risk_result.risk_amount,
            )
            return CoordinatorResult(
                **base,
                signal=signal,
                risk_approved=True,
                position_size=risk_result.position_size,
                risk_amount=risk_result.risk_amount,
            )

        logger.info(
            "Signal REJECTED for %s: %s",
            symbol, risk_result.rejection_reason,
        )
        return CoordinatorResult(
            **base,
            signal=signal,
            risk_approved=False,
            rejection_reason=risk_result.rejection_reason.value
                if risk_result.rejection_reason else None,
            rejection_message=risk_result.rejection_message,
        )

    def run_pipeline(self) -> list[CoordinatorResult]:
        """
        Evaluate all supported symbols and return all results.
        Caller can filter to approved signals: [r for r in results if r.risk_approved]
        """
        results = []
        for symbol in SUPPORTED_SYMBOLS:
            result = self.evaluate_symbol(symbol)
            results.append(result)
            logger.debug("Pipeline result: %s", result)
        return results

    def update_system_state(self, system_state: SystemState) -> None:
        """
        Update system state between pipeline runs.
        Call this after each trade close or P&L update.
        """
        self._system_state = system_state

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _signal_to_dict(signal: TechnicalSignal) -> dict:
        """
        Convert TechnicalSignal to the dict format RiskGate.validate() expects.
        """
        return {
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "direction": signal.direction,
        }
