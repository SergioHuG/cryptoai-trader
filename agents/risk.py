"""
Risk Management Gate — agents/risk.py

CRITICAL: This module is the final gate before any trade execution.
No order reaches the execution layer without passing through here.

The constants below are HARDCODED. They are never:
- Configurable at runtime
- Overridable by other agents
- Changed without explicit human instruction
"""
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum

# ── Hardcoded Risk Constants ──────────────────────────────────────────────────
# These values were set during system design and are non-negotiable.
# Changing them requires: (1) human instruction, (2) test updates, (3) documented ADR update.

MAX_RISK_PER_TRADE_PCT: float = 0.01          # 1% of total capital per trade
MAX_CONCURRENT_POSITIONS: int = 3              # Maximum open positions at once
DAILY_DRAWDOWN_KILL_SWITCH_PCT: float = 0.05  # 5% daily loss → full system halt
SIGNAL_TIMEOUT_SECONDS: int = 90              # Auto-cancel if no human response


# ── Data Models ───────────────────────────────────────────────────────────────

class RejectionReason(Enum):
    KILL_SWITCH_ACTIVE = "kill_switch_active"
    MAX_POSITIONS_REACHED = "max_positions_reached"
    RISK_TOO_HIGH = "risk_too_high"
    INVALID_RISK_REWARD = "invalid_risk_reward"
    INSUFFICIENT_CAPITAL = "insufficient_capital"
    INVALID_SIGNAL = "invalid_signal"


@dataclass
class RiskValidationResult:
    approved: bool
    position_size: Decimal | None = None
    risk_amount: Decimal | None = None
    rejection_reason: RejectionReason | None = None
    rejection_message: str | None = None


@dataclass
class SystemState:
    total_capital: Decimal
    available_capital: Decimal
    open_positions: int
    daily_pnl_pct: float
    kill_switch_active: bool = False


# ── Risk Gate ─────────────────────────────────────────────────────────────────

class RiskGate:
    """
    The single point of truth for all risk decisions.
    Every trade signal passes through validate() before execution.
    """

    def validate(
        self,
        signal: dict,
        system_state: SystemState,
    ) -> RiskValidationResult:
        """
        Validate a trade signal against all risk rules.
        Returns approved=True only if ALL checks pass.
        """

        # Check 1: Kill switch — if active, reject everything
        if system_state.kill_switch_active:
            return RiskValidationResult(
                approved=False,
                rejection_reason=RejectionReason.KILL_SWITCH_ACTIVE,
                rejection_message="System kill switch is active. Manual reset required.",
            )

        # Check 2: Daily drawdown — halt if breached
        if system_state.daily_pnl_pct <= -DAILY_DRAWDOWN_KILL_SWITCH_PCT:
            return RiskValidationResult(
                approved=False,
                rejection_reason=RejectionReason.KILL_SWITCH_ACTIVE,
                rejection_message=(
                    f"Daily drawdown limit reached: {system_state.daily_pnl_pct:.1%}. "
                    f"Limit is {DAILY_DRAWDOWN_KILL_SWITCH_PCT:.1%}."
                ),
            )

        # Check 3: Maximum concurrent positions
        if system_state.open_positions >= MAX_CONCURRENT_POSITIONS:
            return RiskValidationResult(
                approved=False,
                rejection_reason=RejectionReason.MAX_POSITIONS_REACHED,
                rejection_message=(
                    f"Maximum concurrent positions reached: "
                    f"{system_state.open_positions}/{MAX_CONCURRENT_POSITIONS}."
                ),
            )

        # Check 4: Signal validity
        entry = signal.get("entry_price")
        stop_loss = signal.get("stop_loss")
        take_profit = signal.get("take_profit")
        direction = signal.get("direction")

        if not all([entry, stop_loss, take_profit, direction]):
            return RiskValidationResult(
                approved=False,
                rejection_reason=RejectionReason.INVALID_SIGNAL,
                rejection_message="Signal is missing required fields.",
            )

        entry = Decimal(str(entry))
        stop_loss = Decimal(str(stop_loss))
        take_profit = Decimal(str(take_profit))

        # Check 5: Risk/reward minimum 1:1.5
        risk_distance = abs(entry - stop_loss)
        reward_distance = abs(take_profit - entry)

        if risk_distance == 0:
            return RiskValidationResult(
                approved=False,
                rejection_reason=RejectionReason.INVALID_SIGNAL,
                rejection_message="Entry price equals stop loss — invalid signal.",
            )

        rr_ratio = reward_distance / risk_distance
        if rr_ratio < Decimal("1.5"):
            return RiskValidationResult(
                approved=False,
                rejection_reason=RejectionReason.INVALID_RISK_REWARD,
                rejection_message=f"Risk/reward ratio {rr_ratio:.2f} is below minimum 1.5.",
            )

        # Check 6: Position size calculation
        max_risk_amount = system_state.total_capital * Decimal(str(MAX_RISK_PER_TRADE_PCT))
        position_size = max_risk_amount / risk_distance

        # Check 7: Sufficient capital
        required_capital = position_size * entry
        if required_capital > system_state.available_capital:
            return RiskValidationResult(
                approved=False,
                rejection_reason=RejectionReason.INSUFFICIENT_CAPITAL,
                rejection_message=(
                    f"Required capital ${required_capital:.2f} exceeds "
                    f"available ${system_state.available_capital:.2f}."
                ),
            )

        # All checks passed
        return RiskValidationResult(
            approved=True,
            position_size=position_size.quantize(Decimal("0.00000001")),
            risk_amount=max_risk_amount.quantize(Decimal("0.01")),
        )

    def check_kill_switch(self, daily_pnl_pct: float) -> bool:
        """
        Returns True if the kill switch should be activated.
        Called after every trade close and P&L update.
        """
        return daily_pnl_pct <= -DAILY_DRAWDOWN_KILL_SWITCH_PCT


# ── Singleton instance ─────────────────────────────────────────────────────────
risk_gate = RiskGate()
