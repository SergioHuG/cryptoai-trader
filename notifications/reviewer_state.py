"""
Reviewer State Logger — notifications/reviewer_state.py

Persists per-decision metadata to the reviewer_sessions table and
computes deviation metrics to flag anomalous human override behaviour.

Override definition: a non-approval (cancel or timeout) on a signal
with confidence >= OVERRIDE_CONFIDENCE_THRESHOLD (0.70).

Escalation fires when today's override rate is more than 2 standard
deviations above the 90-day historical mean AND there are enough rows
to make the statistic meaningful (MIN_HISTORY_FOR_ESCALATION = 10).
"""
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database.models import ReviewerSessionModel
from notifications.types import ApprovalResult, TradeSignal

logger = logging.getLogger(__name__)

# Non-approval on a signal with confidence >= this threshold is an override.
OVERRIDE_CONFIDENCE_THRESHOLD = Decimal("0.70")

# Below this many historical rows we cannot compute a meaningful std dev.
MIN_HISTORY_FOR_ESCALATION = 10


class ReviewerStateLogger:
    """
    Records every HITL decision and monitors override-rate drift.

    Inject the SQLAlchemy Session; do not use get_session() internally
    so callers control transaction boundaries.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ── Public API ────────────────────────────────────────────────────────────

    def log_decision(self, signal: TradeSignal, result: ApprovalResult) -> None:
        """
        Persist one ReviewerSessionModel row.
        Computes deviation metrics at write time so the row is a
        self-contained audit snapshot (metrics won't drift after the fact).
        Commits the session.
        """
        now = datetime.now(timezone.utc)

        if result.approved:
            decision = "approve"
        elif result.timed_out:
            decision = "timeout"
        else:
            decision = "cancel"

        is_override = (
            not result.approved
            and signal.confidence >= OVERRIDE_CONFIDENCE_THRESHOLD
        )

        # Compute metrics before writing so the row is self-describing.
        hist_mean = self.historical_mean_override_rate()
        z_score = self._compute_z_score()
        escalation = self.should_escalate()

        # sent_at is reconstructed from the decided_at timestamp and latency.
        sent_at = now - timedelta(seconds=result.decision_latency_seconds)
        decided_at = now if decision != "timeout" else None

        row = ReviewerSessionModel(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            direction=signal.direction,
            confidence=signal.confidence,
            recommended_size=signal.position_size,
            sent_at=sent_at,
            decided_at=decided_at,
            decision=decision,
            decision_latency_seconds=Decimal(
                str(round(result.decision_latency_seconds, 3))
            ),
            hour_of_day=now.hour,
            is_override=is_override,
            historical_override_rate=hist_mean,
            deviation_z_score=z_score,
            escalation_flag=escalation,
        )

        self._session.add(row)
        self._session.commit()

        if escalation:
            logger.warning(
                "Reviewer override rate is anomalous (z=%.2f) — "
                "escalation flag set on signal %s.",
                float(z_score) if z_score is not None else 0.0,
                signal.signal_id,
            )

    def current_session_override_rate(self, window_hours: int = 24) -> Decimal:
        """
        Fraction of decisions in the last window_hours that were non-approvals.
        Returns Decimal('0') when there are no decisions in the window.
        """
        since = datetime.now(timezone.utc) - timedelta(hours=window_hours)

        rows = self._session.execute(
            select(ReviewerSessionModel.is_override)
            .where(ReviewerSessionModel.decided_at >= since)
        ).fetchall()

        if not rows:
            return Decimal("0")

        total = len(rows)
        overrides = sum(1 for r in rows if r.is_override)
        return Decimal(str(overrides)) / Decimal(str(total))

    def historical_mean_override_rate(self, window_days: int = 90) -> Decimal:
        """
        Mean of daily override rates over the past window_days.
        Returns Decimal('0') when insufficient history.
        """
        daily_rates = self._daily_rates(window_days)
        if not daily_rates:
            return Decimal("0")
        return Decimal(str(round(statistics.mean(daily_rates), 4)))

    def should_escalate(self) -> bool:
        """
        True when today's override rate is > 2 std deviations above the
        90-day historical mean AND there are >= MIN_HISTORY_FOR_ESCALATION rows.
        False in all other cases, including insufficient history.
        """
        if self._history_row_count() < MIN_HISTORY_FOR_ESCALATION:
            return False

        today_rate = self.current_session_override_rate()
        mean = self.historical_mean_override_rate()
        stddev = self._historical_stddev()

        if stddev == Decimal("0"):
            return False

        z = (today_rate - mean) / stddev
        return z > Decimal("2")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _history_row_count(self, window_days: int = 90) -> int:
        since = datetime.now(timezone.utc) - timedelta(days=window_days)
        result = self._session.execute(
            select(func.count()).select_from(ReviewerSessionModel)
            .where(ReviewerSessionModel.decided_at >= since)
        ).scalar()
        return int(result or 0)

    def _historical_stddev(self, window_days: int = 90) -> Decimal:
        daily_rates = self._daily_rates(window_days)
        if len(daily_rates) < 2:
            return Decimal("0")
        try:
            return Decimal(str(round(statistics.stdev(daily_rates), 4)))
        except statistics.StatisticsError:
            return Decimal("0")

    def _daily_rates(self, window_days: int = 90) -> list[float]:
        """
        Returns a list of per-day override rates (one float per calendar day).
        Used by both mean and stddev calculations to avoid duplicate queries.
        """
        since = datetime.now(timezone.utc) - timedelta(days=window_days)

        rows = self._session.execute(
            select(
                ReviewerSessionModel.is_override,
                ReviewerSessionModel.decided_at,
            )
            .where(ReviewerSessionModel.decided_at >= since)
            .where(ReviewerSessionModel.decided_at.is_not(None))
        ).fetchall()

        if len(rows) < MIN_HISTORY_FOR_ESCALATION:
            return []

        daily: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            day_key = row.decided_at.strftime("%Y-%m-%d")
            daily[day_key].append(row.is_override)

        return [
            sum(1 for v in decisions if v) / len(decisions)
            for decisions in daily.values()
        ]

    def _compute_z_score(self) -> Decimal | None:
        """
        Returns the z-score of today's override rate vs historical mean.
        Returns None when there is insufficient history.
        """
        if self._history_row_count() < MIN_HISTORY_FOR_ESCALATION:
            return None

        today_rate = self.current_session_override_rate()
        mean = self.historical_mean_override_rate()
        stddev = self._historical_stddev()

        if stddev == Decimal("0"):
            return None

        return (today_rate - mean) / stddev
