"""
Discretionary Signal Ingestor — notifications/discretionary.py

Allows the portfolio manager to submit directional trade ideas with a
conviction score. These are stored with provenance='human' and fed into
the meta-labeling layer (Phase 3) as additional features alongside
model-sourced signals.

Conviction range: [0.0, 1.0]. Direction: "long" | "short".
"""
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import update as sa_update
from sqlalchemy.orm import Session

from database.models import DiscretionarySignalModel

logger = logging.getLogger(__name__)

VALID_DIRECTIONS = frozenset({"long", "short"})


@dataclass(frozen=True)
class DiscretionarySignal:
    """Read model returned by get_pending()."""
    signal_ref: str
    symbol: str
    direction: str
    conviction: Decimal
    provenance: str
    submitted_at: datetime
    meta_label_consumed: bool


class DiscretionarySignalIngestor:
    """
    Persists human-sourced trade ideas and serves them to the meta-labeling
    layer. Inject the SQLAlchemy Session; caller controls transactions.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def submit(
        self,
        symbol: str,
        direction: str,
        conviction: Decimal,
    ) -> str:
        """
        Persist a human-sourced directional idea.

        Args:
            symbol:    Trading pair, e.g. "BTC/USD".
            direction: "long" or "short" — raises ValueError otherwise.
            conviction: In [0.0, 1.0] — raises ValueError otherwise.

        Returns:
            signal_ref — a UUID string identifying this signal.
        """
        if direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be 'long' or 'short', got '{direction}'."
            )

        if not (Decimal("0") <= conviction <= Decimal("1")):
            raise ValueError(
                f"conviction must be in [0.0, 1.0], got {conviction}."
            )

        signal_ref = str(uuid.uuid4())

        row = DiscretionarySignalModel(
            signal_ref=signal_ref,
            symbol=symbol,
            direction=direction,
            conviction=conviction,
            provenance="human",
            submitted_at=datetime.now(timezone.utc),
            meta_label_consumed=False,
        )

        self._session.add(row)
        self._session.commit()

        logger.info(
            "Discretionary signal submitted: %s %s conviction=%.2f ref=%s",
            symbol,
            direction,
            float(conviction),
            signal_ref,
        )

        return signal_ref

    def get_pending(
        self,
        symbol: str | None = None,
    ) -> list[DiscretionarySignal]:
        """
        Return all unconsumed human signals, optionally filtered by symbol.
        Used by the meta-labeling layer to fetch inputs for Phase 3.
        """
        from sqlalchemy import select

        query = (
            select(DiscretionarySignalModel)
            .where(DiscretionarySignalModel.meta_label_consumed.is_(False))
        )
        if symbol is not None:
            query = query.where(DiscretionarySignalModel.symbol == symbol)

        rows = self._session.execute(query).scalars().all()

        return [
            DiscretionarySignal(
                signal_ref=r.signal_ref,
                symbol=r.symbol,
                direction=r.direction,
                conviction=Decimal(str(r.conviction)),
                provenance=r.provenance,
                submitted_at=r.submitted_at,
                meta_label_consumed=r.meta_label_consumed,
            )
            for r in rows
        ]

    def mark_consumed(self, signal_ref: str) -> None:
        """
        Mark a signal as consumed by the meta-labeling layer.
        Uses UPDATE — does not add a new row.
        """
        self._session.execute(
            sa_update(DiscretionarySignalModel)
            .where(DiscretionarySignalModel.signal_ref == signal_ref)
            .values(
                meta_label_consumed=True,
                consumed_at=datetime.now(timezone.utc),
            )
        )
        self._session.commit()
