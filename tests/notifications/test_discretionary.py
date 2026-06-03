"""
Tests for notifications/discretionary.py

The SQLAlchemy Session is a plain MagicMock. Tests verify:
  - Field persistence on submit()
  - provenance is always 'human'
  - Validation rejects invalid direction and out-of-range conviction
  - mark_consumed uses UPDATE (execute), not INSERT (add)
"""
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from notifications.discretionary import DiscretionarySignalIngestor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock()
    session.add = MagicMock()
    session.commit = MagicMock()
    session.execute = MagicMock()
    # get_pending uses scalars().all()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = []
    session.execute.return_value.scalars.return_value = mock_scalars
    return session


@pytest.fixture
def ingestor(mock_session: MagicMock) -> DiscretionarySignalIngestor:
    return DiscretionarySignalIngestor(session=mock_session)


# ── submit — return value ─────────────────────────────────────────────────────

def test_submit_returns_non_empty_string(ingestor: DiscretionarySignalIngestor) -> None:
    signal_ref = ingestor.submit("BTC/USD", "long", Decimal("0.85"))
    assert isinstance(signal_ref, str)
    assert len(signal_ref) > 0


def test_submit_returns_unique_refs(ingestor: DiscretionarySignalIngestor) -> None:
    ref1 = ingestor.submit("BTC/USD", "long", Decimal("0.80"))
    ref2 = ingestor.submit("ETH/USD", "short", Decimal("0.70"))
    assert ref1 != ref2


# ── submit — persisted fields ─────────────────────────────────────────────────

def test_submit_sets_provenance_human(
    ingestor: DiscretionarySignalIngestor,
    mock_session: MagicMock,
) -> None:
    ingestor.submit("ETH/USD", "short", Decimal("0.70"))
    row = mock_session.add.call_args[0][0]
    assert row.provenance == "human"


def test_submit_persists_symbol_direction_conviction(
    ingestor: DiscretionarySignalIngestor,
    mock_session: MagicMock,
) -> None:
    ingestor.submit("BTC/USD", "long", Decimal("0.90"))
    row = mock_session.add.call_args[0][0]
    assert row.symbol == "BTC/USD"
    assert row.direction == "long"
    assert row.conviction == Decimal("0.90")


def test_submit_sets_meta_label_consumed_false(
    ingestor: DiscretionarySignalIngestor,
    mock_session: MagicMock,
) -> None:
    ingestor.submit("BTC/USD", "long", Decimal("0.80"))
    row = mock_session.add.call_args[0][0]
    assert row.meta_label_consumed is False


def test_submit_commits_session(
    ingestor: DiscretionarySignalIngestor,
    mock_session: MagicMock,
) -> None:
    ingestor.submit("BTC/USD", "long", Decimal("0.75"))
    mock_session.commit.assert_called_once()


# ── submit — validation ───────────────────────────────────────────────────────

def test_submit_rejects_invalid_direction(ingestor: DiscretionarySignalIngestor) -> None:
    with pytest.raises(ValueError, match="direction"):
        ingestor.submit("BTC/USD", "hold", Decimal("0.80"))


def test_submit_rejects_buy_direction(ingestor: DiscretionarySignalIngestor) -> None:
    """'buy' is not a valid direction — must use 'long'."""
    with pytest.raises(ValueError, match="direction"):
        ingestor.submit("BTC/USD", "buy", Decimal("0.80"))


def test_submit_rejects_conviction_above_one(ingestor: DiscretionarySignalIngestor) -> None:
    with pytest.raises(ValueError, match="conviction"):
        ingestor.submit("BTC/USD", "long", Decimal("1.001"))


def test_submit_rejects_negative_conviction(ingestor: DiscretionarySignalIngestor) -> None:
    with pytest.raises(ValueError, match="conviction"):
        ingestor.submit("BTC/USD", "long", Decimal("-0.001"))


def test_submit_accepts_boundary_zero(ingestor: DiscretionarySignalIngestor) -> None:
    """Conviction=0.0 is a valid boundary."""
    ref = ingestor.submit("BTC/USD", "long", Decimal("0.0"))
    assert len(ref) > 0


def test_submit_accepts_boundary_one(ingestor: DiscretionarySignalIngestor) -> None:
    """Conviction=1.0 is a valid boundary."""
    ref = ingestor.submit("BTC/USD", "short", Decimal("1.0"))
    assert len(ref) > 0


def test_submit_accepts_both_valid_directions(ingestor: DiscretionarySignalIngestor) -> None:
    ref_long = ingestor.submit("BTC/USD", "long", Decimal("0.7"))
    ref_short = ingestor.submit("ETH/USD", "short", Decimal("0.6"))
    assert ref_long != ref_short


# ── mark_consumed ─────────────────────────────────────────────────────────────

def test_mark_consumed_uses_execute_not_add(
    ingestor: DiscretionarySignalIngestor,
    mock_session: MagicMock,
) -> None:
    """mark_consumed must UPDATE an existing row, not INSERT a new one."""
    ingestor.mark_consumed("some-uuid-ref")
    mock_session.execute.assert_called_once()
    mock_session.add.assert_not_called()


def test_mark_consumed_commits(
    ingestor: DiscretionarySignalIngestor,
    mock_session: MagicMock,
) -> None:
    ingestor.mark_consumed("some-uuid-ref")
    mock_session.commit.assert_called_once()


# ── get_pending ───────────────────────────────────────────────────────────────

def test_get_pending_returns_empty_list_when_none(
    ingestor: DiscretionarySignalIngestor,
) -> None:
    result = ingestor.get_pending()
    assert result == []
