"""
Tests for research/seam.py

SignalPacket is a frozen dataclass — tests verify construction,
validation, and that the type is the sole export of the module.
"""
from decimal import Decimal

import pytest

from research.seam import SignalPacket


# ── Construction ──────────────────────────────────────────────────────────────

def test_long_signal_constructs() -> None:
    packet = SignalPacket(side=1, confidence=Decimal("0.82"))
    assert packet.side == 1
    assert packet.confidence == Decimal("0.82")
    assert packet.metadata == {}


def test_short_signal_constructs() -> None:
    packet = SignalPacket(side=-1, confidence=Decimal("0.65"))
    assert packet.side == -1


def test_metadata_stored_correctly() -> None:
    meta = {"ewma_vol": Decimal("0.01520000")}
    packet = SignalPacket(side=1, confidence=Decimal("0.75"), metadata=meta)
    assert packet.metadata["ewma_vol"] == Decimal("0.01520000")


def test_default_metadata_is_empty_dict() -> None:
    packet = SignalPacket(side=1, confidence=Decimal("0.50"))
    assert packet.metadata == {}


def test_is_frozen() -> None:
    """SignalPacket must be immutable — assignment raises AttributeError."""
    packet = SignalPacket(side=1, confidence=Decimal("0.70"))
    with pytest.raises(AttributeError):
        packet.side = -1  # type: ignore[misc]


# ── side validation ───────────────────────────────────────────────────────────

def test_side_zero_raises() -> None:
    with pytest.raises(ValueError, match="side"):
        SignalPacket(side=0, confidence=Decimal("0.70"))


def test_side_two_raises() -> None:
    with pytest.raises(ValueError, match="side"):
        SignalPacket(side=2, confidence=Decimal("0.70"))


def test_side_negative_two_raises() -> None:
    with pytest.raises(ValueError, match="side"):
        SignalPacket(side=-2, confidence=Decimal("0.70"))


# ── confidence validation ─────────────────────────────────────────────────────

def test_confidence_above_one_raises() -> None:
    with pytest.raises(ValueError, match="confidence"):
        SignalPacket(side=1, confidence=Decimal("1.0001"))


def test_confidence_negative_raises() -> None:
    with pytest.raises(ValueError, match="confidence"):
        SignalPacket(side=1, confidence=Decimal("-0.01"))


def test_confidence_exactly_zero_is_valid() -> None:
    packet = SignalPacket(side=1, confidence=Decimal("0"))
    assert packet.confidence == Decimal("0")


def test_confidence_exactly_one_is_valid() -> None:
    packet = SignalPacket(side=-1, confidence=Decimal("1"))
    assert packet.confidence == Decimal("1")


# ── metadata type validation ──────────────────────────────────────────────────

def test_metadata_non_decimal_value_raises() -> None:
    with pytest.raises(TypeError, match="Decimal"):
        SignalPacket(
            side=1,
            confidence=Decimal("0.80"),
            metadata={"ewma_vol": 0.015},  # float — not Decimal
        )


def test_metadata_string_value_raises() -> None:
    with pytest.raises(TypeError, match="Decimal"):
        SignalPacket(
            side=1,
            confidence=Decimal("0.80"),
            metadata={"ewma_vol": "0.015"},  # str — not Decimal
        )


def test_metadata_multiple_keys_all_decimal() -> None:
    packet = SignalPacket(
        side=1,
        confidence=Decimal("0.80"),
        metadata={
            "ewma_vol": Decimal("0.015"),
            "label_weight": Decimal("0.93"),
        },
    )
    assert len(packet.metadata) == 2


# ── metadata access pattern ───────────────────────────────────────────────────

def test_metadata_get_returns_none_for_missing_key() -> None:
    """Live stack always uses .get() — must return None, not KeyError."""
    packet = SignalPacket(side=1, confidence=Decimal("0.70"))
    assert packet.metadata.get("ewma_vol") is None


def test_metadata_get_returns_value_when_present() -> None:
    packet = SignalPacket(
        side=1,
        confidence=Decimal("0.80"),
        metadata={"ewma_vol": Decimal("0.0123")},
    )
    assert packet.metadata.get("ewma_vol") == Decimal("0.0123")


# ── module exports ────────────────────────────────────────────────────────────

def test_all_contains_only_signal_packet() -> None:
    import research.seam as seam_mod
    assert seam_mod.__all__ == ["SignalPacket"]


def test_signal_packet_is_importable_directly() -> None:
    """Verify the public import path used by the live stack works."""
    from research.seam import SignalPacket as SP
    assert SP is SignalPacket
