"""
research/seam.py — Research / Live Stack boundary

This is the ONLY file that agents/ and execution/ are permitted to import
from the research/ package. Everything else inside research/ is off-limits
to the live stack.

Exports one type: SignalPacket

Design constraints (locked):
  - side: int   —  1 for long, -1 for short (keeps arithmetic clean)
  - confidence: Decimal  —  model output, 0.0000–1.0000
  - metadata: dict[str, Decimal]  —  extensible carrier for research outputs
      Current keys:
          'ewma_vol'   — EWMA std dev of returns (volatility.py → RiskGate)
      Future keys added here as research layer grows; live stack reads by key,
      never by position, so additions are non-breaking.

A CI import-boundary test (tests/test_import_boundary.py) asserts that no
module under agents/ or execution/ imports from research/ except through
this file.
"""
from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(frozen=True)
class SignalPacket:
    """
    Typed carrier crossing the research → live boundary.

    Attributes:
        side:       1 = long, -1 = short.
        confidence: Model confidence in [0.0, 1.0].
        metadata:   Arbitrary Decimal-valued outputs from the research layer.
                    The live stack reads keys by name; never assume ordering.

    Usage (live stack side):
        from research.seam import SignalPacket

        packet = SignalPacket(
            side=1,
            confidence=Decimal("0.82"),
            metadata={"ewma_vol": Decimal("0.0152")},
        )
        vol = packet.metadata.get("ewma_vol")  # Decimal or None — always .get()
    """
    side: int
    confidence: Decimal
    metadata: dict[str, Decimal] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.side not in (1, -1):
            raise ValueError(
                f"SignalPacket.side must be 1 (long) or -1 (short), got {self.side!r}."
            )
        if not (Decimal("0") <= self.confidence <= Decimal("1")):
            raise ValueError(
                f"SignalPacket.confidence must be in [0.0, 1.0], got {self.confidence!r}."
            )
        for key, val in self.metadata.items():
            if not isinstance(val, Decimal):
                raise TypeError(
                    f"SignalPacket.metadata values must be Decimal; "
                    f"got {type(val).__name__!r} for key {key!r}."
                )


__all__ = ["SignalPacket"]
