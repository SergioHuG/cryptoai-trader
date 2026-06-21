"""Triple-barrier labeling primitives (AFML Ch.3).

The vertical barrier is measured in *bars forward* (Decision 1), a deliberate
deviation from AFML's wall-clock ``addVerticalBarrier`` to stay coherent with
information-driven dollar-bar sampling. Path-walk and labeling use *simple*
returns (AFML Snippets 3.2 / 3.5).
"""
from __future__ import annotations

from enum import IntEnum


class Barrier(IntEnum):
    """Which barrier bound an event first.

    Values are chosen to align with ``bin``'s sign in the symmetric case while
    remaining semantically distinct, and to store cleanly as ``int8``.
    """

    LOWER = -1      # stop-loss / lower horizontal touched first
    VERTICAL = 0    # vertical barrier (timeout) bound first
    UPPER = 1       # profit-take / upper horizontal touched first
