"""Shared HDF5 key normalizers.

Extracted out of ``data/bars.py`` because labels (``research/labels/storage.py``)
live in the *same* ``.h5`` file as bars -- a sibling
``/{SYM_KEY}/thr_{N}/labels/cfg_{hash}`` group -- so the symbol/threshold ->
HDF5-group-name convention must be a single shared source of truth rather
than duplicated across ``data/`` and ``research/labels/``.

``data/bars.py`` aliases its private ``_symbol_to_key``/``_threshold_to_key``
to these public functions, so its existing tests and behavior stay untouched.
"""
from __future__ import annotations

from decimal import Decimal

__all__ = ["symbol_to_key", "threshold_to_key"]


def symbol_to_key(symbol: str) -> str:
    """Normalise a CCXT symbol string to an HDF5-safe group name.

    Examples
    --------
    'BTC/USD' → 'BTC_USD'
    'btc/usd' → 'BTC_USD'
    """
    return symbol.replace("/", "_").upper()


def threshold_to_key(threshold: Decimal) -> str:
    """Normalise a Decimal threshold to an HDF5-safe group name.

    HDF5 group names cannot start with a digit, so we prefix 'thr_'.

    Examples
    --------
    Decimal('1000000')    → 'thr_1000000'
    Decimal('500000.99')  → 'thr_500000'  (int-truncated)
    """
    return f"thr_{int(threshold)}"
