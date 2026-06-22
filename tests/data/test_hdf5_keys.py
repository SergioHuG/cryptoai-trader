"""Acceptance tests for shared HDF5 key normalizers (Task 8b).

symbol_to_key/threshold_to_key are extracted out of data/bars.py into this
new public, shared module because labels live in the *same* .h5 file as
bars (sibling /{SYM_KEY}/thr_{N}/labels/cfg_{hash} group) -- the normalizer
must be a single source of truth rather than duplicated across data/ and
research/labels/.
"""
from decimal import Decimal

from data.hdf5_keys import symbol_to_key, threshold_to_key


class TestSymbolToKey:
    def test_slash_separated_symbol(self):
        assert symbol_to_key("BTC/USD") == "BTC_USD"

    def test_already_underscore_separated(self):
        assert symbol_to_key("ETH_USD") == "ETH_USD"

    def test_lowercase_symbol_is_uppercased(self):
        assert symbol_to_key("btc/usd") == "BTC_USD"


class TestThresholdToKey:
    def test_integer_threshold(self):
        assert threshold_to_key(Decimal("1000000")) == "thr_1000000"

    def test_decimal_threshold_is_int_truncated(self):
        assert threshold_to_key(Decimal("500000.99")) == "thr_500000"
