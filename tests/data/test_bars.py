"""Tests for data/bars.py — dollar bar constructor."""
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from data.bars import (
    Bar,
    DollarBarAccumulator,
    _bars_to_df,
    _hdf5_key,
    _read_checkpoint,
    _symbol_to_key,
    _threshold_to_key,
    _write_checkpoint,
    build_dollar_bars,
    load_dollar_bars,
)


# ===========================================================================
# Bar dataclass
# ===========================================================================

class TestBar:
    def _make_bar(self) -> Bar:
        return Bar(
            bar_start_ts=1000,
            bar_end_ts=2000,
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("90"),
            close=Decimal("105"),
            volume=Decimal("1.5"),
            dollar_volume=Decimal("157.5"),
            num_ticks=3,
        )

    def test_bar_is_frozen(self):
        bar = self._make_bar()
        with pytest.raises(Exception):  # FrozenInstanceError
            bar.close = Decimal("999")  # type: ignore[misc]

    def test_bar_field_bar_start_ts(self):
        assert self._make_bar().bar_start_ts == 1000

    def test_bar_field_bar_end_ts(self):
        assert self._make_bar().bar_end_ts == 2000

    def test_bar_field_open(self):
        assert self._make_bar().open == Decimal("100")

    def test_bar_field_high(self):
        assert self._make_bar().high == Decimal("110")

    def test_bar_field_low(self):
        assert self._make_bar().low == Decimal("90")

    def test_bar_field_close(self):
        assert self._make_bar().close == Decimal("105")

    def test_bar_field_volume(self):
        assert self._make_bar().volume == Decimal("1.5")

    def test_bar_field_dollar_volume(self):
        assert self._make_bar().dollar_volume == Decimal("157.5")

    def test_bar_field_num_ticks(self):
        assert self._make_bar().num_ticks == 3


# ===========================================================================
# Key helpers
# ===========================================================================

class TestKeyHelpers:
    def test_symbol_to_key_slash(self):
        assert _symbol_to_key("BTC/USD") == "BTC_USD"

    def test_symbol_to_key_already_underscore(self):
        assert _symbol_to_key("ETH_USD") == "ETH_USD"

    def test_symbol_to_key_lowercase(self):
        assert _symbol_to_key("btc/usd") == "BTC_USD"

    def test_threshold_to_key_integer(self):
        assert _threshold_to_key(Decimal("1000000")) == "thr_1000000"

    def test_threshold_to_key_decimal_truncated(self):
        assert _threshold_to_key(Decimal("500000.99")) == "thr_500000"

    def test_hdf5_key_btc(self):
        assert _hdf5_key("BTC/USD", Decimal("1000000")) == "/BTC_USD/thr_1000000/bars"

    def test_hdf5_key_eth(self):
        assert _hdf5_key("ETH/USD", Decimal("500000")) == "/ETH_USD/thr_500000/bars"


# ===========================================================================
# DollarBarAccumulator
# ===========================================================================

_THRESHOLD = Decimal("1000")  # $1,000 threshold for all accumulator tests


class TestDollarBarAccumulator:
    def _acc(self) -> DollarBarAccumulator:
        return DollarBarAccumulator(_THRESHOLD)

    # --- state before first ingest ---

    def test_empty_before_first_ingest(self):
        assert self._acc().is_empty is True

    def test_not_empty_after_first_ingest(self):
        acc = self._acc()
        acc.ingest(ts_ms=1000, price=Decimal("100"), amount=Decimal("1"))
        assert acc.is_empty is False

    # --- below threshold: no emission ---

    def test_ingest_below_threshold_returns_none(self):
        acc = self._acc()
        result = acc.ingest(ts_ms=1000, price=Decimal("100"), amount=Decimal("9"))
        # $900 < $1000
        assert result is None

    # --- at threshold: emit bar ---

    def test_ingest_emits_bar_at_threshold(self):
        acc = self._acc()
        bar = acc.ingest(ts_ms=2000, price=Decimal("1000"), amount=Decimal("1"))
        assert bar is not None
        assert bar.dollar_volume == Decimal("1000")

    def test_emission_resets_accumulator_to_empty(self):
        acc = self._acc()
        acc.ingest(ts_ms=2000, price=Decimal("1000"), amount=Decimal("1"))
        assert acc.is_empty is True

    # --- bar field correctness ---

    def test_bar_open_is_first_trade_price(self):
        acc = self._acc()
        # 80*7=560 + 100*5=500 = 1060 >= 1000 → bar emitted on second ingest
        acc.ingest(ts_ms=1000, price=Decimal("80"), amount=Decimal("7"))
        bar = acc.ingest(ts_ms=2000, price=Decimal("100"), amount=Decimal("5"))
        assert bar.open == Decimal("80")

    def test_bar_close_is_last_trade_price(self):
        acc = self._acc()
        acc.ingest(ts_ms=1000, price=Decimal("80"), amount=Decimal("7"))
        bar = acc.ingest(ts_ms=2000, price=Decimal("100"), amount=Decimal("5"))
        assert bar.close == Decimal("100")

    def test_bar_high_tracks_maximum(self):
        acc = self._acc()
        acc.ingest(ts_ms=1000, price=Decimal("80"), amount=Decimal("7"))
        bar = acc.ingest(ts_ms=2000, price=Decimal("100"), amount=Decimal("5"))
        assert bar.high == Decimal("100")

    def test_bar_low_tracks_minimum(self):
        acc = self._acc()
        acc.ingest(ts_ms=1000, price=Decimal("80"), amount=Decimal("7"))
        bar = acc.ingest(ts_ms=2000, price=Decimal("100"), amount=Decimal("5"))
        assert bar.low == Decimal("80")

    def test_bar_volume_sums_amounts(self):
        acc = self._acc()
        acc.ingest(ts_ms=1000, price=Decimal("100"), amount=Decimal("6"))
        bar = acc.ingest(ts_ms=2000, price=Decimal("100"), amount=Decimal("4"))
        assert bar.volume == Decimal("10")

    def test_bar_num_ticks_counts_trades(self):
        acc = self._acc()
        acc.ingest(ts_ms=1000, price=Decimal("100"), amount=Decimal("5"))
        bar = acc.ingest(ts_ms=2000, price=Decimal("100"), amount=Decimal("5"))
        assert bar.num_ticks == 2

    def test_bar_start_ts_is_first_trade(self):
        acc = self._acc()
        acc.ingest(ts_ms=5000, price=Decimal("100"), amount=Decimal("5"))
        bar = acc.ingest(ts_ms=9000, price=Decimal("100"), amount=Decimal("5"))
        assert bar.bar_start_ts == 5000

    def test_bar_end_ts_is_last_trade(self):
        acc = self._acc()
        acc.ingest(ts_ms=5000, price=Decimal("100"), amount=Decimal("5"))
        bar = acc.ingest(ts_ms=9000, price=Decimal("100"), amount=Decimal("5"))
        assert bar.bar_end_ts == 9000

    # --- multi-bar sequence ---

    def test_two_bars_emit_sequentially(self):
        acc = self._acc()
        b1_none = acc.ingest(ts_ms=1000, price=Decimal("500"), amount=Decimal("1"))
        assert b1_none is None
        b1 = acc.ingest(ts_ms=2000, price=Decimal("500"), amount=Decimal("1"))
        assert b1 is not None and b1.num_ticks == 2

        b2 = acc.ingest(ts_ms=3000, price=Decimal("1000"), amount=Decimal("1"))
        assert b2 is not None
        assert b2.bar_start_ts == 3000
        assert b2.num_ticks == 1

    def test_second_bar_has_independent_open(self):
        acc = self._acc()
        acc.ingest(ts_ms=1000, price=Decimal("500"), amount=Decimal("1"))
        acc.ingest(ts_ms=2000, price=Decimal("500"), amount=Decimal("1"))  # bar 1 emitted
        b2 = acc.ingest(ts_ms=3000, price=Decimal("1000"), amount=Decimal("1"))
        assert b2.open == Decimal("1000")

    # --- flush ---

    def test_flush_empty_returns_none(self):
        assert self._acc().flush() is None

    def test_flush_partial_bar_returns_bar(self):
        acc = self._acc()
        acc.ingest(ts_ms=1000, price=Decimal("100"), amount=Decimal("5"))
        bar = acc.flush()
        assert bar is not None
        assert bar.num_ticks == 1
        assert bar.dollar_volume == Decimal("500")

    def test_flush_end_ts_is_last_ingested_trade(self):
        acc = self._acc()
        acc.ingest(ts_ms=1000, price=Decimal("100"), amount=Decimal("5"))
        acc.ingest(ts_ms=2500, price=Decimal("100"), amount=Decimal("1"))
        bar = acc.flush()
        assert bar.bar_end_ts == 2500

    def test_flush_resets_accumulator(self):
        acc = self._acc()
        acc.ingest(ts_ms=1000, price=Decimal("100"), amount=Decimal("5"))
        acc.flush()
        assert acc.is_empty is True

    def test_flush_after_flush_returns_none(self):
        acc = self._acc()
        acc.ingest(ts_ms=1000, price=Decimal("100"), amount=Decimal("5"))
        acc.flush()
        assert acc.flush() is None

    # --- validation ---

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="positive"):
            DollarBarAccumulator(Decimal("-1"))

    def test_zero_threshold_raises(self):
        with pytest.raises(ValueError, match="positive"):
            DollarBarAccumulator(Decimal("0"))


# ===========================================================================
# _bars_to_df
# ===========================================================================

def _make_bar(start: int = 1_000, end: int = 2_000,
              price: str = "100", amount: str = "10") -> Bar:
    """Reusable Bar factory for DataFrame tests."""
    p = Decimal(price)
    a = Decimal(amount)
    return Bar(
        bar_start_ts=start,
        bar_end_ts=end,
        open=p,
        high=p + Decimal("5"),
        low=p - Decimal("5"),
        close=p,
        volume=a,
        dollar_volume=p * a,
        num_ticks=2,
    )


class TestBarsToDF:
    def test_empty_list_returns_dataframe(self):
        assert isinstance(_bars_to_df([]), pd.DataFrame)

    def test_empty_list_has_zero_rows(self):
        assert len(_bars_to_df([])) == 0

    def test_empty_df_has_expected_columns(self):
        df = _bars_to_df([])
        for col in ["bar_start_ts", "open", "high", "low", "close",
                    "volume", "dollar_volume", "num_ticks"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_single_bar_produces_one_row(self):
        assert len(_bars_to_df([_make_bar()])) == 1

    def test_index_is_datetimeindex(self):
        df = _bars_to_df([_make_bar()])
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_index_is_utc(self):
        df = _bars_to_df([_make_bar()])
        assert str(df.index.tz) == "UTC"

    def test_index_name_is_bar_end_ts(self):
        assert _bars_to_df([_make_bar()]).index.name == "bar_end_ts"

    def test_index_value_matches_bar_end_ts_ms(self):
        bar = _make_bar(end=5_000)
        df = _bars_to_df([bar])
        expected = pd.Timestamp(5_000, unit="ms", tz="UTC")
        assert df.index[0] == expected

    def test_open_column_is_float64(self):
        assert _bars_to_df([_make_bar()])["open"].dtype == "float64"

    def test_num_ticks_is_int64(self):
        assert _bars_to_df([_make_bar()])["num_ticks"].dtype == "int64"

    def test_bar_start_ts_stored_as_integer(self):
        df = _bars_to_df([_make_bar(start=1234)])
        assert df["bar_start_ts"].iloc[0] == 1234

    def test_multiple_bars_correct_row_count(self):
        bars = [_make_bar(start=i * 1000, end=(i + 1) * 1000) for i in range(5)]
        assert _bars_to_df(bars).shape[0] == 5


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

class TestCheckpoint:
    def test_read_checkpoint_no_file_returns_none(self, tmp_path: Path):
        assert _read_checkpoint(tmp_path / "nope.h5", "/X/thr_1/bars") is None

    def test_read_checkpoint_key_absent_returns_none(self, tmp_path: Path):
        store_path = tmp_path / "bars.h5"
        # Create file but write to a different key
        with pd.HDFStore(store_path, mode="w") as _:
            pass
        assert _read_checkpoint(store_path, "/missing/thr_1/bars") is None

    def test_write_then_read_roundtrips(self, tmp_path: Path):
        store_path = tmp_path / "bars.h5"
        key = _hdf5_key("BTC/USD", Decimal("1000"))
        df = _bars_to_df([_make_bar()])
        with pd.HDFStore(store_path, mode="w", complevel=5, complib="blosc") as store:
            store.append(key, df, format="table", data_columns=True)
        _write_checkpoint(store_path, key, 99_999)
        assert _read_checkpoint(store_path, key) == 99_999

    def test_checkpoint_survives_close_and_reopen(self, tmp_path: Path):
        store_path = tmp_path / "bars.h5"
        key = _hdf5_key("ETH/USD", Decimal("500000"))
        df = _bars_to_df([_make_bar()])
        with pd.HDFStore(store_path, mode="w", complevel=5, complib="blosc") as store:
            store.append(key, df, format="table", data_columns=True)
        _write_checkpoint(store_path, key, 123_456_789)
        assert _read_checkpoint(store_path, key) == 123_456_789

    def test_write_checkpoint_overwrites_previous(self, tmp_path: Path):
        store_path = tmp_path / "bars.h5"
        key = _hdf5_key("BTC/USD", Decimal("1000"))
        df = _bars_to_df([_make_bar()])
        with pd.HDFStore(store_path, mode="w", complevel=5, complib="blosc") as store:
            store.append(key, df, format="table", data_columns=True)
        _write_checkpoint(store_path, key, 111)
        _write_checkpoint(store_path, key, 222)
        assert _read_checkpoint(store_path, key) == 222


# ===========================================================================
# build_dollar_bars
# ===========================================================================

def _trade(ts_ms: int, price: str, amount: str) -> dict:
    """Minimal CCXT trade dict."""
    return {"timestamp": ts_ms, "price": price, "amount": amount}


def _make_exchange(pages: list[list[dict]]) -> Any:
    """Mock exchange whose fetch_trades() yields pages in sequence."""
    exchange = MagicMock()
    exchange.fetch_trades.side_effect = pages
    return exchange


class TestBuildDollarBars:
    _THRESHOLD = Decimal("1000")
    _SYMBOL = "BTC/USD"

    def _build(
        self,
        tmp_path: Path,
        pages: list[list[dict]],
        *,
        since_ms: int | None = None,
        rate_limit_delay: float = 0.0,
        page_size: int = 1000,
        checkpoint_interval: int = 1,
        store_name: str = "bars.h5",
    ) -> tuple[pd.DataFrame, Any, list[float], Path]:
        sleep_calls: list[float] = []

        def fake_sleep(s: float) -> None:
            sleep_calls.append(s)

        exchange = _make_exchange(pages)
        store_path = tmp_path / store_name
        df = build_dollar_bars(
            self._SYMBOL,
            self._THRESHOLD,
            store_path,
            exchange=exchange,
            since_ms=since_ms,
            rate_limit_delay=rate_limit_delay,
            page_size=page_size,
            checkpoint_interval=checkpoint_interval,
            _sleep=fake_sleep,
        )
        return df, exchange, sleep_calls, store_path

    # --- return type & columns ---

    def test_returns_dataframe(self, tmp_path: Path):
        df, *_ = self._build(tmp_path, [[]])
        assert isinstance(df, pd.DataFrame)

    def test_empty_history_returns_empty_df(self, tmp_path: Path):
        df, *_ = self._build(tmp_path, [[]])
        assert len(df) == 0

    def test_df_has_expected_columns(self, tmp_path: Path):
        trades = [_trade(1_000, "1000", "1")]
        df, *_ = self._build(tmp_path, [trades, []])
        for col in ["bar_start_ts", "open", "high", "low", "close",
                    "volume", "dollar_volume", "num_ticks"]:
            assert col in df.columns

    def test_df_index_is_utc_datetimeindex(self, tmp_path: Path):
        trades = [_trade(1_000, "1000", "1")]
        df, *_ = self._build(tmp_path, [trades, []])
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == "UTC"

    def test_single_page_produces_bar(self, tmp_path: Path):
        trades = [_trade(1_000, "500", "1"), _trade(2_000, "500", "1")]
        df, *_ = self._build(tmp_path, [trades, []])
        assert len(df) >= 1

    # --- CCXT call arguments ---

    def test_first_call_since_none_on_fresh_run(self, tmp_path: Path):
        _, exchange, _, _ = self._build(tmp_path, [[]])
        exchange.fetch_trades.assert_called_with(
            self._SYMBOL, since=None, limit=1000
        )

    def test_since_ms_override_used_on_first_call(self, tmp_path: Path):
        _, exchange, _, _ = self._build(tmp_path, [[]], since_ms=9_000)
        exchange.fetch_trades.assert_called_with(
            self._SYMBOL, since=9_000, limit=1000
        )

    def test_page_size_forwarded_to_fetch_trades(self, tmp_path: Path):
        _, exchange, _, _ = self._build(tmp_path, [[]], page_size=500)
        exchange.fetch_trades.assert_called_with(
            self._SYMBOL, since=None, limit=500
        )

    def test_cursor_advances_after_full_page(self, tmp_path: Path):
        page1 = [_trade(ts, "100", "0.1") for ts in range(1_000, 1_003)]  # 3 trades
        page2: list[dict] = []
        _, exchange, _, _ = self._build(tmp_path, [page1, page2], page_size=3)
        calls = exchange.fetch_trades.call_args_list
        assert len(calls) == 2
        _, kwargs = calls[1]
        assert kwargs["since"] == 1003  # last_ts + 1

    # --- loop termination ---

    def test_stops_on_empty_page(self, tmp_path: Path):
        _, exchange, _, _ = self._build(tmp_path, [[]])
        assert exchange.fetch_trades.call_count == 1

    def test_stops_on_short_page(self, tmp_path: Path):
        page1 = [_trade(1_000, "100", "1")]  # 1 trade < page_size=1000
        _, exchange, _, _ = self._build(tmp_path, [page1])
        assert exchange.fetch_trades.call_count == 1

    # --- rate limiting ---

    def test_sleep_called_between_full_pages(self, tmp_path: Path):
        page1 = [_trade(ts, "100", "0.1") for ts in range(1_000, 1_003)]
        page2 = [_trade(ts, "100", "0.1") for ts in range(2_000, 2_003)]
        page3: list[dict] = []
        _, _, sleep_calls, _ = self._build(
            tmp_path, [page1, page2, page3], rate_limit_delay=1.5, page_size=3
        )
        assert sleep_calls == [1.5, 1.5]

    def test_sleep_not_called_after_empty_page(self, tmp_path: Path):
        _, _, sleep_calls, _ = self._build(tmp_path, [[]], rate_limit_delay=1.0)
        assert sleep_calls == []

    def test_sleep_not_called_after_short_page(self, tmp_path: Path):
        page = [_trade(1_000, "100", "1")]
        _, _, sleep_calls, _ = self._build(tmp_path, [page], rate_limit_delay=1.0)
        assert sleep_calls == []

    # --- HDF5 persistence ---

    def test_bars_persisted_and_loadable(self, tmp_path: Path):
        trades = [_trade(1_000, "1000", "1")]
        _, _, _, store_path = self._build(tmp_path, [trades, []])
        df = load_dollar_bars(self._SYMBOL, self._THRESHOLD, store_path)
        assert len(df) >= 1

    def test_checkpoint_written_after_run(self, tmp_path: Path):
        trades = [_trade(5_000, "1000", "1")]
        _, _, _, store_path = self._build(tmp_path, [trades, []])
        key = _hdf5_key(self._SYMBOL, self._THRESHOLD)
        assert _read_checkpoint(store_path, key) == 5_000

    # --- resume from checkpoint ---

    def test_resume_uses_checkpoint_plus_one(self, tmp_path: Path):
        t1 = [_trade(5_000, "1000", "1")]
        _, _, _, store_path = self._build(tmp_path, [t1, []])

        exchange2 = _make_exchange([[]])
        build_dollar_bars(
            self._SYMBOL, self._THRESHOLD, store_path,
            exchange=exchange2, rate_limit_delay=0.0, _sleep=lambda _: None,
        )
        exchange2.fetch_trades.assert_called_with(
            self._SYMBOL, since=5_001, limit=1000
        )

    def test_since_ms_overrides_checkpoint(self, tmp_path: Path):
        t1 = [_trade(5_000, "1000", "1")]
        _, _, _, store_path = self._build(tmp_path, [t1, []])

        exchange2 = _make_exchange([[]])
        build_dollar_bars(
            self._SYMBOL, self._THRESHOLD, store_path,
            exchange=exchange2, since_ms=1_000,
            rate_limit_delay=0.0, _sleep=lambda _: None,
        )
        exchange2.fetch_trades.assert_called_with(
            self._SYMBOL, since=1_000, limit=1000
        )


# ===========================================================================
# load_dollar_bars
# ===========================================================================

class TestLoadDollarBars:
    _THRESHOLD = Decimal("1000")
    _SYMBOL = "BTC/USD"

    def test_missing_file_returns_empty_df(self, tmp_path: Path):
        df = load_dollar_bars(self._SYMBOL, self._THRESHOLD, tmp_path / "nope.h5")
        assert isinstance(df, pd.DataFrame) and len(df) == 0

    def test_missing_key_returns_empty_df(self, tmp_path: Path):
        store_path = tmp_path / "bars.h5"
        other_key = _hdf5_key("ETH/USD", self._THRESHOLD)
        df_in = _bars_to_df([_make_bar()])
        with pd.HDFStore(store_path, mode="w", complevel=5, complib="blosc") as store:
            store.append(other_key, df_in, format="table", data_columns=True)
        df = load_dollar_bars(self._SYMBOL, self._THRESHOLD, store_path)
        assert len(df) == 0

    def test_loads_what_was_written(self, tmp_path: Path):
        store_path = tmp_path / "bars.h5"
        key = _hdf5_key(self._SYMBOL, self._THRESHOLD)
        bars = [_make_bar(start=i * 1000, end=(i + 1) * 1000) for i in range(3)]
        df_in = _bars_to_df(bars)
        with pd.HDFStore(store_path, mode="w", complevel=5, complib="blosc") as store:
            store.append(key, df_in, format="table", data_columns=True)
        df_out = load_dollar_bars(self._SYMBOL, self._THRESHOLD, store_path)
        assert len(df_out) == 3

    def test_loaded_df_has_utc_index(self, tmp_path: Path):
        store_path = tmp_path / "bars.h5"
        key = _hdf5_key(self._SYMBOL, self._THRESHOLD)
        df_in = _bars_to_df([_make_bar()])
        with pd.HDFStore(store_path, mode="w", complevel=5, complib="blosc") as store:
            store.append(key, df_in, format="table", data_columns=True)
        df_out = load_dollar_bars(self._SYMBOL, self._THRESHOLD, store_path)
        assert str(df_out.index.tz) == "UTC"


# ===========================================================================
# Exception branch coverage
# ===========================================================================

class TestExceptionBranches:
    def test_read_checkpoint_missing_attr_returns_none(self, tmp_path: Path):
        """Key present in store but last_trade_ts attr was never written → None.
        Covers the except KeyError branch in _read_checkpoint."""
        store_path = tmp_path / "bars.h5"
        key = _hdf5_key("BTC/USD", Decimal("1000"))
        df = _bars_to_df([_make_bar()])
        # Write the dataset but deliberately skip writing the checkpoint attr
        with pd.HDFStore(store_path, mode="w", complevel=5, complib="blosc") as store:
            store.append(key, df, format="table", data_columns=True)
        # No _write_checkpoint call → attrs["last_trade_ts"] missing → KeyError
        assert _read_checkpoint(store_path, key) is None

    def test_load_dollar_bars_corrupted_file_returns_empty(self, tmp_path: Path):
        """Unreadable HDF5 file → except Exception branch → empty DataFrame.
        Covers lines 457-459 in load_dollar_bars."""
        store_path = tmp_path / "bars.h5"
        # Write garbage bytes — not a valid HDF5 file
        store_path.write_bytes(b"not an hdf5 file at all \x00\x01\x02")
        df = load_dollar_bars("BTC/USD", Decimal("1000"), store_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
