"""Label persistence -- store/load/list triple-barrier label sets in HDF5.

Labels live in the *same* ``.h5`` file as dollar bars, as a sibling group:

    /{SYM_KEY}/thr_{N}/labels/cfg_{hash}

Persistence is intentionally separate from computation (see
:func:`research.labels.pipeline.build_triple_barrier_labels`, which is pure
compute, no I/O) -- this module is the only place that opens an HDFStore for
labels.

``load_labels`` deliberately diverges from ``data.bars.load_dollar_bars``'s
empty-frame-on-missing: a bars load backs a resumable checkpoint where
absence is expected; a labels load is a specific-recipe lookup where absence
is almost always a typo, so it fails loud instead.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd

from data.hdf5_keys import symbol_to_key, threshold_to_key
from research.labels.config import LabelConfig

__all__ = ["store_labels", "load_labels", "list_label_configs"]

_HDF5_COMPLEVEL: int = 5
_HDF5_COMPLIB: str = "blosc"


def _labels_key(symbol: str, threshold: Decimal, config: LabelConfig) -> str:
    """Return the full HDFStore key for (symbol, threshold, config).

    Example: '/BTC_USD/thr_1000000/labels/cfg_2e31b10f8bf7'
    """
    return (
        f"/{symbol_to_key(symbol)}/{threshold_to_key(threshold)}"
        f"/labels/cfg_{config.config_hash()}"
    )


def _labels_prefix(symbol: str, threshold: Decimal) -> str:
    """Return the key prefix shared by every config stored for (symbol, threshold)."""
    return f"/{symbol_to_key(symbol)}/{threshold_to_key(threshold)}/labels/cfg_"


def store_labels(
    labels: pd.DataFrame,
    config: LabelConfig,
    symbol: str,
    threshold: Decimal,
    store_path: Path,
    overwrite: bool = True,
) -> None:
    """Persist a label frame for (symbol, threshold, config) to HDF5.

    Overwrite-by-default (``overwrite=True``): the recipe is deterministic,
    so re-running and re-storing is always either a no-op or an intentional
    refresh. ``overwrite=False`` raises :class:`KeyError` if the key already
    exists, rather than silently replacing it.

    Writes the following ``storer.attrs`` alongside the frame:
      * ``label_config``  -- ``dataclasses.asdict(config)``
      * ``symbol``        -- as given (e.g. ``'BTC/USD'``)
      * ``threshold``     -- as given (``Decimal``)
      * ``t1_encoding``   -- ``"datetime"`` (tz-aware datetime column)
      * ``created_at``    -- UTC ISO-8601 timestamp of this write
      * ``n_labels``      -- ``len(labels)``
    """
    key = _labels_key(symbol, threshold, config)
    mode = "a" if store_path.exists() else "w"
    with pd.HDFStore(
        store_path, mode=mode, complevel=_HDF5_COMPLEVEL, complib=_HDF5_COMPLIB
    ) as store:
        if key in store:
            if not overwrite:
                raise KeyError(
                    f"Labels already stored at {key!r} and overwrite=False."
                )
            store.remove(key)
        store.put(key, labels, format="table", data_columns=True)
        storer = store.get_storer(key)
        storer.attrs["label_config"] = asdict(config)
        storer.attrs["symbol"] = symbol
        storer.attrs["threshold"] = threshold
        storer.attrs["t1_encoding"] = "datetime"
        storer.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        storer.attrs["n_labels"] = len(labels)


def load_labels(
    config: LabelConfig,
    symbol: str,
    threshold: Decimal,
    store_path: Path,
) -> pd.DataFrame:
    """Load a previously stored label frame for (symbol, threshold, config).

    Fails loud -- a deliberate divergence from ``load_dollar_bars``:

    Raises:
        FileNotFoundError: if ``store_path`` does not exist.
        KeyError: if the store exists but this exact config hash was never
            stored for (symbol, threshold).
    """
    if not store_path.exists():
        raise FileNotFoundError(f"No HDF5 store at {store_path}.")
    key = _labels_key(symbol, threshold, config)
    with pd.HDFStore(store_path, mode="r") as store:
        if key not in store:
            raise KeyError(
                f"No labels stored at {key!r} for symbol={symbol!r}, "
                f"threshold={threshold!r}, config_hash={config.config_hash()!r}."
            )
        return store[key]


def list_label_configs(
    symbol: str,
    threshold: Decimal,
    store_path: Path,
) -> pd.DataFrame:
    """List every config stored for (symbol, threshold), without loading any frame.

    Returns a :class:`pd.DataFrame` indexed by ``cfg_hash``, with one column
    per :class:`LabelConfig` field (flat scalars) plus ``n_labels``. Read
    entirely from ``storer.attrs`` -- never opens a label frame via
    ``store[key]``. Returns an empty frame (correct columns, ``cfg_hash``
    index name) if the store file doesn't exist or no config has been
    stored for this (symbol, threshold) pair.
    """
    import dataclasses

    columns = [f.name for f in dataclasses.fields(LabelConfig)] + ["n_labels"]
    empty = pd.DataFrame(columns=columns)
    empty.index.name = "cfg_hash"

    if not store_path.exists():
        return empty

    prefix = _labels_prefix(symbol, threshold)
    rows: dict[str, dict] = {}
    with pd.HDFStore(store_path, mode="r") as store:
        for key in store.keys():
            if not key.startswith(prefix):
                continue
            cfg_hash = key[len(prefix):]
            attrs = store.get_storer(key).attrs
            row = dict(attrs["label_config"])
            row["n_labels"] = attrs["n_labels"]
            rows[cfg_hash] = row

    if not rows:
        return empty

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "cfg_hash"
    return df
