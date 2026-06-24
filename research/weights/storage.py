"""Weight persistence -- store/load/list sample-weight sets in HDF5 (Q8).

Weights live in the *same* ``.h5`` file as bars and labels, nested one
level deeper than labels to encode provenance directly in the key:

    /{SYM_KEY}/thr_{N}/weights/lbl_{label_hash}/cfg_{weight_hash}

The ``lbl_{label_hash}`` segment is a sibling of ``labels/`` under the
same ``thr_`` group, so "list every weight recipe derived from label set
X" is a clean prefix scan, and a stored weight set is physically
co-located with the label set it came from.

``store_weights`` takes the ``LabelConfig`` OBJECT (not a bare hash
string) -- mirrors ``store_labels`` taking ``LabelConfig``, and the
orchestrator already holds the ``LabelConfig`` that produced the labels
it's weighting, so passing it through is natural.

A stored weight set is fully self-describing: attrs record BOTH configs
(``weight_config``, ``label_config``) plus ``label_config_hash`` as a
queryable provenance link, so a weight set's parent label recipe can be
read back without touching the label group at all.

``load_weights`` is fail-loud, same posture as ``load_labels``: a weights
lookup is a specific-recipe query where absence is almost always a typo,
not a resumable checkpoint.
"""
from __future__ import annotations

import dataclasses
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd

from data.hdf5_keys import symbol_to_key, threshold_to_key
from research.labels.config import LabelConfig
from research.weights.config import WeightConfig

__all__ = ["store_weights", "load_weights", "list_weight_configs"]

_HDF5_COMPLEVEL: int = 5
_HDF5_COMPLIB: str = "blosc"


def _weights_key(
    symbol: str,
    threshold: Decimal,
    weight_config: WeightConfig,
    label_config: LabelConfig,
) -> str:
    """Return the full HDFStore key for this (symbol, threshold,
    label_config, weight_config) combination.

    Example: '/BTC_USD/thr_1000000/weights/lbl_2e31b10f8bf7/cfg_9f1a0c2b3d4e'
    """
    return (
        f"/{symbol_to_key(symbol)}/{threshold_to_key(threshold)}"
        f"/weights/lbl_{label_config.config_hash()}"
        f"/cfg_{weight_config.config_hash()}"
    )


def _weights_prefix(symbol: str, threshold: Decimal) -> str:
    """Key prefix shared by every weight recipe under (symbol, threshold),
    across ALL parent label sets."""
    return f"/{symbol_to_key(symbol)}/{threshold_to_key(threshold)}/weights/"


def store_weights(
    weights: pd.DataFrame,
    weight_config: WeightConfig,
    label_config: LabelConfig,
    symbol: str,
    threshold: Decimal,
    store_path: Path,
    overwrite: bool = True,
) -> None:
    """Persist a sample-weight frame for (symbol, threshold, label_config,
    weight_config) to HDF5.

    Overwrite-by-default (``overwrite=True``): the recipe is deterministic,
    so re-running and re-storing is always either a no-op or an
    intentional refresh. ``overwrite=False`` raises :class:`KeyError` if
    the key already exists, rather than silently replacing it.

    Writes the following ``storer.attrs`` alongside the frame:
      * ``weight_config``      -- ``dataclasses.asdict(weight_config)``
      * ``label_config``       -- ``dataclasses.asdict(label_config)``
      * ``label_config_hash``  -- ``label_config.config_hash()`` (the
        provenance link, queryable without reparsing ``label_config``)
      * ``symbol``              -- as given (e.g. ``'BTC/USD'``)
      * ``threshold``            -- as given (``Decimal``)
      * ``created_at``            -- UTC ISO-8601 timestamp of this write
      * ``n_weights``              -- ``len(weights)``
    """
    key = _weights_key(symbol, threshold, weight_config, label_config)
    mode = "a" if store_path.exists() else "w"
    with pd.HDFStore(
        store_path, mode=mode, complevel=_HDF5_COMPLEVEL, complib=_HDF5_COMPLIB
    ) as store:
        if key in store:
            if not overwrite:
                raise KeyError(
                    f"Weights already stored at {key!r} and overwrite=False."
                )
            store.remove(key)
        store.put(key, weights, format="table", data_columns=True)
        storer = store.get_storer(key)
        storer.attrs["weight_config"] = asdict(weight_config)
        storer.attrs["label_config"] = asdict(label_config)
        storer.attrs["label_config_hash"] = label_config.config_hash()
        storer.attrs["symbol"] = symbol
        storer.attrs["threshold"] = threshold
        storer.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        storer.attrs["n_weights"] = len(weights)


def load_weights(
    weight_config: WeightConfig,
    label_config: LabelConfig,
    symbol: str,
    threshold: Decimal,
    store_path: Path,
) -> pd.DataFrame:
    """Load a previously stored weight frame for (symbol, threshold,
    label_config, weight_config).

    Fails loud -- same posture as ``load_labels``.

    Raises
    ------
    FileNotFoundError
        If ``store_path`` does not exist.
    KeyError
        If the store exists but this exact (``label_config``,
        ``weight_config``) pair was never stored for (symbol, threshold).
        This includes the case where the same ``weight_config`` was stored
        under a *different* ``label_config`` -- the parent label hash
        genuinely partitions storage, it is not cosmetic.
    """
    if not store_path.exists():
        raise FileNotFoundError(f"No HDF5 store at {store_path}.")
    key = _weights_key(symbol, threshold, weight_config, label_config)
    with pd.HDFStore(store_path, mode="r") as store:
        if key not in store:
            raise KeyError(
                f"No weights stored at {key!r} for symbol={symbol!r}, "
                f"threshold={threshold!r}, "
                f"label_config_hash={label_config.config_hash()!r}, "
                f"weight_config_hash={weight_config.config_hash()!r}."
            )
        return store[key]


def list_weight_configs(
    symbol: str,
    threshold: Decimal,
    store_path: Path,
) -> pd.DataFrame:
    """List every weight recipe stored for (symbol, threshold), across ALL
    parent label sets, without loading any weight frame.

    Returns a :class:`pd.DataFrame` indexed by ``weight_config_hash``, with
    one column per :class:`WeightConfig` field (flat scalars) plus
    ``label_config_hash`` (the provenance link) and ``n_weights``. Read
    entirely from ``storer.attrs`` -- never opens a weight frame via
    ``store[key]``. Returns an empty frame (correct columns,
    ``weight_config_hash`` index name) if the store file doesn't exist or
    no weight recipe has been stored for this (symbol, threshold) pair.
    """
    columns = [f.name for f in dataclasses.fields(WeightConfig)] + [
        "label_config_hash",
        "n_weights",
    ]
    empty = pd.DataFrame(columns=columns)
    empty.index.name = "weight_config_hash"

    if not store_path.exists():
        return empty

    prefix = _weights_prefix(symbol, threshold)
    rows: dict[str, dict] = {}
    with pd.HDFStore(store_path, mode="r") as store:
        for key in store.keys():
            if not key.startswith(prefix):
                continue
            # key shape: {prefix}lbl_{label_hash}/cfg_{weight_hash}
            weight_hash = key.rsplit("/cfg_", 1)[-1]
            attrs = store.get_storer(key).attrs
            row = dict(attrs["weight_config"])
            row["label_config_hash"] = attrs["label_config_hash"]
            row["n_weights"] = attrs["n_weights"]
            rows[weight_hash] = row

    if not rows:
        return empty

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "weight_config_hash"
    return df
