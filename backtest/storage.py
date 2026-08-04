"""backtest.storage -- HDF5 append/load for the trial ledger and result
verdicts (Phase 2, Task 6). The only module in ``backtest/`` that opens
an HDFStore; ``ledger.py`` stays pure compute, no I/O.

Local key constants live here deliberately, NOT in ``data/hdf5_keys.py``
(Step 0c): those are a symbol/threshold -> group-name convention for the
bars/labels file, a different axis entirely. Backtest is keyed by
``config_hash`` (results) and ``trial_id`` (ledger entries) against its
own store, ``backtest/results/backtest.h5``, with two top-level
namespaces:

    /ledger/trial_{trial_id}   -- one entry per real-data trial (append-only)
    /results/cfg_{config_hash} -- one entry per distinct config's verdict

The ledger is append-ONLY by design (the "you can't hide trials" ethic,
Q11): :func:`append_trial` raises rather than overwrite if a
``trial_id`` already exists, and no function here ever drops a trial
based on ``config_hash`` collisions -- deduping by ``config_hash`` is
:mod:`backtest.ledger`'s reduction-time concern, not a storage-time one.
Re-running the same config on purpose is a normal, expected event that
must still show up in the physical record; only :func:`backtest.ledger.n_trials`
et al. treat it as superseded.

``load_ledger``/``load_result`` fail loud on a missing store or missing
key -- the same posture ``research.labels.storage.load_labels`` uses,
for the same reason: a lookup by a specific identity (a trial_id or a
config_hash) is almost always a typo when absent, not a resumable
checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.ledger import TrialRecord

__all__ = ["append_trial", "load_ledger", "save_result", "load_result"]

_HDF5_COMPLEVEL: int = 5
_HDF5_COMPLIB: str = "blosc"

_LEDGER_PREFIX: str = "/ledger/trial_"
_RESULTS_PREFIX: str = "/results/cfg_"


def _trial_key(trial_id: str) -> str:
    """Return the full HDFStore key for a trial. Example:
    '/ledger/trial_a1b2c3d4'."""
    return f"{_LEDGER_PREFIX}{trial_id}"


def _result_key(config_hash: str) -> str:
    """Return the full HDFStore key for a result. Example:
    '/results/cfg_2e31b10f8bf7'."""
    return f"{_RESULTS_PREFIX}{config_hash}"


def append_trial(record: TrialRecord, store_path: Path) -> None:
    """Append one trial to the ledger. Append-only: raises rather than
    overwrites if ``record.trial_id`` already exists -- a duplicate
    trial_id is a caller bug (trial_ids must be unique per physical
    run), never a legitimate re-run (re-runs get a fresh trial_id and
    are handled by dedup at read time).

    Persists ``record.returns`` as the table body; every other field is
    written as a ``storer.attrs`` scalar.

    Raises
    ------
    KeyError
        If ``record.trial_id`` is already present in the store.
    """
    key = _trial_key(record.trial_id)
    mode = "a" if store_path.exists() else "w"
    with pd.HDFStore(
        store_path, mode=mode, complevel=_HDF5_COMPLEVEL, complib=_HDF5_COMPLIB
    ) as store:
        if key in store:
            raise KeyError(
                f"append_trial: trial_id={record.trial_id!r} already "
                f"exists at {key!r} -- the ledger is append-only; "
                f"trial_ids must be unique per physical run."
            )
        store.put(key, record.returns, format="table", data_columns=True)
        storer = store.get_storer(key)
        storer.attrs["trial_id"] = record.trial_id
        storer.attrs["config_hash"] = record.config_hash
        storer.attrs["created_at"] = record.created_at.isoformat()
        storer.attrs["mode"] = record.mode
        storer.attrs["sharpe"] = record.sharpe
        storer.attrs["n_bets"] = record.n_bets


def load_ledger(store_path: Path) -> list:
    """Load every trial ever appended, as a ``list[TrialRecord]`` --
    deduping (by ``config_hash``, latest ``created_at`` wins) is
    :mod:`backtest.ledger`'s job, not this function's; this returns the
    full, undeduped physical record.

    Raises
    ------
    FileNotFoundError
        If ``store_path`` does not exist.
    KeyError
        If the store exists but no trial has ever been appended to it.
    """
    if not store_path.exists():
        raise FileNotFoundError(f"No HDF5 store at {store_path}.")
    records = []
    with pd.HDFStore(store_path, mode="r") as store:
        for key in store.keys():
            if not key.startswith(_LEDGER_PREFIX):
                continue
            returns = store[key]
            attrs = store.get_storer(key).attrs
            records.append(
                TrialRecord(
                    trial_id=str(attrs["trial_id"]),
                    config_hash=str(attrs["config_hash"]),
                    created_at=str(attrs["created_at"]),
                    mode=str(attrs["mode"]),
                    sharpe=float(attrs["sharpe"]),
                    n_bets=int(attrs["n_bets"]),
                    returns=returns,
                )
            )
    if not records:
        raise KeyError(
            f"load_ledger: no trials have ever been appended to {store_path}."
        )
    return records


def save_result(
    paths: pd.DataFrame,
    config_hash: str,
    created_at,
    metadata: dict,
    store_path: Path,
    overwrite: bool = True,
) -> None:
    """Persist a backtest verdict's dense ``paths`` matrix, keyed by
    ``config_hash``.

    ``paths`` columns are integer ``path_id``s (per
    :func:`backtest.paths.reconstruct_paths`) -- stored WITHOUT
    ``data_columns`` (pytables' table format rejects non-string column
    labels for indexed data columns; whole-frame load is all this ever
    needs, so columnar querying is not a loss).

    Overwrite-by-default (``overwrite=True``): a config's verdict is
    deterministic given the same trial population, so re-saving is
    either a no-op or an intentional refresh. ``overwrite=False`` raises
    :class:`KeyError` if the key already exists.

    Parameters
    ----------
    paths:
        The dense OOS path matrix (WF: 1 column; CPCV: phi columns; MC:
        n_sims columns).
    config_hash:
        The :class:`~backtest.config.BacktestConfig` recipe hash --
        the storage key.
    created_at:
        Timestamp of this verdict (any value ``pd.Timestamp`` accepts).
    metadata:
        The verdict's scalars/tri-states/per-mode breakdown -- whatever
        shape ``engine.py``'s ``BacktestResult`` settles on; stored
        as-is as a ``storer.attrs`` dict.
    """
    key = _result_key(config_hash)
    mode = "a" if store_path.exists() else "w"
    with pd.HDFStore(
        store_path, mode=mode, complevel=_HDF5_COMPLEVEL, complib=_HDF5_COMPLIB
    ) as store:
        if key in store:
            if not overwrite:
                raise KeyError(
                    f"save_result: result already stored at {key!r} and "
                    f"overwrite=False."
                )
            store.remove(key)
        store.put(key, paths, format="table")
        storer = store.get_storer(key)
        storer.attrs["config_hash"] = config_hash
        storer.attrs["created_at"] = pd.Timestamp(created_at).isoformat()
        storer.attrs["metadata"] = dict(metadata)


def load_result(config_hash: str, store_path: Path) -> tuple:
    """Load a previously saved verdict for ``config_hash``.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        ``(paths, info)`` -- ``info`` contains ``config_hash``,
        ``created_at`` (as a ``pd.Timestamp``), and every key from the
        original ``metadata`` dict, merged flat.

    Raises
    ------
    FileNotFoundError
        If ``store_path`` does not exist.
    KeyError
        If no result was ever saved for this ``config_hash``.
    """
    if not store_path.exists():
        raise FileNotFoundError(f"No HDF5 store at {store_path}.")
    key = _result_key(config_hash)
    with pd.HDFStore(store_path, mode="r") as store:
        if key not in store:
            raise KeyError(
                f"load_result: no result stored at {key!r} for "
                f"config_hash={config_hash!r}."
            )
        paths = store[key]
        attrs = store.get_storer(key).attrs
        info = {
            "config_hash": str(attrs["config_hash"]),
            "created_at": pd.Timestamp(str(attrs["created_at"])),
            **dict(attrs["metadata"]),
        }
    return paths, info
