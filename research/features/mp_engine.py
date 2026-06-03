"""
Multiprocessing Engine — research/features/mp_engine.py

Single entry-point for parallelising embarrassingly-parallel research tasks
(feature engineering, labeling, backtesting) across a pool of workers.

Design follows the pattern from AFML Ch.20 (multiprocessing & vectorisation):
  - Atoms are the indivisible units of work (e.g. one bar, one label, one path).
  - numThreads controls the worker pool size.
  - mpBatches splits atoms into batches so progress can be reported and
    partial results reduced without waiting for the full job.
  - redux / reduxArgs / reduxInPlace control how per-batch outputs are combined.

All pandas/numpy stays inside research/. This module never crosses the seam.

Usage example:
    from research.features.mp_engine import mp_pandas_obj

    results = mp_pandas_obj(
        func=my_feature_func,
        pd_obj=("molecule", molecule_index),
        numThreads=4,
    )
"""
from __future__ import annotations

import copyreg
import io
import logging
import multiprocessing as mp
import pickle
import types
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Pickle registration ───────────────────────────────────────────────────────
# Bound methods are not picklable by default in Python < 3.5 and remain
# unreliable across multiprocessing boundaries with some class hierarchies.
# Registering a custom reducer makes them safely serialisable.

def _reduce_method(m: Any) -> tuple:
    """Pickle reducer for bound methods."""
    return getattr, (m.__self__, m.__func__.__name__)


def _register_pickle_reducers() -> None:
    """
    Register custom pickle reducers for types that are not natively picklable.
    Called once at module import time. Safe to call multiple times.
    """
    copyreg.pickle(types.MethodType, _reduce_method)


_register_pickle_reducers()


# ── Worker ────────────────────────────────────────────────────────────────────

def _process_jobs(jobs: list[dict[str, Any]], redux: Callable | None, reduxArgs: dict, reduxInPlace: bool) -> Any:
    """
    Execute a list of jobs in the current process and reduce results.
    Each job dict must have keys: 'func', 'args' (tuple), 'kwargs' (dict).
    """
    out: list[Any] = []

    for job in jobs:
        result = job["func"](*job["args"], **job["kwargs"])
        out.append(result)

    return _reduce_results(out, redux, reduxArgs, reduxInPlace)


def _reduce_results(
    results: list[Any],
    redux: Callable | None,
    reduxArgs: dict,
    reduxInPlace: bool,
) -> Any:
    """
    Combine a list of per-batch results into a single output.

    If redux is None, attempts to concatenate assuming pandas/list output.
    If redux is provided, applies it iteratively across results.
    """
    if not results:
        return results

    if redux is None:
        # Default: try pandas concat, fall back to list flatten
        try:
            import pandas as pd
            return pd.concat(results)
        except Exception:
            flat: list[Any] = []
            for r in results:
                if isinstance(r, list):
                    flat.extend(r)
                else:
                    flat.append(r)
            return flat

    # Apply redux function iteratively
    out = results[0]
    for r in results[1:]:
        if reduxInPlace:
            redux(out, r, **reduxArgs)
        else:
            out = redux(out, r, **reduxArgs)
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def mp_pandas_obj(
    func: Callable,
    pd_obj: tuple[str, Any],
    numThreads: int = 1,
    mpBatches: int = 1,
    redux: Callable | None = None,
    reduxArgs: dict | None = None,
    reduxInPlace: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Parallelise func over a pandas index or list of atoms.

    Args:
        func:         Callable accepting (pd_obj_name=atoms_batch, **kwargs).
                      Must be picklable — use a module-level function or a
                      bound method of a class registered with copyreg.
        pd_obj:       Two-tuple (arg_name, atoms) where atoms is a pandas
                      Index, Series, or list. arg_name is the keyword argument
                      name func expects for its slice of atoms.
        numThreads:   Worker pool size. 1 → single-process (easier to debug).
        mpBatches:    Number of batches to split atoms into. Each batch is one
                      job; more batches = finer-grained progress reporting.
        redux:        Optional reduction function(accumulated, new, **reduxArgs).
        reduxArgs:    Keyword arguments forwarded to redux.
        reduxInPlace: If True, redux mutates the first argument in-place.
        **kwargs:     Additional keyword arguments forwarded to func unchanged.

    Returns:
        Reduced result of applying func across all atom batches.

    Raises:
        ValueError: If numThreads < 1 or mpBatches < 1.
        PicklingError: If func or its arguments are not picklable.
    """
    if numThreads < 1:
        raise ValueError(f"numThreads must be >= 1, got {numThreads}.")
    if mpBatches < 1:
        raise ValueError(f"mpBatches must be >= 1, got {mpBatches}.")

    if reduxArgs is None:
        reduxArgs = {}

    arg_name, atoms = pd_obj

    # Split atoms into batches
    try:
        # pandas Index / Series
        parts = [atoms[i::mpBatches] for i in range(mpBatches)]
    except TypeError:
        # plain list
        parts = [atoms[i::mpBatches] for i in range(mpBatches)]

    # Remove empty batches (happens when mpBatches > len(atoms))
    parts = [p for p in parts if len(p) > 0]

    if not parts:
        logger.warning("mp_pandas_obj: no atoms to process.")
        return None

    jobs = [
        {
            "func": func,
            "args": (),
            "kwargs": {arg_name: part, **kwargs},
        }
        for part in parts
    ]

    logger.info(
        "mp_pandas_obj: %d atoms → %d batch(es), %d thread(s).",
        len(atoms),
        len(jobs),
        numThreads,
    )

    # Single-process path — easier to debug, same result
    if numThreads == 1:
        return _process_jobs(jobs, redux, reduxArgs, reduxInPlace)

    # Multiprocess path
    # Verify picklability before submitting to pool to get a clean error
    _assert_picklable(func, jobs)

    results: list[Any] = []
    with mp.Pool(processes=numThreads) as pool:
        futures = [
            pool.apply_async(
                job["func"],
                args=job["args"],
                kwds=job["kwargs"],
            )
            for job in jobs
        ]
        for future in futures:
            results.append(future.get())

    return _reduce_results(results, redux, reduxArgs, reduxInPlace)


def _assert_picklable(func: Callable, jobs: list[dict]) -> None:
    """
    Smoke-test picklability before submitting to a Pool.
    Raises pickle.PicklingError with a clear message on failure.
    """
    buf = io.BytesIO()
    try:
        pickle.dump(func, buf)
        for job in jobs[:1]:   # check first job only — all share the same func
            pickle.dump(job["kwargs"], buf)
    except Exception as exc:
        raise pickle.PicklingError(
            f"mp_pandas_obj: func or its arguments are not picklable. "
            f"Ensure func is a module-level function or a bound method "
            f"whose class is registered with copyreg. Original error: {exc}"
        ) from exc
