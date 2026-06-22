"""Triple-barrier labeling primitives (AFML Ch.3).

The vertical barrier is measured in *bars forward* (Decision 1), a deliberate
deviation from AFML's wall-clock ``addVerticalBarrier`` to stay coherent with
information-driven dollar-bar sampling. Path-walk and labeling use *simple*
returns (AFML Snippets 3.2 / 3.5).
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd


class Barrier(IntEnum):
    """Which barrier bound an event first.

    Values are chosen to align with ``bin``'s sign in the symmetric case while
    remaining semantically distinct, and to store cleanly as ``int8``.
    """

    LOWER = -1      # stop-loss / lower horizontal touched first
    VERTICAL = 0    # vertical barrier (timeout) bound first
    UPPER = 1       # profit-take / upper horizontal touched first


def _vertical_barrier(
    index: pd.DatetimeIndex,
    t_events: pd.DatetimeIndex,
    max_hp: int,
) -> pd.Series:
    """Bars-forward vertical barrier (Decision 1).

    For each event timestamp, the vertical barrier falls ``max_hp`` *bars*
    later in ``index`` -- a deliberate deviation from AFML's wall-clock
    ``addVerticalBarrier`` (Snippet 3.4), chosen to stay coherent with
    information-driven dollar-bar sampling. Events whose horizon runs past
    the final bar receive ``NaT`` (the path-walk later treats these as
    "walk to end of data"; see :func:`get_events`).

    Parameters
    ----------
    index:
        The full bar index (UTC ``DatetimeIndex``), typically ``close.index``.
    t_events:
        Event timestamps, a subset of ``index`` (typically the output of
        :func:`research.labels.filters.cusum_filter`).
    max_hp:
        Vertical horizon in bars; must be >= 1.

    Returns
    -------
    pd.Series
        Named ``t1``, indexed by ``t_events``, tz-aware datetime dtype
        (tz of ``index`` preserved). ``NaT`` wherever ``position + max_hp``
        exceeds the last bar.

    Raises
    ------
    ValueError
        If ``max_hp < 1``, or if any timestamp in ``t_events`` is not a
        member of ``index`` (fail loud rather than silently misalign).
    """
    if max_hp < 1:
        raise ValueError(f"max_hp must be >= 1, got {max_hp}.")

    positions = index.get_indexer(t_events)
    if (positions == -1).any():
        raise ValueError(
            "t_events contains timestamps that are not members of index."
        )

    n = len(index)
    target = positions + max_hp
    # Clip into range so the gather is always valid, then NaT out the tail.
    clipped = np.clip(target, 0, n - 1) if n else target
    t1 = pd.Series(index[clipped], index=t_events, name="t1")
    t1[target >= n] = pd.NaT
    return t1


def _apply_pt_sl_on_t1(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float],
) -> pd.DataFrame:
    """First-touch path-walk over the triple barrier (AFML Snippet 3.2).

    For each event, walks the *simple-return* path from the event start to its
    vertical barrier ``t1`` (or to the last available bar when ``t1`` is
    ``NaT``), oriented by ``side``, and records which barrier was touched
    first. Touch detection is **close-only** (Decision Q2); the OHLC high/low
    variant is deferred to a Phase-2 ADR.

    Conventions (locked):
      * Returns are oriented: ``r = (close[loc:end] / close[loc] - 1) * side``,
        so ``UPPER`` is always the profit-take side and ``LOWER`` the
        stop-loss side, for both long and short.
      * Thresholds are ``pt = pt_mult * trgt`` and ``sl = -sl_mult * trgt``;
        a non-positive multiplier disables that horizontal barrier.
      * Touch uses AFML's strict inequalities (``r > pt`` / ``r < sl``); a
        return landing exactly on a threshold is *not* a touch.
      * Horizontal-wins-on-the-vertical-bar (Decision Q4): if the earliest
        horizontal touch lands on or before the vertical bar, the horizontal
        barrier is recorded, never ``VERTICAL``.
      * Same-bar PT+SL collision cannot occur under close-only (a single
        close return is either ``> pt > 0`` or ``< sl < 0``, never both).

    Parameters
    ----------
    close:
        Bar close prices, UTC ``DatetimeIndex``, ascending.
    events:
        Indexed by event-start timestamps, with columns ``t1`` (vertical
        barrier, ``NaT``-able), ``trgt`` (per-event unit target), and ``side``
        (``1`` long / ``-1`` short; ``get_events`` defaults it to ``1``).
    pt_sl:
        ``(pt_mult, sl_mult)`` multipliers applied to ``trgt``.

    Returns
    -------
    pd.DataFrame
        Indexed by ``events.index`` with columns:
          * ``t1``      -- realized first-touch timestamp (earliest of the
            profit-take, stop-loss, and vertical barriers); ``NaT`` when an
            event is unresolved (``NaT`` vertical and no horizontal touch
            before the data ends).
          * ``barrier`` -- nullable ``Int8`` :class:`Barrier` value;
            ``<NA>`` for unresolved events. Downstream (:func:`get_bins`)
            drops ``NaT``-``t1`` rows and casts to plain ``int8``.
    """
    pt_mult, sl_mult = pt_sl
    last_ts = close.index[-1] if len(close) else None

    out_t1: list = []
    out_barrier: list = []

    for loc in events.index:
        t1_v = events.at[loc, "t1"]
        trgt = events.at[loc, "trgt"]
        side = events.at[loc, "side"]

        end = t1_v if pd.notna(t1_v) else last_ts
        path = close.loc[loc:end]
        ret = (path / close.loc[loc] - 1.0) * side

        pt = pt_mult * trgt
        sl = -sl_mult * trgt
        pt_ts = ret.index[ret > pt].min() if pt_mult > 0 else pd.NaT
        sl_ts = ret.index[ret < sl].min() if sl_mult > 0 else pd.NaT

        # Earliest horizontal touch (and which one).
        horizontals = [
            (ts, barrier)
            for ts, barrier in ((pt_ts, Barrier.UPPER), (sl_ts, Barrier.LOWER))
            if pd.notna(ts)
        ]
        h_ts, h_barrier = (
            min(horizontals, key=lambda pair: pair[0])
            if horizontals
            else (pd.NaT, None)
        )

        if pd.notna(h_ts) and (pd.isna(t1_v) or h_ts <= t1_v):
            out_t1.append(h_ts)
            out_barrier.append(int(h_barrier))
        elif pd.notna(t1_v):
            out_t1.append(t1_v)
            out_barrier.append(int(Barrier.VERTICAL))
        else:
            out_t1.append(pd.NaT)
            out_barrier.append(pd.NA)

    return pd.DataFrame(
        {
            "t1": pd.Series(
                pd.to_datetime(out_t1, utc=True),
                index=events.index,
                name="t1",
            ),
            "barrier": pd.array(out_barrier, dtype="Int8"),
        },
        index=events.index,
    )


def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: tuple[float, float],
    trgt: pd.Series,
    max_hp: int,
    min_ret: float = 0.0,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """Assemble triple-barrier events (AFML Snippet 3.3).

    Aligns the per-bar target ``trgt`` to ``t_events``, drops events whose
    target is undefined or below ``min_ret``, attaches the bars-forward
    vertical barrier, runs the close-only first-touch path-walk, and returns
    an events frame whose ``t1`` is the *realized* first-touch timestamp.

    Symmetric vs meta-labeling:
      * ``side is None`` -> symmetric. A synthetic ``side = 1`` drives the
        (orientation-invariant) path-walk, and the ``side`` column is dropped
        from the result, exactly as AFML does.
      * ``side`` given    -> meta-labeling. Returns oriented by ``side``, and
        the ``side`` column is retained for :func:`get_bins`.

    Parameters
    ----------
    close:
        Bar close prices, UTC ``DatetimeIndex``, ascending.
    t_events:
        Event timestamps (typically :func:`cusum_filter` output).
    pt_sl:
        ``(pt_mult, sl_mult)`` multipliers applied to ``trgt``.
    trgt:
        Per-bar unit target (e.g. ``ewma_vol``), indexed like ``close``.
    max_hp:
        Vertical horizon in bars (>= 1).
    min_ret:
        Minimum target to keep an event; the filter is AFML-strict
        (``trgt > min_ret``), which also drops ``NaN`` targets.
    side:
        Optional per-event bet side (``1``/``-1``) for meta-labeling.

    Returns
    -------
    pd.DataFrame
        Indexed by surviving event-start timestamps, with columns:
          * ``t1``      -- realized first-touch timestamp (``NaT`` if an event
            is still unresolved at the end of the data).
          * ``trgt``    -- the (filtered) per-event target, ``float64``.
          * ``barrier`` -- nullable ``Int8`` :class:`Barrier` value
            (``<NA>`` for unresolved events).
          * ``side``    -- ``int8`` bet side; present only in meta-labeling.

        Unresolved events are retained here; :func:`get_bins` drops them.
    """
    trgt = trgt.reindex(pd.DatetimeIndex(t_events))
    trgt = trgt[trgt > min_ret]  # AFML-strict; also drops NaN targets
    event_index = trgt.index

    if side is None:
        side_ = pd.Series(1, index=event_index, dtype="int8")
        keep_side = False
    else:
        side_ = side.reindex(event_index).astype("int8")
        keep_side = True

    t1 = _vertical_barrier(close.index, event_index, max_hp)
    events_ = pd.DataFrame(
        {"t1": t1, "trgt": trgt, "side": side_}, index=event_index
    )

    touched = _apply_pt_sl_on_t1(close, events_, pt_sl)

    out = pd.DataFrame(index=event_index)
    out["t1"] = touched["t1"]
    out["trgt"] = trgt
    out["barrier"] = touched["barrier"]
    if keep_side:
        out["side"] = side_
    return out


def get_bins(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Compute labels from triple-barrier events (AFML Snippet 3.5).

    Drops unresolved events (``t1`` is ``NaT``) -- these have no realized
    outcome yet and cannot be labeled. For the remainder, computes the
    *simple* return from each event's start to its realized ``t1``, then:

      * Symmetric (no ``side`` column on ``events``): ``bin = sign(ret)``,
        i.e. ``{-1, 0, 1}``.
      * Meta-labeling (``side`` present): ``ret`` is oriented by ``side``
        first, then ``bin = sign(ret)`` with every non-positive outcome
        collapsed to ``0`` -- AFML's exact rule, so a losing bet (``ret<0``)
        and a flat one (``ret==0``) are both "the side called it wrong",
        landing in ``{0, 1}``.

    Parameters
    ----------
    events:
        Output of :func:`get_events` -- indexed by event-start timestamp,
        with ``t1`` (``NaT``-able), ``trgt``, ``barrier`` (nullable ``Int8``),
        and optionally ``side``.
    close:
        Bar close prices, UTC ``DatetimeIndex``, ascending.

    Returns
    -------
    pd.DataFrame
        Indexed by surviving event-start timestamps, columns ``ret``
        (``float64``), ``bin`` (``int8``), ``t1`` (tz-aware ``datetime64``),
        ``barrier`` (plain ``int8`` -- the ``NaT``/``<NA>`` pairing in
        :func:`get_events` means no nulls survive the drop), and ``side``
        (``int8``) only in the meta-labeling path.
    """
    resolved = events.dropna(subset=["t1"])
    has_side = "side" in resolved.columns

    px_start = close.reindex(resolved.index).to_numpy(dtype="float64")
    px_t1 = close.reindex(resolved["t1"]).to_numpy(dtype="float64")
    ret = px_t1 / px_start - 1.0

    if has_side:
        ret = ret * resolved["side"].to_numpy(dtype="float64")

    bin_ = np.sign(ret).astype("int8")
    if has_side:
        bin_ = np.where(ret <= 0, 0, bin_).astype("int8")

    out = pd.DataFrame(index=resolved.index)
    out["ret"] = ret.astype("float64")
    out["bin"] = bin_
    out["t1"] = resolved["t1"]
    out["barrier"] = resolved["barrier"].astype("int8")
    if has_side:
        out["side"] = resolved["side"].astype("int8")
    return out