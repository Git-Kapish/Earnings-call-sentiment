"""Event-study machinery: day-0 alignment, market model, abnormal returns, CAR.

All positions are integer offsets on the TRADING-DAY index (the SPY price index), never calendar days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULES = ("prd", "alt", "same", "next")


def day0(call_date: pd.Timestamp, timing: str, idx: pd.DatetimeIndex, rule: str = "prd") -> pd.Timestamp | None:
    """Day-0 = first trading day on which the market can react to the call. Uses only the call
    date + timing flag (known at call time).

    rule 'prd'  : before_open -> call date; during / after_close -> NEXT trading day (PRD / CLAUDE.md rule).
    rule 'alt'  : before_open / during -> call date; after_close -> next trading day (robustness).
    rule 'same' : always call date; rule 'next': always next trading day (usable for unknown timing too).
    unknown timing returns None under 'prd'/'alt' (never guess silently).
    A non-trading call date (weekend/holiday) rolls forward to the next session.
    """
    if rule not in RULES:
        raise ValueError(rule)
    if rule in ("prd", "alt") and timing == "unknown":
        return None
    same = {"prd": timing == "before_open", "alt": timing in ("before_open", "during"),
            "same": True, "next": False}[rule]
    # same-day: first session >= call_date; next-day: first session strictly > call_date
    pos = idx.searchsorted(call_date, side="left" if same else "right")
    return idx[pos] if pos < len(idx) else None


def estimation_slice(d0_pos: int, est: tuple[int, int] = (-250, -30)) -> slice:
    """Integer slice of the estimation window relative to day-0 position. Asserts it ends strictly
    before day 0 (and thus before every event-window day t >= 0)."""
    lo, hi = d0_pos + est[0], d0_pos + est[1]
    assert est[1] < 0 and hi < d0_pos, "estimation window must end before the event window"
    return slice(max(lo, 0), hi + 1)


def market_model(stock: np.ndarray, mkt: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float, float, int]:
    """OLS stock = a + b*mkt on non-NaN (and mask-True) obs. Returns (alpha, beta, resid_sd, n_obs)."""
    ok = ~(np.isnan(stock) | np.isnan(mkt))
    if mask is not None:
        ok &= mask
    n = int(ok.sum())
    if n < 3:
        return np.nan, np.nan, np.nan, n
    b, a = np.polyfit(mkt[ok], stock[ok], 1)
    resid = stock[ok] - (a + b * mkt[ok])
    return float(a), float(b), float(resid.std(ddof=2)), n


def abnormal_returns(ret: pd.Series, mkt: pd.Series, d0: pd.Timestamp, model: str = "market_model",
                     est: tuple[int, int] = (-250, -30), min_obs: int = 120, path: tuple[int, int] = (-5, 5),
                     exclude_days: pd.DatetimeIndex | None = None) -> dict | None:
    """AR_t for t in `path` around d0. model 'market_model' (alpha, beta from the estimation window,
    excluding `exclude_days`, e.g. other earnings reactions) or 'market_adjusted' (AR = R - R_mkt).
    Returns None if the estimation window has < min_obs usable days or the path runs off the data."""
    idx = ret.index
    p0 = idx.get_loc(d0)
    if p0 + path[1] >= len(idx) or p0 + path[0] < 0:
        return None
    a = b = 0.0
    n = 0
    if model == "market_model":
        sl = estimation_slice(p0, est)
        mask = None if exclude_days is None else ~idx[sl].isin(exclude_days)
        a, b, _, n = market_model(ret.values[sl], mkt.values[sl], mask)
        if n < min_obs:
            return None
    elif model == "market_adjusted":
        b = 1.0
    else:
        raise ValueError(model)
    sl = slice(p0 + path[0], p0 + path[1] + 1)
    ar = ret.values[sl] - (a + b * mkt.values[sl])
    return {"ar": pd.Series(ar, index=list(range(path[0], path[1] + 1))), "alpha": a, "beta": b, "n_est": n}


def car(ar: pd.Series, lo: int, hi: int) -> float:
    """CAR[lo, hi] = sum of AR_t for t in lo..hi (relative to day 0)."""
    return float(ar.loc[lo:hi].sum())
