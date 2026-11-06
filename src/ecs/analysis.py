"""Quintile sorts, regressions with clustered SE, horse race, robustness grid.

Timing: sentiment scores come only from the call transcript; controls (prior_ret, pre_vol, log_dvol) are
measured on trading days [-60,-2] before day 0 (see scripts/make_events.py). Nothing post-event enters a regressor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

CONTROLS = ["prior_ret", "pre_vol", "log_dvol"]
SCORES = {"fb": "fb_net", "lm": "lm_net"}


def build_panel(events: pd.DataFrame, lm: pd.DataFrame, fb: pd.DataFrame, rule: str, model: str,
                section: str = "all") -> pd.DataFrame:
    """One row per event for (rule, model); sentiment from `section`; z-scores computed on this sample."""
    e = events[(events.rule == rule) & (events.model == model)]
    s = (lm[lm.section == section][["call_id", "lm_net", "lm_neg", "lm_unc"]]
         .merge(fb[fb.section == section][["call_id", "fb_net", "fb_pct_pos", "fb_pct_neg"]], on="call_id"))
    d = e.merge(s, on="call_id").dropna(subset=["fb_net", "lm_net"] + CONTROLS).copy()
    for c in ("fb_net", "lm_net"):
        d[f"{c}_z"] = (d[c] - d[c].mean()) / d[c].std()
    d["yq"] = d.call_date.dt.to_period("Q").astype(str)
    d["sector"] = d.sector.fillna("Unknown")
    return d


def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    return s.clip(s.quantile(p), s.quantile(1 - p))


def quintile_sort(d: pd.DataFrame, score: str, car: str) -> dict:
    """Mean CAR by pooled score quintile; Q5-Q1 spread with firm-clustered t-stat."""
    q = pd.qcut(d[score], 5, labels=False, duplicates="drop") + 1
    tab = d.assign(q=q).groupby("q")[car].agg(["mean", "std", "count"])
    sub = d.assign(q=q)[q.isin([1, 5])].assign(top=lambda x: (x.q == 5).astype(float))
    m = smf.ols(f"{car} ~ top", sub).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sub.ticker)[0]})
    return {"table": tab, "spread": m.params["top"], "t": m.tvalues["top"], "p": m.pvalues["top"], "n": int(m.nobs)}


def regress(d: pd.DataFrame, xs: list[str], car: str, cluster: str = "firm", fe: bool = True):
    """CAR ~ xs + controls (+ sector & year-quarter FE). cluster: 'firm' | 'two_way' (firm x date) | 'hc1'."""
    rhs = " + ".join(xs + CONTROLS + (["C(sector)", "C(yq)"] if fe else []))
    m = smf.ols(f"{car} ~ {rhs}", d)
    if cluster == "firm":
        return m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ticker)[0]})
    if cluster == "two_way":
        g = np.column_stack([pd.factorize(d.ticker)[0], pd.factorize(d.d0)[0]])
        return m.fit(cov_type="cluster", cov_kwds={"groups": g})
    return m.fit(cov_type="HC1")


def row(res, x: str, label: str, d: pd.DataFrame, **kw) -> dict:
    return {"spec": label, "x": x, "coef": res.params[x], "se": res.bse[x], "t": res.tvalues[x],
            "p": res.pvalues[x], "n": int(res.nobs), "r2": res.rsquared, "adj_r2": res.rsquared_adj,
            "n_firms": d.ticker.nunique(), **kw}


def horse_race(d: pd.DataFrame, car: str, cluster: str = "firm") -> pd.DataFrame:
    """FinBERT alone, LM alone, both (z-scored; coef = CAR change per 1 s.d. of tone)."""
    out = []
    for label, xs in [("FinBERT only", ["fb_net_z"]), ("LM only", ["lm_net_z"]), ("Both", ["fb_net_z", "lm_net_z"])]:
        r = regress(d, xs, car, cluster)
        for x in xs:
            out.append(row(r, x, label, d))
    return pd.DataFrame(out)


def rank_corr(d: pd.DataFrame) -> float:
    return float(stats.spearmanr(d.fb_net, d.lm_net)[0])
