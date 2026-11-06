"""Figures (PNG, labelled axes, N in title) for the README. Pure functions of the analysis panel/tables."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

FIG = Path("results/figures")


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def sentiment_distribution(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].hist(d.fb_net, bins=40, color="#2a6f97")
    ax[0].set(xlabel="FinBERT net score (mean P(pos) - P(neg))", ylabel="Calls")
    ax[1].hist(d.lm_net * 100, bins=40, color="#c1666b")
    ax[1].set(xlabel="Loughran-McDonald net score (% of words)", ylabel="Calls")
    fig.suptitle(f"Call-level sentiment distribution (N={len(d)} calls)")
    _save(fig, "sentiment_distribution.png")


def scatter_fb_lm(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(d.lm_net * 100, d.fb_net, s=6, alpha=0.4)
    rho = stats.spearmanr(d.fb_net, d.lm_net)[0]
    ax.set(xlabel="LM net score (% of words)", ylabel="FinBERT net score",
           title=f"FinBERT vs LM (N={len(d)}, Spearman rho={rho:.2f})")
    _save(fig, "finbert_vs_lm_scatter.png")


def car_by_quintile(d: pd.DataFrame, score: str, car: str, name: str, label: str) -> None:
    q = pd.qcut(d[score], 5, labels=False) + 1
    g = d.assign(q=q).groupby("q")[car]
    m, se = g.mean() * 100, g.std() / np.sqrt(g.count()) * 100
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(m.index, m.values, yerr=1.96 * se.values, color="#2a6f97", capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel=f"{label} quintile (1 = most negative)", ylabel=f"Mean {car.upper().replace('_', ' ')} (%)",
           title=f"CAR by {label} quintile (N={len(d)}; 95% CI, unclustered)")
    _save(fig, name)


def car_path(d: pd.DataFrame, score: str, name: str, label: str) -> None:
    q = pd.qcut(d[score], 5, labels=False) + 1
    cols = [f"ar_{t}" for t in range(-5, 6)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for qq, c, lab in ((5, "#2a6f97", "Top quintile"), (1, "#c1666b", "Bottom quintile")):
        p = d.loc[q == qq, cols].mean().cumsum() * 100
        ax.plot(range(-5, 6), p.values, marker="o", color=c, label=f"{lab} (n={(q == qq).sum()})")
    ax.axvline(-0.5, color="grey", ls=":")
    ax.set(xlabel="Trading days relative to day 0", ylabel="Cumulative mean AR from day -5 (%)",
           title=f"Average abnormal-return path, {label} (N={len(d)})")
    ax.legend()
    _save(fig, name)


def coef_plot(hr: pd.DataFrame, name: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    h = hr.reset_index(drop=True)
    ylab = [f"{r.car.replace('car_', 'CAR[').replace('_', ',+')}]  {r.spec}: {r.x.split('_')[0]}" for r in h.itertuples()]
    ax.errorbar(h.coef * 100, range(len(h)), xerr=1.96 * h.se * 100, fmt="o", color="#2a6f97", capsize=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(h)), ylab, fontsize=7)
    ax.invert_yaxis()
    ax.set(xlabel="Effect of +1 s.d. tone on CAR (pp), 95% CI, firm-clustered",
           title=f"Horse race coefficients (N={int(h.n.iloc[0])})")
    _save(fig, name)
