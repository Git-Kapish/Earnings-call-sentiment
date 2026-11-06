"""Load, clean, section and sentence-split earnings-call transcripts.

Timing: `call_dt` is the call start as given by the source, assumed US/Eastern
(see docs/data_sources.md). Nothing here looks at prices or post-call information.
"""
from __future__ import annotations

import logging
import re

import nltk
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
nltk.download("punkt_tab", quiet=True)

HF_ID = "kurry/sp500_earnings_transcripts"
_QA_START = re.compile(
    r"(question[- ]and[- ]answer|q\s*&\s*a|first question|take (?:your|our) questions|open (?:it|the (?:call|line|floor)) (?:up )?for questions|your (?:first )?question comes)",
    re.I,
)


def call_timing(call_dt: pd.Timestamp) -> str:
    """Flag from the call's local (ET) start time: before_open (<09:30), during (09:30-16:00),
    after_close (>=16:00), unknown (midnight = no time in source)."""
    if pd.isna(call_dt) or (call_dt.hour == 0 and call_dt.minute == 0):
        return "unknown"
    m = call_dt.hour * 60 + call_dt.minute
    return "before_open" if m < 570 else "during" if m < 960 else "after_close"


def clean_text(s: str) -> str:
    """Fix the source's broken apostrophes/quotes and collapse whitespace."""
    s = s.replace("\ufffd", "'")
    return re.sub(r"\s+", " ", s).strip()


def select_basket(meta: pd.DataFrame, sectors: pd.Series, n: int, min_calls: int,
                  start: str, end: str, seed: int) -> list[str]:
    """Pre-declared basket rule (uses no return data): tickers with >= min_calls calls in
    [start, end] and a known sector; allocate n proportionally across sectors (largest
    remainder), then draw within sector with a seeded RNG."""
    m = meta[(meta.date >= start) & (meta.date <= end)]
    cnt = m.groupby("symbol").size()
    elig = cnt[cnt >= min_calls].index.intersection(sectors.dropna().index)
    sec = sectors.loc[elig]
    share = sec.value_counts() / len(sec) * n
    alloc = share.astype(int)
    alloc[(share - alloc).sort_values(ascending=False).index[: n - alloc.sum()]] += 1
    rng = np.random.RandomState(seed)
    picks: list[str] = []
    for s, k in alloc.items():
        pool = sorted(sec[sec == s].index)
        picks += list(rng.choice(pool, size=min(k, len(pool)), replace=False))
    return sorted(picks)


def split_sections(segments: list[dict]) -> list[tuple[str, str, str]]:
    """(section, speaker, text) per turn. Operator turns are dropped. Q&A starts at the first
    operator turn matching _QA_START after turn 0; if none is found, section='unknown'."""
    qa_at = next((i for i, s in enumerate(segments)
                  if i > 0 and s["speaker"].strip().lower() == "operator" and _QA_START.search(s["text"])), None)
    out = []
    for i, s in enumerate(segments):
        if s["speaker"].strip().lower() == "operator":
            continue
        sec = "unknown" if qa_at is None else ("prepared" if i < qa_at else "qa")
        out.append((sec, s["speaker"].strip(), clean_text(s["text"])))
    return out


def split_sentences(text: str) -> list[str]:
    return [x for x in nltk.sent_tokenize(text) if len(x.split()) >= 3]
