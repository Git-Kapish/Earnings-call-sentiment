"""Loughran-McDonald dictionary scorer.

Dictionary: Master Dictionary 1993-2025 (Notre Dame SRAF), downloaded 2026-11-06.
No date semantics: pure function of the text passed in.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

_WORD = re.compile(r"[A-Za-z]+")


@lru_cache(maxsize=1)
def load_lexicon(path: str = "data/raw/LM_MasterDictionary.csv") -> dict[str, frozenset[str]]:
    """Category -> set of UPPERCASE words (column value > 0 means the word is in the list)."""
    d = pd.read_csv(Path(path), usecols=["Word", "Negative", "Positive", "Uncertainty"])
    return {c.lower(): frozenset(d.loc[d[c] > 0, "Word"]) for c in ["Negative", "Positive", "Uncertainty"]}


def score_text(text: str, lex: dict[str, frozenset[str]] | None = None) -> dict[str, float]:
    """lm_net = (pos - neg) / total words; also lm_neg, lm_pos, lm_unc shares. NaN if no words."""
    lex = lex or load_lexicon()
    words = [w.upper() for w in _WORD.findall(text)]
    n = len(words)
    if n == 0:
        return dict(lm_net=float("nan"), lm_pos=float("nan"), lm_neg=float("nan"), lm_unc=float("nan"), n_words=0)
    pos = sum(w in lex["positive"] for w in words)
    neg = sum(w in lex["negative"] for w in words)
    unc = sum(w in lex["uncertainty"] for w in words)
    return dict(lm_net=(pos - neg) / n, lm_pos=pos / n, lm_neg=neg / n, lm_unc=unc / n, n_words=n)
