"""Batched, disk-cached FinBERT sentence scorer (ProsusAI/finbert, pinned revision).

Timing: pure function of text; no dates. Sentences longer than the model limit are split into
<=510-token chunks (+[CLS]/[SEP]) and chunk probabilities are token-weighted, so no text is dropped.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

log = logging.getLogger(__name__)
MODEL = "ProsusAI/finbert"
REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
MAX_LEN = 512
LABELS = ["positive", "negative", "neutral"]  # model.config.id2label order


def chunk_ids(ids: list[int], size: int = MAX_LEN - 2) -> list[list[int]]:
    """Split content token ids (no special tokens) into chunks of <= size; concatenation == input."""
    return [ids[i:i + size] for i in range(0, max(len(ids), 1), size)]


class FinBert:
    def __init__(self, batch_size: int = 32, device: str = "cpu"):
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION).eval().to(device)
        assert [self.model.config.id2label[i] for i in range(3)] == LABELS
        self.bs, self.device = batch_size, device

    @torch.no_grad()
    def _probs(self, id_chunks: list[list[int]]) -> np.ndarray:
        out = np.zeros((len(id_chunks), 3), dtype=np.float32)
        order = np.argsort([len(c) for c in id_chunks])  # length-sorted batches: less padding
        for i in range(0, len(order), self.bs):
            idx = order[i:i + self.bs]
            seqs = [[self.tok.cls_token_id, *id_chunks[j], self.tok.sep_token_id] for j in idx]
            L = max(map(len, seqs))
            ids = torch.tensor([q + [self.tok.pad_token_id] * (L - len(q)) for q in seqs], device=self.device)
            mask = torch.tensor([[1] * len(q) + [0] * (L - len(q)) for q in seqs], device=self.device)
            logits = self.model(input_ids=ids, attention_mask=mask).logits
            out[idx] = torch.softmax(logits, dim=-1).cpu().numpy()
        return out

    def score(self, texts: list[str]) -> np.ndarray:
        """(n, 3) probs [pos, neg, neu]. Asserts every content token was fed to the model."""
        enc = self.tok(texts, add_special_tokens=False, truncation=False)["input_ids"]
        chunks, owner = [], []
        for k, ids in enumerate(enc):
            for c in chunk_ids(ids):
                chunks.append(c)
                owner.append(k)
        assert sum(map(len, chunks)) == sum(map(len, enc)), "tokens dropped"
        p, w = self._probs(chunks), np.array([max(len(c), 1) for c in chunks], dtype=np.float32)
        res = np.zeros((len(texts), 3), dtype=np.float32)
        tot = np.zeros(len(texts), dtype=np.float32)
        for k, pk, wk in zip(owner, p, w):
            res[k] += pk * wk
            tot[k] += wk
        return res / tot[:, None]


def score_calls(sent: pd.DataFrame, cache_dir: str | Path, model: FinBert | None = None) -> None:
    """Score every call's sentences, one parquet per call at <cache>/<rev8>/<call_id>.parquet
    (keyed by call id, sent_idx, revision). Existing files are skipped, so runs are resumable."""
    d = Path(cache_dir) / REVISION[:8]
    d.mkdir(parents=True, exist_ok=True)
    model = model or FinBert()
    calls = sorted(sent.call_id.unique())
    for n, cid in enumerate(calls):
        f = d / f"{cid}.parquet"
        if f.exists():
            continue
        g = sent[sent.call_id == cid]
        p = model.score(g.text.tolist())
        pd.DataFrame({"call_id": cid, "sent_idx": g.sent_idx.values, "revision": REVISION,
                      "p_pos": p[:, 0], "p_neg": p[:, 1], "p_neu": p[:, 2]}).to_parquet(f)
        if n % 25 == 0:
            log.info("scored %d/%d calls", n + 1, len(calls))
