"""M2/M3: call-level sentiment scores per section (all / prepared / qa) -> data/processed/scores_lm.parquet."""
import logging
from pathlib import Path

import pandas as pd

from ecs import lexicon

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("make_scores")
sent = pd.read_parquet("data/interim/sentences.parquet")
lex = lexicon.load_lexicon()
rows = []
for (cid, sec), g in sent.groupby(["call_id", "section"]):
    rows.append({"call_id": cid, "section": sec, **lexicon.score_text(" ".join(g.text), lex)})
allrows = [{"call_id": cid, "section": "all", **lexicon.score_text(" ".join(g.text), lex)} for cid, g in sent.groupby("call_id")]
out = pd.DataFrame(rows + allrows)
out.to_parquet("data/processed/scores_lm.parquet")
log.info("LM scores: %d rows", len(out))

# --- FinBERT (cached, resumable) ---
from ecs import finbert  # noqa: E402

finbert.score_calls(sent, "data/interim/finbert_cache")
cache = pd.concat(pd.read_parquet(f) for f in sorted((Path("data/interim/finbert_cache") / finbert.REVISION[:8]).glob("*.parquet")))
m = sent.merge(cache, on=["call_id", "sent_idx"], validate="one_to_one")
assert len(m) == len(sent), "missing FinBERT scores"
m["net"] = m.p_pos - m.p_neg
m["is_pos"] = (m[["p_pos", "p_neg", "p_neu"]].values.argmax(1) == 0)
m["is_neg"] = (m[["p_pos", "p_neg", "p_neu"]].values.argmax(1) == 1)


def agg(g):
    return pd.Series({"fb_net": g.net.mean(), "fb_pct_pos": g.is_pos.mean(), "fb_pct_neg": g.is_neg.mean(), "n_sent": len(g)})


fb = pd.concat([m.groupby(["call_id", "section"]).apply(agg).reset_index(),
                m.groupby("call_id").apply(agg).reset_index().assign(section="all")])
fb["revision"] = finbert.REVISION
fb.to_parquet("data/processed/scores_finbert.parquet")
log.info("FinBERT scores: %d rows", len(fb))
