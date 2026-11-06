"""M2/M3: call-level sentiment scores per section (all / prepared / qa) -> data/processed/scores_lm.parquet."""
import logging

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
