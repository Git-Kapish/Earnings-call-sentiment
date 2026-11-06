"""M1: select basket, build calls + sentences tables (data/interim)."""
import logging
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset

from ecs import prices
from ecs import transcripts as T

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("make_data")
cfg = yaml.safe_load(open("configs/default.yaml"))
raw, interim = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["interim"])
raw.mkdir(parents=True, exist_ok=True); interim.mkdir(parents=True, exist_ok=True)
b = cfg["basket"]

ds = load_dataset(T.HF_ID, split="train")
meta = ds.select_columns(["symbol", "year", "quarter", "date"]).to_pandas()
meta["date"] = pd.to_datetime(meta["date"])

sp = interim / "sectors.parquet"
if not sp.exists():
    elig = meta[(meta.date >= b["start"]) & (meta.date <= b["end"])].groupby("symbol").size()
    tick = list(elig[elig >= b["min_calls"]].index)
    log.info("fetching sectors for %d candidates", len(tick))
    prices.fetch_sectors(tick).to_frame().to_parquet(sp)
sectors = pd.read_parquet(sp)["sector"]
basket = T.select_basket(meta, sectors, b["n_tickers"], b["min_calls"], b["start"], b["end"], cfg["seed"])
basket = [str(t) for t in basket]
log.info("basket (%d): %s", len(basket), basket)

bset = set(basket)
sel = ds.filter(lambda r: r["symbol"] in bset and b["start"] <= r["date"][:10] <= b["end"])
rows, sents = [], []
for r in sel:
    ts = pd.Timestamp(r["date"])
    cid = f'{r["symbol"]}_{r["year"]}Q{r["quarter"]}'
    turns = T.split_sections(r["structured_content"])
    rows.append(dict(call_id=cid, ticker=r["symbol"], company=r["company_name"], fiscal_year=r["year"],
                     fiscal_quarter=r["quarter"], call_dt=ts, call_date=ts.normalize(),
                     call_timing=T.call_timing(ts), sector=sectors.get(r["symbol"]),
                     has_qa=any(t[0] == "qa" for t in turns), n_turns=len(turns)))
    i = 0
    for sec, spk, txt in turns:
        for s in T.split_sentences(txt):
            sents.append((cid, i, sec, spk, s)); i += 1
calls = pd.DataFrame(rows).sort_values(["ticker", "call_dt"]).reset_index(drop=True)
sent = pd.DataFrame(sents, columns=["call_id", "sent_idx", "section", "speaker", "text"])
calls.to_parquet(interim / "calls.parquet"); sent.to_parquet(interim / "sentences.parquet")
log.info("calls=%d sentences=%d", len(calls), len(sent))
print(calls.groupby("ticker").size().describe())
print(calls.call_timing.value_counts()); print("has_qa share", calls.has_qa.mean().round(3))
print(sent.groupby("section").size())
