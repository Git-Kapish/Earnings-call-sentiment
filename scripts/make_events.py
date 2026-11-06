"""M4: day-0 alignment, market model, CAR table -> data/processed/events.parquet (long: call x rule x model)."""
import logging

import numpy as np
import pandas as pd
import yaml

from ecs import events

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("make_events")
cfg = yaml.safe_load(open("configs/default.yaml"))["event"]
calls = pd.read_parquet("data/interim/calls.parquet")
close = pd.read_parquet("data/interim/close.parquet")
vol = pd.read_parquet("data/interim/volume.parquet")
ret = close.pct_change(fill_method=None)
mkt = ret[cfg["benchmark"]]
idx = ret.index
CTRL = 60  # controls use trading days [-60, -2] relative to day 0 (all strictly before day 0)
rows, drops = [], []
for rule in events.RULES:
    d0s = {r.call_id: events.day0(r.call_date, r.call_timing, idx, rule) for r in calls.itertuples()}
    calls["d0"] = calls.call_id.map(d0s)
    for t, g in calls.dropna(subset=["d0"]).groupby("ticker"):
        g = g.sort_values("d0")
        pos = np.array([idx.get_loc(d) for d in g.d0])
        # overlapping event windows ([-1,+5] vs [-1,+5]) -> drop the later event
        gap_ok = np.r_[True, np.diff(pos) > 6]
        for k, r in enumerate(g.itertuples()):
            if not gap_ok[k]:
                drops.append((r.call_id, rule, "overlap"))
                continue
            others = np.concatenate([np.arange(p, p + 6) for j, p in enumerate(pos) if j != k])
            excl = idx[others[others < len(idx)]]
            p0 = pos[k]
            if p0 - CTRL < 0:
                drops.append((r.call_id, rule, "no_history"))
                continue
            for model in ("market_model", "market_adjusted"):
                res = events.abnormal_returns(ret[t], mkt, r.d0, model, tuple(cfg["estimation_window"]),
                                              cfg["min_estimation_obs"], (-5, 5), excl)
                if res is None:
                    drops.append((r.call_id, rule, f"est_obs_{model}"))
                    continue
                w = slice(p0 - CTRL, p0 - 1)  # positions -60..-2
                row = {"call_id": r.call_id, "rule": rule, "model": model, "d0": r.d0, "alpha": res["alpha"],
                       "beta": res["beta"], "n_est": res["n_est"],
                       "prior_ret": float((1 + ret[t].iloc[w].fillna(0)).prod() - 1),
                       "pre_vol": float(ret[t].iloc[w].std()),
                       "log_dvol": float(np.log((close[t].iloc[w] * vol[t].iloc[w]).mean()))}
                for lo, hi in cfg["car_windows"]:
                    row[f"car_{lo}_{hi}"] = events.car(res["ar"], lo, hi)
                row.update({f"ar_{t_}": v for t_, v in res["ar"].items()})
                rows.append(row)
ev = pd.DataFrame(rows).merge(calls[["call_id", "ticker", "sector", "call_timing", "call_date", "fiscal_year",
                                     "fiscal_quarter"]], on="call_id")
ev.to_parquet("data/processed/events.parquet")
dr = pd.DataFrame(drops, columns=["call_id", "rule", "reason"])
log.info("events rows: %d", len(ev))
print(ev.groupby(["rule", "model"]).size().unstack())
print(dr.groupby(["rule", "reason"]).size().unstack(fill_value=0))
print(ev[(ev.rule == "prd") & (ev.model == "market_model")][["car_0_1", "car_0_3", "car_0_5", "beta", "n_est"]].describe().round(4))
