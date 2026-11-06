"""M5/M6: primary spec, quintile sorts, horse race, robustness grid -> results/tables/*.csv.

PRIMARY (pre-declared): call-level FinBERT net score (all sections) -> CAR[0,+1], market model, alignment rule
from config (alt), regression with controls + sector & year-quarter FE, SE clustered by firm.
Everything else is EXPLORATORY and is labelled so; the number of exploratory specs is written out.
"""
import logging
from pathlib import Path

import pandas as pd
import yaml

from ecs import analysis as A

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("analysis")
cfg = yaml.safe_load(open("configs/default.yaml"))["event"]
out = Path("results/tables")
out.mkdir(parents=True, exist_ok=True)
ev = pd.read_parquet("data/processed/events.parquet")
lm = pd.read_parquet("data/processed/scores_lm.parquet")
fb = pd.read_parquet("data/processed/scores_finbert.parquet")
RULE, WINS = cfg["primary_rule"], [f"car_{lo}_{hi}" for lo, hi in cfg["car_windows"]]
PRIMARY_CAR = "car_%d_%d" % tuple(cfg["primary_window"])

# ---- primary ----
d = A.build_panel(ev, lm, fb, RULE, "market_model", "all")
prim = A.regress(d, ["fb_net_z"], PRIMARY_CAR)
primary = pd.DataFrame([A.row(prim, "fb_net_z", "PRIMARY: FinBERT -> CAR[0,+1]", d, status="primary")])
primary.to_csv(out / "primary.csv", index=False)
log.info("PRIMARY: %s", primary.round(4).to_dict("records"))

# ---- quintile sorts (FinBERT and LM, each window) ----
qs = []
for sc in ("fb_net", "lm_net"):
    for car in WINS:
        r = A.quintile_sort(d, sc, car)
        qs.append({"score": sc, "car": car, "q1": r["table"]["mean"].iloc[0], "q5": r["table"]["mean"].iloc[-1],
                   "spread_q5_q1": r["spread"], "t_firm_cluster": r["t"], "p": r["p"], "n_q1q5": r["n"],
                   "status": "primary" if (sc, car) == ("fb_net", PRIMARY_CAR) else "exploratory"})
        r["table"].assign(score=sc, car=car).to_csv(out / f"quintiles_{sc}_{car}.csv")
pd.DataFrame(qs).to_csv(out / "quintile_spreads.csv", index=False)

# ---- horse race (primary window; plus other windows as exploratory) ----
hr = pd.concat([A.horse_race(d, c).assign(car=c, status="primary" if c == PRIMARY_CAR else "exploratory")
                for c in WINS])
hr.to_csv(out / "horse_race.csv", index=False)
log.info("Spearman(FinBERT, LM) = %.3f (n=%d)", A.rank_corr(d), len(d))

# ---- robustness grid (all EXPLORATORY) ----
rob = []
for rule in ("alt", "prd", "same", "next"):
    for model in ("market_model", "market_adjusted"):
        for section in ("all", "prepared", "qa"):
            p = A.build_panel(ev, lm, fb, rule, model, section)
            for car in WINS:
                for wins in (False, True):
                    y = p.assign(**{car: A.winsorize(p[car])}) if wins else p
                    for lab, x in (("FinBERT", "fb_net_z"), ("LM", "lm_net_z")):
                        r = A.regress(y, [x], car)
                        rob.append(A.row(r, x, lab, y, rule=rule, model=model, section=section, car=car,
                                         winsor=wins, status="exploratory"))
rob = pd.DataFrame(rob)
rob.loc[(rob.rule == RULE) & (rob.model == "market_model") & (rob.section == "all") & (rob.car == PRIMARY_CAR)
        & (~rob.winsor) & (rob.spec == "FinBERT"), "status"] = "primary"
rob.to_csv(out / "robustness.csv", index=False)

# two-way clustering and HC1 for the primary spec, as SE checks
se = pd.DataFrame([A.row(A.regress(d, ["fb_net_z"], PRIMARY_CAR, cl), "fb_net_z", f"primary, SE={cl}", d)
                   for cl in ("firm", "two_way", "hc1")])
se.to_csv(out / "primary_se_variants.csv", index=False)
log.info("exploratory specs run: %d (multiple-testing caveat applies)", len(rob))
print(primary.round(4).T)
print(se[["spec", "coef", "se", "t", "n"]].round(4))
print(hr[hr.car == PRIMARY_CAR][["spec", "x", "coef", "se", "t", "n", "adj_r2"]].round(4))
print(pd.DataFrame(qs)[["score", "car", "q1", "q5", "spread_q5_q1", "t_firm_cluster", "n_q1q5"]].round(4))
print(f"robustness: {len(rob)} regs; share with p<0.05: {(rob.p < 0.05).mean():.3f} "
      f"(FinBERT {(rob[rob.spec == 'FinBERT'].p < 0.05).mean():.3f}, LM {(rob[rob.spec == 'LM'].p < 0.05).mean():.3f})")
