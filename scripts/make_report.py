"""M7: figures from the analysis tables -> results/figures/*.png, plus results/tables/summary.md."""
import logging

import pandas as pd
import yaml

from ecs import analysis as A
from ecs import report as R

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
cfg = yaml.safe_load(open("configs/default.yaml"))["event"]
car = "car_%d_%d" % tuple(cfg["primary_window"])
ev = pd.read_parquet("data/processed/events.parquet")
lm = pd.read_parquet("data/processed/scores_lm.parquet")
fb = pd.read_parquet("data/processed/scores_finbert.parquet")
d = A.build_panel(ev, lm, fb, cfg["primary_rule"], "market_model", "all")

R.sentiment_distribution(d)
R.scatter_fb_lm(d)
R.car_by_quintile(d, "fb_net", car, "car_by_quintile_finbert.png", "FinBERT")
R.car_by_quintile(d, "lm_net", car, "car_by_quintile_lm.png", "LM")
R.car_path(d, "fb_net", "car_path_finbert.png", "FinBERT quintiles")
hr = pd.read_csv("results/tables/horse_race.csv")
R.coef_plot(hr[hr.car == car], "coef_plot.png")

prim = pd.read_csv("results/tables/primary.csv")
hrp = hr[hr.car == car][["spec", "x", "coef", "se", "t", "p", "n", "adj_r2"]]
with open("results/tables/summary.md", "w") as f:
    f.write("## Primary spec (pre-declared)\n\n" + "```
" + prim.round(4).to_string(index=False) + "
```" + "\n\n")
    f.write("## Horse race, CAR[0,+1] (z-scored tone; SE clustered by firm)\n\n" + "```
" + hrp.round(4).to_string(index=False) + "
```" + "\n")
logging.info("report done")
