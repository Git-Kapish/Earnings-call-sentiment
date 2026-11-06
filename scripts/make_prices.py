"""M4a: download adjusted prices (+SPY) for the basket -> data/interim/{close,volume}.parquet."""
import logging

import pandas as pd

from ecs import prices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
calls = pd.read_parquet("data/interim/calls.parquet")
tick = sorted(calls.ticker.unique()) + ["SPY"]
close, vol = prices.download_prices(tick, "2013-10-01", "2025-02-28")
close.to_parquet("data/interim/close.parquet"); vol.to_parquet("data/interim/volume.parquet")
missing = [t for t in tick if t not in close or close[t].dropna().empty]
logging.info("prices %s, %s..%s, missing=%s, yfinance download date %s", close.shape, close.index.min().date(),
             close.index.max().date(), missing, pd.Timestamp.today().date())
print(close.notna().sum().sort_values().head(8))
