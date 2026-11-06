"""M0 probe: coverage stats for the candidate transcript dataset (no transcript text saved)."""
import pandas as pd
from datasets import load_dataset

ds = load_dataset("kurry/sp500_earnings_transcripts", split="train")
df = ds.select_columns(["symbol", "year", "quarter", "date"]).to_pandas()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
print("rows", len(df), "tickers", df.symbol.nunique(), "null dates", df.date.isna().sum())
print("date range", df.date.min(), df.date.max())
df["hhmm"] = df.date.dt.strftime("%H:%M")
print("midnight (no time) share", (df.hhmm == "00:00").mean().round(3))
print(df.hhmm.value_counts().head(12))
print("dup (symbol,year,quarter)", df.duplicated(["symbol", "year", "quarter"]).sum())
rec = df[df.date >= "2015-01-01"]
per = rec.groupby("symbol").size()
print("since 2015: tickers", len(per), ">=8 calls", (per >= 8).sum(), ">=16", (per >= 16).sum(), ">=24", (per >= 24).sum())
print(rec.groupby(rec.date.dt.year).size().to_string())
