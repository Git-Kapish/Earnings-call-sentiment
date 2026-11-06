"""Price and sector data via yfinance."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def _sector(ticker: str) -> str | None:
    try:
        return yf.Ticker(ticker.replace(".", "-")).info.get("sector")
    except Exception as e:  # network/delisted: treated as unavailable
        log.debug("sector lookup failed for %s: %s", ticker, e)
        return None


def fetch_sectors(tickers: list[str], workers: int = 8) -> pd.Series:
    """Sector per ticker (as of download time; NaN if unavailable). No date semantics."""
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(_sector, tickers))
    return pd.Series(out, index=tickers, name="sector")


def download_prices(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily split/dividend-adjusted close and volume, wide (date x ticker). Dates are exchange
    session dates (yfinance daily bars); no calendar-day arithmetic is done on them."""
    d = yf.download([t.replace(".", "-") for t in tickers], start=start, end=end, auto_adjust=True,
                    progress=False, threads=True)
    close, vol = d["Close"], d["Volume"]
    for x in (close, vol):
        x.columns = [c.replace("-", ".") for c in x.columns]
    return close.sort_index(), vol.sort_index()
