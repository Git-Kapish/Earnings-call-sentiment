# Data sources

## Transcripts — `kurry/sp500_earnings_transcripts` (Hugging Face) — CONFIRMED by user 2026-11-06
- URL: https://huggingface.co/datasets/kurry/sp500_earnings_transcripts
- Licence: MIT per dataset card, "for research and educational use only". Original transcript provenance is **not stated** on the card (Capital IQ company ids suggest a S&P Capital IQ origin). Transcript text is never committed to git (`data/` is ignored).
- Probed: 2026-11-06 via `scripts/probe_coverage.py` (metadata only).
- Fields: `symbol`, `company_name`, `company_id`, `year`, `quarter` (fiscal), `date` (YYYY-MM-DD HH:MM:SS), `content` (raw text), `structured_content` (list of `{speaker, text}`).

### Coverage (probe results)
| Metric | Value |
|---|---|
| Rows / tickers | 33,362 / 685 |
| Date range | 2005-10-13 to 2025-05-15 |
| Null dates | 0; duplicate (symbol, year, quarter): 0 |
| Calls since 2015 | 638 tickers; 611 with >= 8 calls, 575 with >= 16, 481 with >= 24 |
| Calls per year 2015-2024 | ~1,800-2,200 |

### Call timing
- `date` includes a time of day for 99.4% of rows (0.6% are 00:00, i.e. unknown -> must be flagged `unknown`, run both alignments).
- Most common times: 17:00 (4,710), 08:30 (4,589), 10:00 (4,350), 09:00 (3,933), 11:00 (3,292), 08:00 (2,814), 16:30 (2,294).
- **Timezone is not documented.** The distribution (08:30/09:00 pre-market, 16:30/17:00 post-close) is consistent with US Eastern. Assumption: ET. Must be spot-checked on known calls before M4.
- 10:00-11:00 calls fall *during* the session, so they need a rule (PRD: during/after close -> next trading day) and a both-variants robustness check.

### Known issues
- Some text has encoding damage (`�` replacing apostrophes) -> clean in M1.
- Sections: no prepared/Q&A flag; derive from `structured_content` (Operator turns, analyst vs management speakers) in M1.
- Sample is S&P 500 constituents (survivorship / index-membership bias) -> limitation in README.

## Prices — `yfinance` (daily adjusted close), benchmark SPY
Not yet downloaded. Record download date and yfinance version in M4.

## Lexicon — Loughran-McDonald Master Dictionary 1993-2025 (Notre Dame SRAF)
Downloaded 2026-11-06 from https://sraf.nd.edu/loughranmcdonald-master-dictionary/ (free for academic research; commercial use needs a licence). Stored at data/raw/LM_MasterDictionary.csv (git-ignored).
