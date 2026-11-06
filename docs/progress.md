# Progress

- Scaffold created, py3.12 venv, remote set.
- M0: candidate transcript source probed (kurry/sp500_earnings_transcripts, 33,362 calls, 685 tickers, MIT). See docs/data_sources.md. Awaiting user confirmation before ingestion.
- M1 done: basket = 60 tickers (seeded, sector-stratified, >=16 calls in 2015-2024, no return data used). 2,062 calls (18-40/ticker), 869,072 sentences. call_timing: during 919, before_open 651, after_close 483, unknown 9. 91.2% of calls have a prepared/Q&A split (rest = 'unknown' section).
- M2 done: LM scores (dictionary 1993-2025, downloaded 2026-11-06). Sanity: most negative calls = COVID Q1-2020 (MHK, DOV) and EIX 2018 wildfire; most positive incl. TMO 2021Q1. Corr(prepared, Q&A lm_net) = 0.17. 4 lexicon tests pass; ruff clean.
- NOTE for M4: 45% of calls start 09:30-16:00 ET ('during'). PRD rule maps these to next trading day, but many have a pre-market press release, so both alignments are run as robustness.
- M3 code done: FinBERT pinned to ProsusAI/finbert@0bc2ac10, sentence-level, over-long sentences chunked (tokens asserted), per-call parquet cache keyed (call, sent_idx, revision). CPU throughput ~32 sent/s unsorted (12 threads) -> full run of 869k sentences takes several hours; scoring running, resumable. 3 wrapper tests pass.
- M4 done (code): prices for 60 tickers + SPY (yfinance, downloaded 2026-11-06). Events: 4 alignment rules x 2 models. Primary (prd, market model): 2,050 events (9 unknown-timing + 3 short-history dropped, no overlaps). Estimation window [-250,-30] masks other events' [0,+5] days. 14 event tests pass.
- FINDING (needs user decision): 'during' calls (916, 45%) show mean |AR| 3.3% on the call date vs 1.7% the next day, so the PRD rule (during -> next day) misses most of the reaction. 'alt' rule (during -> call date) is the likely better primary; before_open/after_close behave as expected.
