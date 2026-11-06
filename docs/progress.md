# Progress

- Scaffold created, py3.12 venv, remote set.
- M0: candidate transcript source probed (kurry/sp500_earnings_transcripts, 33,362 calls, 685 tickers, MIT). See docs/data_sources.md. Awaiting user confirmation before ingestion.
- M1 done: basket = 60 tickers (seeded, sector-stratified, >=16 calls in 2015-2024, no return data used). 2,062 calls (18-40/ticker), 869,072 sentences. call_timing: during 919, before_open 651, after_close 483, unknown 9. 91.2% of calls have a prepared/Q&A split (rest = 'unknown' section).
- M2 done: LM scores (dictionary 1993-2025, downloaded 2026-11-06). Sanity: most negative calls = COVID Q1-2020 (MHK, DOV) and EIX 2018 wildfire; most positive incl. TMO 2021Q1. Corr(prepared, Q&A lm_net) = 0.17. 4 lexicon tests pass; ruff clean.
- NOTE for M4: 45% of calls start 09:30-16:00 ET ('during'). PRD rule maps these to next trading day, but many have a pre-market press release, so both alignments are run as robustness.
- M3 code done: FinBERT pinned to ProsusAI/finbert@0bc2ac10, sentence-level, over-long sentences chunked (tokens asserted), per-call parquet cache keyed (call, sent_idx, revision). CPU throughput ~32 sent/s unsorted (12 threads) -> full run of 869k sentences takes several hours; scoring running, resumable. 3 wrapper tests pass.
- M4 done (code): prices for 60 tickers + SPY (yfinance, downloaded 2026-11-06). Events: 4 alignment rules x 2 models. Primary (prd, market model): 2,050 events (9 unknown-timing + 3 short-history dropped, no overlaps). Estimation window [-250,-30] masks other events' [0,+5] days. 14 event tests pass.
- FINDING (needs user decision): 'during' calls (916, 45%) show mean |AR| 3.3% on the call date vs 1.7% the next day, so the PRD rule (during -> next day) misses most of the reaction. 'alt' rule (during -> call date) is the likely better primary; before_open/after_close behave as expected.
- DECISION (user, 2026-11-06): primary day-0 rule = 'alt' (during -> call date). PRD rule 'prd' is a labelled robustness variant. Primary spec: FinBERT call-level fb_net (all sections) -> CAR[0,+1], market model, rule alt.
- M5/M6 code done (analysis.py, make_analysis.py, report.py, make_report.py): quintile sorts, regressions (firm / two-way / HC1 SE), horse race, 288-regression exploratory robustness grid; 4 synthetic-planted-effect tests pass. Waiting on FinBERT scoring to produce real results.
- M3 done: FinBERT scored all 2,062 calls (869,072 sentences), rev 0bc2ac10, ~5 h wall-clock on CPU (12 threads), cache in data/interim/finbert_cache.
- M5/M6 done: PRIMARY FinBERT -> CAR[0,+1] = +0.88pp, SE 0.17 (firm cluster), t=5.15, N=2,050, 60 firms. Two-way cluster t=5.01, HC1 t=5.95. Q5-Q1 spread FinBERT +1.89pp (t=4.0), LM +2.63pp (t=5.1). Horse race: LM absorbs FinBERT (both: FB t=1.49, LM t=3.62). Spearman(FB,LM)=0.76. 288 exploratory regressions, 79% p<0.05. Move is on day 0, no drift.
- M7 done: README, 6 figures, tables in results/tables. 25 tests pass, ruff clean.
