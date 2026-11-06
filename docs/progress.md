# Progress

- Scaffold created, py3.12 venv, remote set.
- M0: candidate transcript source probed (kurry/sp500_earnings_transcripts, 33,362 calls, 685 tickers, MIT). See docs/data_sources.md. Awaiting user confirmation before ingestion.
- M1 done: basket = 60 tickers (seeded, sector-stratified, >=16 calls in 2015-2024, no return data used). 2,062 calls (18-40/ticker), 869,072 sentences. call_timing: during 919, before_open 651, after_close 483, unknown 9. 91.2% of calls have a prepared/Q&A split (rest = 'unknown' section).
- M2 done: LM scores (dictionary 1993-2025, downloaded 2026-11-06). Sanity: most negative calls = COVID Q1-2020 (MHK, DOV) and EIX 2018 wildfire; most positive incl. TMO 2021Q1. Corr(prepared, Q&A lm_net) = 0.17. 4 lexicon tests pass; ruff clean.
- NOTE for M4: 45% of calls start 09:30-16:00 ET ('during'). PRD rule maps these to next trading day, but many have a pre-market press release, so both alignments are run as robustness.
