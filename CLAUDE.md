# CLAUDE.md — Earnings Call Sentiment Analyzer

## What this project is
Scores earnings-call transcripts with FinBERT and a Loughran-McDonald dictionary, then runs an event study to test whether sentiment predicts post-earnings abnormal returns (1–5 days). Read `PRD.md` before any task.

## Stack
Python 3.12, pandas, numpy, scipy, statsmodels, transformers, torch, nltk or spaCy (sentence splitting only), yfinance, matplotlib, pyyaml, pytest, ruff. Ask before adding anything else.

## Commands
```
make setup
make data       # ingest + clean transcripts, download prices
make score      # LM + FinBERT (FinBERT results cached)
make events     # day-0 alignment, market model, CAR
make analyze    # sorts, regressions, robustness
make report
make test
make lint
```

## Layout
`src/ecs/`: transcripts, prices, lexicon, finbert, events, analysis, report. Parameters in `configs/default.yaml`. `data/` is git-ignored. `docs/data_sources.md` records where every dataset came from, its licence, and download date.

## Non-negotiable rules
1. **Day-0 alignment is the most error-prone step.** Before open → call date; after close → next trading day. Always keep the `call_timing` flag; never guess silently. If timing is unknown, mark the event and run both variants.
2. The market-model estimation window must end strictly **before** the event window. Test it.
3. No look-ahead: nothing after the event may feed the sentiment score, controls, or the estimation window.
4. FinBERT truncates at 512 tokens. Never feed a whole transcript. Score sentence-by-sentence or in ≤ 512-token chunks, and assert no text is silently dropped.
5. Cache FinBERT outputs to disk keyed by (call id, sentence index, model revision). Do not re-run inference needlessly. Pin the model revision.
6. Pre-declare the primary spec (call-level FinBERT score → CAR[0,+1]). Everything else is labelled exploratory. Don't hunt for the window or variant that "works".
7. Standard errors: heteroskedasticity-robust at minimum; cluster by firm and/or date. Always print N.
8. Respect data licences and site terms of service when sourcing transcripts; rate-limit any scraping. Never commit transcript data.
9. Do not claim causality. State the earnings-surprise confound unless surprise data is controlled for.

## Conventions
- Type hints; docstrings state the timing of every date-based function.
- Deterministic: seeds in config; fixed model revision; batch inference with `torch.no_grad()`.
- Intermediate files as parquet in `data/interim`; schemas documented in `docs/`.
- Trading calendar: use a proper exchange calendar (e.g. `pandas_market_calendars`) or the price index itself; no calendar-day arithmetic for event windows.
- Plots: PNG in `results/figures/`, labelled axes, N in the title/caption.
- Use `logging`, not `print`.

## Testing
- `test_events_alignment.py`: before-open, after-close, Friday-after-close, holiday cases.
- `test_estimation_window.py`: ends before event window.
- `test_car.py`: synthetic series with injected jump.
- `test_lexicon.py`: tiny hand-labelled sentences.
- `test_finbert_wrapper.py`: chunking covers 100% of tokens; deterministic output.
Run `make test` and `make lint` before marking a milestone done.

## Workflow
1. Milestone by milestone (M0 → M7). Show evidence at each (counts, sample rows, test output).
2. Phase 0 first: confirm the transcript source with the user before building ingestion. If coverage is poor, stop and ask rather than quietly shrinking the sample.
3. Ask before changing the primary spec, event windows, or sample.
4. Keep notebooks exploratory only; production logic lives in `src/`.
5. Record progress in `docs/progress.md`.

## Definition of done
Code + tests + lint clean + progress note. For analysis milestones, also a results table with N, estimates, and standard errors.

## Reporting rules
Every effect reported with N, t-stat/SE, and the clustering used. Say plainly when results are insignificant. Include limitations: sample size, survivorship in the ticker basket, missing surprise control, overlapping windows.

## Out of scope
Fine-tuning FinBERT (stretch only, ask first), audio/tone analysis, live trading.
