# PRD — Earnings Call Sentiment Analyzer (ML + Quant Finance, Project 2)

**Owner:** Kapish · **Status:** Draft v1 · **Build tool:** Claude Code

## 1. One-line summary
Score earnings-call transcripts with FinBERT, compare against a Loughran-McDonald dictionary baseline, and run an event study testing whether call sentiment predicts abnormal returns over the 1–5 trading days after the call.

## 2. Why this project
- Combines NLP (transformers, long-document handling) with proper finance statistics (event studies, abnormal returns, clustered standard errors).
- The hard parts are not the model: they are **getting transcripts**, **aligning call timestamps to trading days**, and **not fooling yourself** with look-ahead or confounds (earnings surprise). Showing you handled these is the signal.
- An honest null or weak result is acceptable; a clearly explained, correct pipeline is the goal.

## 3. Goals / Non-goals
**Goals**
1. Transcript dataset for a basket of tickers with call date, time, and speaker/section structure.
2. Two sentiment scorers: FinBERT (transformer) and Loughran-McDonald (dictionary).
3. Event study: cumulative abnormal returns (CAR) over [0,+1], [0,+3], [0,+5].
4. Statistical tests: quintile sorts and regression of CAR on sentiment, with robust/clustered SE.
5. Clear comparison: does FinBERT add anything beyond LM?
6. Reproducible repo, README with results and limitations.

**Non-goals**
- Fine-tuning FinBERT (optional stretch only).
- Intraday / audio-tone analysis.
- Building a trading strategy (a simple sentiment-sorted portfolio is a stretch).

## 4. Data
**Transcripts (the main risk, resolve in Phase 0).** Candidate sources, to verify before committing: public datasets on Hugging Face or Kaggle, a financial data API with transcript endpoints, or scraping a public site (check its terms of service first; respect robots.txt and rate limits). Pick the source in Phase 0 and record the choice, licence, and coverage in `docs/data_sources.md`. Required fields: ticker, fiscal quarter, call date, call time (or before-open/after-close flag), full text, and, if possible, a prepared-remarks vs Q&A split.

**Prices:** daily adjusted close via `yfinance` (or equivalent). Benchmark: SPY (or a Fama-French factor file if easy to obtain).

**Lexicon:** Loughran-McDonald Master Dictionary (Notre Dame SRAF); store version and download date.

**Basket:** 30–100 tickers across sectors, ≥ 8 quarters each (more is better for statistical power). Choose by data availability, not by hindsight on returns.

## 5. Methodology

### 5.1 Preprocessing
Clean boilerplate/operator text, split into sections (prepared remarks, Q&A), and split into sentences. Optionally restrict to management speech vs analysts.

### 5.2 Sentiment scoring
- **FinBERT** (`ProsusAI/finbert`): FinBERT has a 512-token limit, so score **sentence by sentence** (or chunk to ≤ 512 tokens), then aggregate. Per-sentence output = P(pos), P(neg), P(neu). Call-level score = mean(P(pos) − P(neg)), also report % positive and % negative sentences. Cache model outputs to disk (inference is the slow step; batch it; use GPU if available).
- **Loughran-McDonald:** (positive − negative words) / total words; also negative-only and uncertainty-word share as variants.
- Compute scores separately for prepared remarks and Q&A (Q&A is less scripted and often more informative).

### 5.3 Event study
- **Day 0 alignment (critical):** if the call is before market open → day 0 = call date; if during/after close → day 0 = next trading day. Store a `call_timing` flag and test both alignments as a robustness check.
- **Expected return model:** market model estimated over a clean pre-event window (e.g. [−250, −30]) per event, with market-adjusted returns as the simple alternative.
- Abnormal return `AR_t = R_t − (α + β·R_mkt,t)`; `CAR[0,k] = Σ AR`.
- Exclude events with overlapping windows or too little estimation data.

### 5.4 Tests
1. **Quintile sort:** mean CAR by sentiment quintile; long-short spread with t-stat.
2. **Regression:** `CAR = a + b·sentiment + controls + ε` with controls for size, prior return/momentum, pre-event volatility, sector and quarter fixed effects; standard errors clustered by firm and/or date.
3. **Horse race:** FinBERT vs LM, each alone and together; compare R² / t-stats and rank correlation between scores.
4. **Robustness:** prepared remarks vs Q&A; window lengths; alignment variants; winsorising CAR.
5. **Confound note:** earnings surprise (actual vs consensus) usually drives the reaction and correlates with tone. If surprise data isn't available, say so explicitly, and do not claim causality.

## 6. Repo structure
```
earnings-sentiment/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── Makefile                    # make data | score | events | analyze | report | test
├── configs/default.yaml
├── data/{raw,interim,processed}/   # git-ignored
├── docs/data_sources.md
├── src/ecs/
│   ├── transcripts.py          # load/clean/split transcripts
│   ├── prices.py
│   ├── lexicon.py              # Loughran-McDonald scorer
│   ├── finbert.py              # batched, cached FinBERT scorer
│   ├── events.py               # day-0 alignment, market model, CAR
│   ├── analysis.py             # sorts, regressions, robustness
│   └── report.py
├── scripts/
├── notebooks/
├── results/{figures,tables}/
└── tests/
```

## 7. Milestones
| # | Milestone | Done when |
|---|---|---|
| M0 | Scaffold + data-source decision | `docs/data_sources.md` filled, sample of transcripts loaded |
| M1 | Transcript ingest + cleaning + sectioning | dataset table with required fields, counts per ticker/quarter |
| M2 | LM baseline scores | scores per call, sanity-checked on known positive/negative calls |
| M3 | FinBERT scoring (cached) | scores per call, runtime documented |
| M4 | Price data + event windows + CAR | CAR table, alignment tests pass |
| M5 | Quintile sorts + regressions | tables with clustered SE |
| M6 | Robustness + horse race | summary table |
| M7 | README + figures | reproducible from clean clone |

## 8. Testing requirements
- Day-0 alignment unit tests: before-open, after-close, call on Friday after close → Monday, holiday handling.
- Estimation window ends strictly before the event window.
- CAR on a synthetic series with a known injected jump equals the jump.
- LM scorer on tiny hand-labelled sentences.
- FinBERT wrapper: deterministic output on fixed sentences; truncation never silently drops text beyond 512 tokens.

## 9. Deliverables
1. Repo + README (what, data, how to run, key results, limitations).
2. Figures: sentiment distribution, FinBERT vs LM scatter, CAR by quintile, average CAR path (days −5…+5) for top vs bottom quintile, coefficient plot.
3. Results table with t-stats and number of events.

## 10. Risks and pitfalls
| Risk | Mitigation |
|---|---|
| Transcript availability/licence | Phase 0 decision; document source; respect ToS |
| Wrong day-0 alignment | Explicit timing flag, tests, both-variant robustness |
| Long documents vs 512-token limit | Sentence-level scoring + aggregation |
| Look-ahead in estimation window | Window ends before event; test it |
| Earnings-surprise confound | Controls where available; state the limitation |
| Small sample, overlapping windows | Report N; cluster SE; avoid overclaiming |
| Multiple testing over windows/variants | Pre-declare primary spec: FinBERT call-level score → CAR[0,+1]; label others exploratory |

## 11. Success criteria
- Pipeline runs end to end on ≥ 30 tickers and ≥ 8 quarters.
- Alignment and estimation-window tests pass.
- Results reported with proper standard errors and an honest conclusion on whether FinBERT beats the dictionary.

## 12. References (from the source post)
- Yang, Uy, Huang (2020), *FinBERT: A Pretrained Language Model for Financial Communications*.
- Loughran & McDonald (2011), *When Is a Liability Not a Liability?*, Journal of Finance.
- ProsusAI/finBERT (GitHub).
- Stefan Jansen, *ML4T*, chapter on word embeddings for earnings calls and SEC filings.
- Event-study method: MacKinlay (1997), *Event Studies in Economics and Finance*.
