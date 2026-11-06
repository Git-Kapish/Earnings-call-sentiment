# Earnings Call Sentiment Analyzer

Scores earnings-call transcripts with **FinBERT** (`ProsusAI/finbert`) and the **Loughran-McDonald (LM)** dictionary, then runs an event study: does call tone line up with abnormal returns over the 1-5 trading days after the call, and does FinBERT add anything beyond LM?

**Short answer.** Tone from both scorers is strongly associated with the market reaction (primary spec: t = 5.1, N = 2,050). That reaction is **concentrated on day 0** and does not drift afterwards. **FinBERT does not beat the dictionary**: LM is at least as strong alone and absorbs most of FinBERT's effect when both are included. We have **no earnings-surprise data**, so none of this is causal; tone almost certainly proxies for the surprise itself.

## Data
| | |
|---|---|
| Transcripts | `kurry/sp500_earnings_transcripts` (Hugging Face, MIT, "research/educational use only"; original provenance not stated). Not redistributed. |
| Basket | 60 tickers, sector-stratified (seeded), >= 16 calls in 2015-2024, chosen without looking at returns. 2,062 calls, 869,072 sentences. |
| Prices | `yfinance` adjusted close, benchmark SPY (downloaded 2026-11-06). |
| Lexicon | LM Master Dictionary 1993-2025 (Notre Dame SRAF, downloaded 2026-11-06). |

Details, licences and caveats: `docs/data_sources.md`; table schemas: `docs/schemas.md`; build log: `docs/progress.md`.

## Method (short)
- **Sentiment.** FinBERT is scored sentence by sentence (over-long sentences chunked, token coverage asserted), call score = mean(P(pos) - P(neg)). LM score = (pos - neg words) / words. Both are computed for all text, prepared remarks and Q&A. FinBERT output is cached per call at a pinned model revision (CPU run: about 5 h wall-clock for 869k sentences).
- **Day 0.** Call timing is derived from the call's start time (assumed US Eastern, not documented by the source). **Primary rule:** before-open and *during-market* calls -> call date; after-close -> next trading day. The PRD rule (during -> next day) is kept as a robustness variant. *Why:* for the 916 events that start during market hours, mean |AR| is 3.3% on the call date vs 1.7% the next day, so the PRD rule would miss most of the reaction. This was decided from reaction size, before any sentiment-vs-return result existed, and confirmed with the project owner.
- **Abnormal returns.** Market model vs SPY, estimated on trading days [-250, -30] (other earnings reactions masked out, >= 120 obs); CAR[0,+k] summed from day 0. Market-adjusted returns as an alternative. The estimation window ends strictly before day 0 (tested).
- **Primary spec (declared in advance):** FinBERT call-level score (all sections) -> CAR[0,+1], OLS with prior return, pre-event volatility, log dollar volume (all measured on days [-60,-2]), sector and year-quarter fixed effects, SE clustered by firm. **Everything else is exploratory.**

## Results (N shown everywhere; tone z-scored, so coefficients are per +1 s.d.)

**Primary (FinBERT -> CAR[0,+1]):** +0.88 pp, SE 0.17 (firm-clustered), t = 5.15, N = 2,050 events, 60 firms, adj. R² 0.017. Same coefficient with firm x date two-way clustering: t = 5.01; HC1: t = 5.95.

**Quintile sort (FinBERT, CAR[0,+1]):** Q1 -0.75%, Q5 +1.14%; Q5-Q1 spread +1.89 pp, firm-clustered t = 4.0 (820 events in the two extreme quintiles). LM spread: +2.63 pp, t = 5.1. Quintiles are pooled over time, so this is a descriptive sort, not a tradable strategy.

**Horse race, CAR[0,+1] (N = 2,050, firm-clustered SE):**
| Model | FinBERT coef (t) | LM coef (t) | adj. R² |
|---|---|---|---|
| FinBERT only | 0.88 pp (5.15) | | 0.0174 |
| LM only | | 1.00 pp (5.32) | 0.0224 |
| Both | 0.28 pp (1.49) | 0.79 pp (3.62) | 0.0227 |

Spearman correlation between the two scores: 0.76. **FinBERT adds nothing detectable beyond LM here**; LM stays significant when both are included, FinBERT does not.

**Robustness (exploratory, 288 regressions: 4 alignment rules x 2 return models x 3 sections x 3 windows x winsorised or not x 2 scorers).** Share with p < 0.05: 79% overall (FinBERT 74%, LM 85%). Selected, CAR[0,+1], market model, firm-clustered:
| Day-0 rule | Section | FinBERT t | LM t |
|---|---|---|---|
| alt (primary) | all | 5.1 | 5.3 |
| alt | prepared / Q&A | 3.7 / 3.5 | 2.9 / 3.6 |
| prd (PRD rule) | all | 4.2 | 4.4 |
| next-day always | all | 2.6 | 3.4 |
Longer windows give similar coefficients (CAR[0,+3], [0,+5]: about 0.8 pp, t of 3.6 to 4.2). Winsorising CAR at 1/99% changes nothing material. Prepared remarks and Q&A are each informative, with weaker effects than the pooled text. With 288 tests, treat anything beyond the primary spec as descriptive.

**Path.** The tone-sorted gap in average abnormal returns opens on day 0 and then stays flat or partly reverses (`results/figures/car_path_finbert.png`). That pattern fits a same-day reaction to the news in the call, not a post-call drift that could be traded on.

Tables: `results/tables/`. Figures: `results/figures/` (sentiment distribution, FinBERT vs LM scatter, CAR by quintile for both scorers, CAR path top vs bottom quintile, horse-race coefficient plot).

## Limitations
- **No earnings-surprise control (the key confound).** Tone co-moves with actual-vs-consensus surprise, which drives the reaction. The estimates above are associations, **not causal effects**, and may overstate any tone-specific information.
- **Survivorship / selection.** Basket is drawn from the current S&P 500 list with a currently available sector, and requires 16+ calls in 2015-2024; companies that left the index or delisted are underrepresented.
- **Single source, unverified provenance.** Call times are assumed US Eastern (supported by the after-close reaction pattern, but not documented). 9 calls with no time (flagged `unknown`) are excluded from the PRD/alt rules; 3 more lack estimation data. 9% of calls have no detectable prepared/Q&A split (section `unknown`).
- **Sample size / dependence.** 2,050 events, 60 firms; calendar clustering of reporting seasons and a COVID-heavy 2020 may inflate precision despite firm and two-way clustering.
- **Overlapping windows.** None dropped under the 6-day rule (calls are about a quarter apart); other events' days are masked from estimation windows.
- **Controls are proxies.** Size is proxied by trailing dollar volume (no look-ahead market-cap data).
- **Sentence splitting** (NLTK Punkt) is imperfect on abbreviations; FinBERT is not fine-tuned on call language.
- **Multiple testing.** One pre-declared primary spec; 288 exploratory regressions reported.

## Reproduce
```
python -m venv .venv            # Python 3.12
make setup
# LM dictionary: download the Master Dictionary CSV to data/raw/LM_MasterDictionary.csv (see docs/data_sources.md)
make data                       # basket, transcripts, prices (needs network)

make score                      # LM + FinBERT (hours on CPU; cached and resumable)
make events                     # day-0 alignment, market model, CAR
make analyze                    # sorts, regressions, robustness
make report                     # figures + summary table
make test && make lint
```
`data/` is git-ignored; transcripts are never committed. Parameters in `configs/default.yaml`.

## References
Yang, Uy, Huang (2020) FinBERT; Loughran & McDonald (2011) *When Is a Liability Not a Liability?*; MacKinlay (1997) *Event Studies in Economics and Finance*.
