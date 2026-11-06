# Schemas (data/interim, data/processed — git-ignored)

**calls.parquet** — one row per call: `call_id` (TICKER_YYYYQn, fiscal), `ticker`, `company`, `fiscal_year`, `fiscal_quarter`,
`call_dt` (source timestamp, assumed US/Eastern), `call_date`, `call_timing` (before_open <09:30 | during 09:30-16:00 |
after_close >=16:00 | unknown), `sector` (yfinance, as of 2026-11-06), `has_qa`, `n_turns`.

**sentences.parquet** — `call_id`, `sent_idx` (stable order within call), `section` (prepared | qa | unknown), `speaker`, `text`.
Operator turns dropped; sentences < 3 words dropped.

**scores_lm.parquet** — `call_id`, `section` (all | prepared | qa | unknown), `lm_net`, `lm_pos`, `lm_neg`, `lm_unc`, `n_words`.
