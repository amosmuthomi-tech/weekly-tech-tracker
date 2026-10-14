# Weekly Tech Tracker (generic demo)

A config-driven Python pipeline that turns a long panel of company metrics into group-level snapshots and z-score/percentile views, written to an Excel workbook with QA checks.

I built this for a US quantitative research firm, who agreed to publish the code. Their data and configuration are not included.

- `versions/tracker_v1.py` to `tracker_v3.py` — the delivered scripts (v1 aggregation, v2 z-score/percentile, v3 correlation vs VIX / 10Y yield / NFCI)
- `weekly_tracker.py` — a runnable demo on **synthetic data**

## Design

- **No hardcoded groups:** any `target_*` column in `aggregation_config` becomes a group dimension and gets its own output tab
- **Point-in-time safe:** forward-fill only, never backfill, per company and metric
- **Independent metric dates:** quarterly and daily series each use their own latest value
- **Per-metric config:** aggregation (`sum`/`median`), `calc_z_score`, `calc_percentile`, `lookback_years`
- **QA tab:** company and metric coverage, latest date, null counts

## Run it

```bash
pip install -r requirements.txt
python weekly_tracker.py --demo
```

`--demo` generates `sample_panel.csv` and `sample_config.xlsx` (synthetic) and writes `tracker_output.xlsx`.

## Input format

- Panel: `symbol, series_key, date, value`
- Config workbook: `grouping` sheet (`Ticker`, one column per dimension) and `aggregation_config` sheet (`series_key`, `agg_method`, `target_*`, flags)

## Limitations

- Z-scores assume a roughly stable distribution over the lookback; trending series will read as extreme
- Aggregating sums across companies with different fiscal calendars needs care
- No scheduling or tests yet
