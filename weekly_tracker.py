"""Config-driven weekly tracker - generic demo on SYNTHETIC data.

Long panel (symbol, series_key, date, value) + config (grouping, aggregation_config)
-> forward-filled (never backfilled) snapshot per symbol x metric
-> sum/median aggregation per group dimension (any `target_*` column in config)
-> z-score and percentile over a per-metric lookback window
-> Excel workbook with QA checks.

Run `python weekly_tracker.py --demo` to generate sample inputs and outputs.
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from openpyxl.utils import get_column_letter


def load_data(panel_path, config_path):
    df = pd.read_csv(panel_path) if panel_path.endswith(".csv") else pd.read_excel(panel_path)
    grouping = pd.read_excel(config_path, sheet_name="grouping")
    agg_config = pd.read_excel(config_path, sheet_name="aggregation_config")
    return df, grouping, agg_config


def group_dimensions(agg_config):
    """target_basket -> 'Basket'. Add a target_region column and a Region tab appears."""
    return {c[len("target_"):].title(): c for c in agg_config.columns if c.lower().startswith("target_")}


def forward_fill(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "series_key", "date"])
    df["value"] = df.groupby(["symbol", "series_key"])["value"].ffill()  # never bfill: no lookahead
    return df.dropna(subset=["value"])


def build_snapshot(ts):
    snap = (ts.sort_values("date").groupby(["symbol", "series_key"]).last().reset_index()
              [["symbol", "series_key", "value", "date"]].rename(columns={"date": "as_of_date"}))
    return snap


def _select(data, cfg, group_col, target_col):
    d = data[data["series_key"] == cfg["series_key"]]
    target = str(cfg.get(target_col, "all")).strip()
    if target.lower() not in ("all", "nan", ""):
        d = d[d[group_col].isin([g.strip() for g in target.split(",")])]
    return d


def aggregate_snapshot(snap, grouping, agg_config, group_col, target_col):
    merged = snap.merge(grouping[["Ticker", group_col]], left_on="symbol", right_on="Ticker").drop(columns="Ticker")
    parts = []
    for _, cfg in agg_config.iterrows():
        d = _select(merged, cfg, group_col, target_col)
        method = cfg["agg_method"].strip().lower()
        if d.empty or method not in ("sum", "median"):
            continue
        r = getattr(d.groupby(group_col)["value"], method)().reset_index()
        r["col"] = f"{cfg['series_key']}__{method}"
        parts.append(r)
    if not parts:
        return pd.DataFrame()
    wide = pd.concat(parts).pivot(index=group_col, columns="col", values="value").reset_index()
    wide.columns.name = None
    wide.insert(0, "as_of_date", snap["as_of_date"].max().date())
    return wide


def aggregate_timeseries(ts, grouping, agg_config, group_col, target_col):
    merged = ts.merge(grouping[["Ticker", group_col]], left_on="symbol", right_on="Ticker").drop(columns="Ticker")
    parts = []
    for _, cfg in agg_config.iterrows():
        d = _select(merged, cfg, group_col, target_col)
        method = cfg["agg_method"].strip().lower()
        if d.empty or method not in ("sum", "median"):
            continue
        r = getattr(d.groupby(["date", group_col])["value"], method)().reset_index()
        r["series_key"] = cfg["series_key"]
        parts.append(r)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_stats(ts_agg, agg_config, group_col, latest_date):
    rows = []
    for _, cfg in agg_config.iterrows():
        if not (cfg.get("calc_z_score") or cfg.get("calc_percentile")):
            continue
        cutoff = latest_date - pd.DateOffset(years=int(cfg.get("lookback_years", 3)))
        m = ts_agg[(ts_agg["series_key"] == cfg["series_key"]) & (ts_agg["date"] >= cutoff)]
        for grp, g in m.groupby(group_col):
            v = g.sort_values("date")["value"]
            row = {group_col: grp, "series_key": cfg["series_key"], "as_of_date": g["date"].max().date(),
                   "latest_value": v.iloc[-1], "lookback_n": len(v)}
            sd = v.std()
            if cfg.get("calc_z_score"):
                row["z_score"] = (v.iloc[-1] - v.mean()) / sd if sd and not np.isnan(sd) else np.nan
            if cfg.get("calc_percentile") and len(v) > 1:
                row["percentile"] = stats.percentileofscore(v, v.iloc[-1], kind="rank")
            rows.append(row)
    return pd.DataFrame(rows)


def run_qa(snap, grouping, agg_config, outputs):
    rows = [{"check": "company_coverage",
             "status": "PASS" if set(snap["symbol"]) == set(grouping["Ticker"]) else "WARN",
             "detail": f"data {snap['symbol'].nunique()} | grouping {grouping['Ticker'].nunique()}"},
            {"check": "metric_coverage",
             "status": "PASS" if set(agg_config["series_key"]) <= set(snap["series_key"]) else "WARN",
             "detail": f"missing: {sorted(set(agg_config['series_key']) - set(snap['series_key'])) or 'none'}"},
            {"check": "latest_date", "status": "INFO", "detail": str(snap["as_of_date"].max().date())}]
    for g, out in outputs.items():
        rows.append({"check": f"{g.lower()}_nulls", "status": "INFO", "detail": int(out.isna().sum().sum())})
    return pd.DataFrame(rows)


def autofit(writer):
    for ws in writer.sheets.values():
        for col in ws.columns:
            width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(width + 4, 35)


def run(panel_path, config_path, output_path):
    raw, grouping, agg_config = load_data(panel_path, config_path)
    ts = forward_fill(raw)
    snap = build_snapshot(ts)
    latest = pd.to_datetime(snap["as_of_date"].max())
    wides, zs = {}, {}
    for group_col, target_col in group_dimensions(agg_config).items():
        w = aggregate_snapshot(snap, grouping, agg_config, group_col, target_col)
        if not w.empty:
            wides[group_col] = w
        agg = aggregate_timeseries(ts, grouping, agg_config, group_col, target_col)
        if not agg.empty:
            z = build_stats(agg, agg_config, group_col, latest)
            if not z.empty:
                zs[group_col] = z
    qa = run_qa(snap, grouping, agg_config, wides)
    with pd.ExcelWriter(output_path, engine="openpyxl") as xw:
        for g, d in wides.items():
            d.to_excel(xw, sheet_name=f"{g.lower()}_data", index=False)
        for g, d in zs.items():
            d.to_excel(xw, sheet_name=f"{g.lower()}_zscore_pct", index=False)
        snap.to_excel(xw, sheet_name="snapshot_debug", index=False)
        qa.to_excel(xw, sheet_name="qa_checks", index=False)
        autofit(xw)
    return wides, zs, qa


def make_demo_inputs(panel_path="sample_panel.csv", config_path="sample_config.xlsx", seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=800)
    symbols = [f"CO{i:02d}" for i in range(12)]
    grouping = pd.DataFrame({"Ticker": symbols,
                             "Basket": ["Alpha", "Beta", "Gamma"] * 4,
                             "Vertical": ["V1", "V2", "V3", "V4"] * 3})
    rows = []
    for s in symbols:
        for key, base, vol in [("revenue_ttm", 100, 0.002), ("pe_forward", 25, 0.01)]:
            v = base * np.exp(np.cumsum(rng.normal(0, vol, len(dates))))
            if key == "revenue_ttm":                      # quarterly-updated series
                v = pd.Series(v, index=dates).where(np.arange(len(dates)) % 63 == 0)
            else:
                v = pd.Series(v, index=dates)
            rows.append(pd.DataFrame({"symbol": s, "series_key": key, "date": dates, "value": v.values}))
    pd.concat(rows).to_csv(panel_path, index=False)
    cfg = pd.DataFrame({"series_key": ["revenue_ttm", "pe_forward"], "agg_method": ["sum", "median"],
                        "target_basket": ["all", "all"], "target_vertical": ["all", "V1,V2"],
                        "calc_z_score": [True, True], "calc_percentile": [True, True],
                        "lookback_years": [2, 2]})
    with pd.ExcelWriter(config_path) as w:
        grouping.to_excel(w, sheet_name="grouping", index=False)
        cfg.to_excel(w, sheet_name="aggregation_config", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="sample_panel.csv")
    ap.add_argument("--config", default="sample_config.xlsx")
    ap.add_argument("--output", default="tracker_output.xlsx")
    ap.add_argument("--demo", action="store_true", help="generate synthetic inputs first")
    a = ap.parse_args()
    if a.demo:
        make_demo_inputs(a.panel, a.config)
    w, z, q = run(a.panel, a.config, a.output)
    print(q.to_string(index=False))
    for g, d in z.items():
        print(f"\n{g} z-scores/percentiles\n", d.round(2).head(6).to_string(index=False))
