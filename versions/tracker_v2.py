import pandas as pd
import numpy as np
import argparse
from scipy import stats
from openpyxl.utils import get_column_letter


def load_data(csv_path, sheet_path):
    if csv_path.endswith('.csv'):
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_excel(csv_path)

    grouping = pd.read_excel(sheet_path, sheet_name='grouping')
    agg_config = pd.read_excel(sheet_path, sheet_name='aggregation_config')

    print(f"loaded {len(df):,} rows, {df['symbol'].nunique()} companies, {df['series_key'].nunique()} metrics")
    return df, grouping, agg_config


def get_group_dimensions(agg_config):
    dims = {}
    for col in agg_config.columns:
        if col.lower().startswith('target_'):
            suffix = col[len('target_'):]
            dims[suffix.title()] = col
    print(f"group dimensions found: {list(dims.keys())}")
    return dims


def forward_fill(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'series_key', 'date'])
    df['value'] = (
        df.groupby(['symbol', 'series_key'])['value']
          .transform(lambda x: x.ffill())
    )
    return df


def build_snapshot(df):
    print(f"nulls before ffill: {df['value'].isna().sum():,}")
    df = forward_fill(df)
    print(f"nulls after ffill: {df['value'].isna().sum():,}")

    snapshot = (
        df.dropna(subset=['value'])
          .sort_values('date')
          .groupby(['symbol', 'series_key'])
          .last()
          .reset_index()
          [['symbol', 'series_key', 'value', 'date']]
          .rename(columns={'date': 'as_of_date'})
    )

    print(f"snapshot: {len(snapshot):,} rows, latest date: {snapshot['as_of_date'].max().date()}")
    return snapshot


def build_timeseries(df):
    # full forward-filled panel — used for z-score/percentile lookback
    df = forward_fill(df)
    return df.dropna(subset=['value'])


def aggregate_series(data, group_col, target_col, agg_config, date_col='date'):
    # aggregate long format data by group for a given date column
    # returns long format: date | group | series_key | value
    results = []

    for _, cfg in agg_config.iterrows():
        metric = cfg['series_key']
        agg_method = cfg['agg_method'].strip().lower()
        target = str(cfg.get(target_col, 'all')).strip()

        metric_data = data[data['series_key'] == metric].copy()
        if metric_data.empty:
            continue

        if target.lower() not in ('all', 'nan', ''):
            allowed = [g.strip() for g in target.split(',')]
            metric_data = metric_data[metric_data[group_col].isin(allowed)]
            if metric_data.empty:
                continue

        grouped = metric_data.groupby([date_col, group_col])['value']
        if agg_method == 'sum':
            agg_result = grouped.sum().reset_index()
        elif agg_method == 'median':
            agg_result = grouped.median().reset_index()
        else:
            continue

        agg_result['series_key'] = metric
        agg_result['agg_method'] = agg_method
        results.append(agg_result)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def calc_z_score(series):
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return np.nan
    return (series.iloc[-1] - mean) / std


def calc_percentile(series):
    if len(series) < 2:
        return np.nan
    return stats.percentileofscore(series, series.iloc[-1], kind='rank')


def build_stats(ts_long, agg_config, group_col, latest_date):
    # for each group x metric, compute z-score and percentile over lookback window
    records_latest = []
    records_history = []

    for _, cfg in agg_config.iterrows():
        metric = cfg['series_key']
        lookback_years = cfg.get('lookback_years', 3)
        do_zscore = cfg.get('calc_z_score', False)
        do_pct = cfg.get('calc_percentile', False)
        do_export = cfg.get('export_history', False)

        if not do_zscore and not do_pct and not do_export:
            continue

        cutoff = latest_date - pd.DateOffset(years=int(lookback_years))
        metric_ts = ts_long[
            (ts_long['series_key'] == metric) &
            (ts_long['date'] >= cutoff)
        ].copy()

        if metric_ts.empty:
            continue

        for group_val in metric_ts[group_col].unique():
            grp_ts = metric_ts[metric_ts[group_col] == group_val].sort_values('date')

            if do_export:
                for _, row in grp_ts.iterrows():
                    records_history.append({
                        'date': row['date'].date(),
                        group_col: group_val,
                        'series_key': metric,
                        'agg_method': row.get('agg_method', ''),
                        'value': row['value']
                    })

            latest_row = {
                group_col: group_val,
                'series_key': metric,
                'as_of_date': grp_ts['date'].max().date(),
                'latest_value': grp_ts['value'].iloc[-1],
                'lookback_years': lookback_years,
                'lookback_mean': grp_ts['value'].mean(),
                'lookback_std': grp_ts['value'].std(),
                'lookback_n': len(grp_ts)
            }

            if do_zscore:
                latest_row['z_score'] = calc_z_score(grp_ts['value'])

            if do_pct:
                latest_row['percentile'] = calc_percentile(grp_ts['value'])

            records_latest.append(latest_row)

    df_latest = pd.DataFrame(records_latest) if records_latest else pd.DataFrame()
    df_history = pd.DataFrame(records_history) if records_history else pd.DataFrame()

    return df_latest, df_history


def build_v1_wide(snapshot, grouping, agg_config, group_col, target_col):
    merged = snapshot.merge(
        grouping[['Ticker', group_col]],
        left_on='symbol',
        right_on='Ticker',
        how='inner'
    ).drop(columns='Ticker')

    results = []
    as_of_date = snapshot['as_of_date'].max().date()

    for _, cfg in agg_config.iterrows():
        metric = cfg['series_key']
        agg_method = cfg['agg_method'].strip().lower()
        target = str(cfg.get(target_col, 'all')).strip()

        metric_data = merged[merged['series_key'] == metric].copy()
        if metric_data.empty:
            continue

        if target.lower() not in ('all', 'nan', ''):
            allowed = [g.strip() for g in target.split(',')]
            metric_data = metric_data[metric_data[group_col].isin(allowed)]
            if metric_data.empty:
                continue

        grouped = metric_data.groupby(group_col)['value']
        if agg_method == 'sum':
            agg_result = grouped.sum()
        elif agg_method == 'median':
            agg_result = grouped.median()
        else:
            continue

        tmp = agg_result.reset_index()
        tmp.columns = [group_col, 'value']
        tmp['col_name'] = f"{metric}__{agg_method}"
        results.append(tmp)

    if not results:
        return pd.DataFrame()

    long_result = pd.concat(results, ignore_index=True)
    wide = (
        long_result
        .pivot(index=group_col, columns='col_name', values='value')
        .reset_index()
    )
    wide.columns.name = None
    wide.insert(0, 'as_of_date', as_of_date)
    wide.insert(0, 'group_type', group_col.lower())
    return wide


def run_qa(snapshot, grouping, agg_config, results):
    rows = []

    csv_syms = set(snapshot['symbol'].unique())
    grp_syms = set(grouping['Ticker'].unique())
    rows.append({
        'check': 'company_coverage',
        'status': 'PASS' if csv_syms == grp_syms else 'WARN',
        'detail': f"csv: {len(csv_syms)} | grouping: {len(grp_syms)} | gap: {grp_syms - csv_syms or 'none'}"
    })

    for group_col, output_df in results.items():
        all_groups = set(grouping[group_col].unique())
        result_groups = set(output_df[group_col].unique()) if group_col in output_df.columns else set()
        missing = all_groups - result_groups
        rows.append({
            'check': f'{group_col.lower()}_coverage',
            'status': 'WARN' if missing else 'PASS',
            'detail': f"all: {len(all_groups)} | in output: {len(result_groups)} | excluded by config: {sorted(missing) or 'none'}"
        })

    config_m = set(agg_config['series_key'].unique())
    snap_m = set(snapshot['series_key'].unique())
    rows.append({
        'check': 'metric_coverage',
        'status': 'PASS' if not (config_m - snap_m) else 'WARN',
        'detail': f"config: {len(config_m)} | csv: {len(snap_m)} | missing: {config_m - snap_m or 'none'}"
    })

    rows.append({
        'check': 'latest_date',
        'status': 'INFO',
        'detail': str(snapshot['as_of_date'].max().date())
    })

    null_summary = ' | '.join([f"{k.lower()}_data: {v.isnull().sum().sum()} nulls" for k, v in results.items()])
    rows.append({'check': 'nulls_in_output', 'status': 'INFO', 'detail': null_summary})

    return pd.DataFrame(rows)[['check', 'status', 'detail']]


def autofit(writer):
    for sheet_name in writer.sheets:
        ws = writer.sheets[sheet_name]
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                try:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                except:
                    pass
            ws.column_dimensions[col_letter].width = min(max_len + 4, 35)


def run(csv_path, sheet_path, output_path):
    df_raw, grouping, agg_config = load_data(csv_path, sheet_path)

    group_dims = get_group_dimensions(agg_config)

    snapshot = build_snapshot(df_raw)
    ts = build_timeseries(df_raw)

    latest_date = pd.to_datetime(snapshot['as_of_date'].max())

    v1_results = {}
    stats_latest = {}
    stats_history = {}

    for group_col, target_col in group_dims.items():
        print(f"\nv1 aggregation: {group_col}...")
        wide = build_v1_wide(snapshot, grouping, agg_config, group_col, target_col)
        if not wide.empty:
            v1_results[group_col] = wide
            print(f"  {group_col.lower()}_data: {wide.shape[0]} groups x {wide.shape[1]} cols")

        print(f"building timeseries for {group_col}...")
        ts_merged = ts.merge(
            grouping[['Ticker', group_col]],
            left_on='symbol',
            right_on='Ticker',
            how='inner'
        ).drop(columns='Ticker')

        ts_agg = aggregate_series(ts_merged, group_col, target_col, agg_config, date_col='date')

        if ts_agg.empty:
            continue

        print(f"computing z-scores and percentiles for {group_col}...")
        df_latest, df_history = build_stats(ts_agg, agg_config, group_col, latest_date)

        if not df_latest.empty:
            stats_latest[group_col] = df_latest
            print(f"  stats_latest: {df_latest.shape[0]} rows")
        if not df_history.empty:
            stats_history[group_col] = df_history
            print(f"  stats_history: {df_history.shape[0]} rows")

    print("\nrunning qa checks...")
    qa = run_qa(snapshot, grouping, agg_config, v1_results)
    for _, row in qa.iterrows():
        print(f"  [{row['status']}] {row['check']}: {row['detail']}")

    print(f"\nwriting {output_path}...")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for group_col, df in v1_results.items():
            df.to_excel(writer, sheet_name=f"{group_col.lower()}_data", index=False)

        for group_col, df in stats_latest.items():
            df.to_excel(writer, sheet_name=f"{group_col.lower()}_zscore_pct", index=False)

        for group_col, df in stats_history.items():
            df.to_excel(writer, sheet_name=f"{group_col.lower()}_history", index=False)

        snapshot.to_excel(writer, sheet_name='snapshot_debug', index=False)
        qa.to_excel(writer, sheet_name='qa_checks', index=False)

        autofit(writer)

    print("done")
    return v1_results, stats_latest, stats_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='panel_long.xlsx')
    parser.add_argument('--sheet', default='tracker_config.xlsx')
    parser.add_argument('--output', default='tracker_v2_output.xlsx')
    args = parser.parse_args()
    run(args.csv, args.sheet, args.output)
