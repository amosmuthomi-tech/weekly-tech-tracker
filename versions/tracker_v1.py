import pandas as pd
import numpy as np
import argparse
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


def build_snapshot(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'series_key', 'date'])

    print(f"nulls before ffill: {df['value'].isna().sum():,}")

    df['value'] = (
        df.groupby(['symbol', 'series_key'])['value']
          .transform(lambda x: x.ffill())
    )

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


def aggregate(snapshot, grouping, agg_config, group_col, target_col):
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
            print(f"unknown agg_method {agg_method} for {metric}, skipping")
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


def write_output(output_path, results, snapshot, qa):
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for group_col, df in results.items():
            df.to_excel(writer, sheet_name=f"{group_col.lower()}_data", index=False)
        snapshot.to_excel(writer, sheet_name='snapshot_debug', index=False)
        qa.to_excel(writer, sheet_name='qa_checks', index=False)

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
    df, grouping, agg_config = load_data(csv_path, sheet_path)

    group_dims = get_group_dimensions(agg_config)
    snapshot = build_snapshot(df)

    results = {}
    for group_col, target_col in group_dims.items():
        print(f"\naggregating by {group_col}...")
        output = aggregate(snapshot, grouping, agg_config, group_col, target_col)
        if not output.empty:
            results[group_col] = output
            print(f"  {group_col.lower()}_data: {output.shape[0]} groups x {output.shape[1]} cols")

    print("\nrunning qa checks...")
    qa = run_qa(snapshot, grouping, agg_config, results)
    for _, row in qa.iterrows():
        print(f"  [{row['status']}] {row['check']}: {row['detail']}")

    print(f"\nwriting {output_path}...")
    write_output(output_path, results, snapshot, qa)
    print("done")

    for group_col, output in results.items():
        print(f"\n-- {group_col.lower()}_data --")
        preview = [c for c in ['group_type', 'as_of_date', group_col,
                                'revenue_ttm__sum', 'pe_forward__median']
                   if c in output.columns]
        print(output[preview].to_string(index=False))

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='panel_long.xlsx')
    parser.add_argument('--sheet', default='tracker_config.xlsx')
    parser.add_argument('--output', default='tracker_v1_output.xlsx')
    args = parser.parse_args()
    run(args.csv, args.sheet, args.output)
