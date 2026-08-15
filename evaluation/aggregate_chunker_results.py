"""
Combine the per-chunker binary + numeric judge CSVs (written by compute_score_llm.py /
compute_score_numeric.py with --chunker) into one comparison table: rows = chunker,
columns = per-query-type accuracy, both scales. This is the presentable output for
the Chonkie chunker comparison.

Usage:
  python aggregate_chunker_results.py --eval_model qwen2.5:7b --judge_model qwen2.5:14b \
      --chunkers token sentence recursive semantic fast
"""
import argparse
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--eval_model', required=True, type=str)
parser.add_argument('--judge_model', default=None, type=str)
parser.add_argument('--chunkers', nargs='+', default=['token', 'sentence', 'recursive', 'semantic', 'fast'])
args = parser.parse_args()

eval_model = args.eval_model
judge_model = args.judge_model or eval_model
_safe_eval = eval_model.replace(':', '-').replace('/', '-')
_safe_judge = judge_model.replace(':', '-').replace('/', '-')

QUERY_TYPES = ['location', 'reasoning', 'comp', 'hallu']

rows = []
for chunker in args.chunkers:
    cell = f'gen-{_safe_eval}_judge-{_safe_judge}_chunker-{chunker}'
    binary_csv = f'./prediction/result/{cell}_all.csv'
    numeric_csv = f'./prediction/result/numeric_{cell}_all.csv'

    row = {'chunker': chunker}

    if os.path.exists(binary_csv):
        bdf = pd.read_csv(binary_csv)
        for qt in QUERY_TYPES:
            mask = bdf['Task Name'].str.contains(f'_{qt}$', regex=True)
            val = bdf.loc[mask, 'Accuracy'].mean()
            row[f'{qt}_binary_pct'] = round(float(val) * 100, 2) if pd.notna(val) else None
        vals = [row[f'{qt}_binary_pct'] for qt in QUERY_TYPES if row.get(f'{qt}_binary_pct') is not None]
        row['overall_binary_pct'] = round(sum(vals) / len(vals), 2) if vals else None
    else:
        print(f"WARNING: missing {binary_csv} (run compute_score_llm.py --chunker {chunker} first)")

    if os.path.exists(numeric_csv):
        ndf = pd.read_csv(numeric_csv)
        for qt in QUERY_TYPES:
            mask = ndf['Task Name'].str.contains(f'_{qt}$', regex=True)
            val = ndf.loc[mask, 'MeanScoreNorm'].mean()
            row[f'{qt}_numeric_pct'] = round(float(val) * 100, 2) if pd.notna(val) else None
        vals = [row[f'{qt}_numeric_pct'] for qt in QUERY_TYPES if row.get(f'{qt}_numeric_pct') is not None]
        row['overall_numeric_pct'] = round(sum(vals) / len(vals), 2) if vals else None
    else:
        print(f"WARNING: missing {numeric_csv} (run compute_score_numeric.py --chunker {chunker} first)")

    rows.append(row)

out_df = pd.DataFrame(rows)
out_path = './prediction/result/chunker_comparison.csv'
out_df.to_csv(out_path, index=False)

print("\n" + "=" * 100)
print(f"CHUNKER COMPARISON  (generator={eval_model}  judge={judge_model})")
print("=" * 100)
with pd.option_context('display.width', 200, 'display.max_columns', None):
    print(out_df.to_string(index=False))
print(f"\nWritten: {out_path}")
