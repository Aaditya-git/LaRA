"""
Combine the per-top_k binary + numeric judge CSVs (written by compute_score_llm.py /
compute_score_numeric.py with --top_k) into one comparison table: rows = top_k,
columns = reasoning/comp accuracy, both scales. This is the presentable output for
the retrieval chunk-count (top_k) study -- reasoning is swept across --reasoning_top_ks,
comp stays fixed at --comp_top_k (it only ever needs 2 segments, see RESUME.md Task A).

Usage:
  python aggregate_topk_results.py --eval_model qwen2.5:7b --judge_model qwen2.5:14b \
      --chunker token --reasoning_top_ks 5 10 15 --comp_top_k 2
"""
import argparse
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--eval_model', required=True, type=str)
parser.add_argument('--judge_model', default=None, type=str)
parser.add_argument('--chunker', required=True, type=str)
parser.add_argument('--reasoning_top_ks', nargs='+', type=int, default=[5, 10, 15])
parser.add_argument('--comp_top_k', type=int, default=2)
args = parser.parse_args()

eval_model = args.eval_model
judge_model = args.judge_model or eval_model
_safe_eval = eval_model.replace(':', '-').replace('/', '-')
_safe_judge = judge_model.replace(':', '-').replace('/', '-')


def read_pct(cell, qt):
    """returns (binary_pct, numeric_pct) for one query_type from one cell's CSVs"""
    binary_csv = f'./prediction/result/{cell}_all.csv'
    numeric_csv = f'./prediction/result/numeric_{cell}_all.csv'
    binary_pct = numeric_pct = None
    if os.path.exists(binary_csv):
        bdf = pd.read_csv(binary_csv)
        mask = bdf['Task Name'].str.contains(f'_{qt}$', regex=True)
        val = bdf.loc[mask, 'Accuracy'].mean()
        binary_pct = round(float(val) * 100, 2) if pd.notna(val) else None
    else:
        print(f"WARNING: missing {binary_csv}")
    if os.path.exists(numeric_csv):
        ndf = pd.read_csv(numeric_csv)
        mask = ndf['Task Name'].str.contains(f'_{qt}$', regex=True)
        val = ndf.loc[mask, 'MeanScoreNorm'].mean()
        numeric_pct = round(float(val) * 100, 2) if pd.notna(val) else None
    else:
        print(f"WARNING: missing {numeric_csv}")
    return binary_pct, numeric_pct


rows_by_topk = {}


def row(k):
    return rows_by_topk.setdefault(k, {'top_k': k})


for k in args.reasoning_top_ks:
    cell = f'gen-{_safe_eval}_judge-{_safe_judge}_chunker-{args.chunker}_topk-{k}'
    b, n = read_pct(cell, 'reasoning')
    row(k)['reasoning_binary_pct'] = b
    row(k)['reasoning_numeric_pct'] = n

comp_cell = f'gen-{_safe_eval}_judge-{_safe_judge}_chunker-{args.chunker}_topk-{args.comp_top_k}'
b, n = read_pct(comp_cell, 'comp')
row(args.comp_top_k)['comp_binary_pct'] = b
row(args.comp_top_k)['comp_numeric_pct'] = n

out_df = pd.DataFrame(sorted(rows_by_topk.values(), key=lambda r: r['top_k']))
out_path = './prediction/result/topk_comparison.csv'
out_df.to_csv(out_path, index=False)

print("\n" + "=" * 100)
print(f"TOP_K COMPARISON  (generator={eval_model}  judge={judge_model}  chunker={args.chunker})")
print("=" * 100)
with pd.option_context('display.width', 200, 'display.max_columns', None):
    print(out_df.to_string(index=False))
print(f"\nWritten: {out_path}")
