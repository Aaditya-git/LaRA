"""
Combine the binary (compute_score_llm.py) and numeric (compute_score_numeric.py)
per-question debug logs into one log, so every question shows both judging scales
side by side instead of living in two separate files.

Does not re-judge anything: both scripts already ran (LARA_JUDGE_DEBUG=1) and wrote
judge_debug_<CELL>.jsonl / numeric_judge_debug_<CELL>.jsonl. This just joins them by
(tag, question), in order, since both scripts iterate the same prediction files in
the same order.

Usage:
  python merge_judgments.py --eval_model qwen2.5:7b --judge_model qwen2.5:14b
"""
import argparse
import json
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--eval_model', required=True, type=str)
parser.add_argument('--judge_model', default=None, type=str)
parser.add_argument('--chunker', default=None, type=str)
parser.add_argument('--top_k', default=None, type=int)
args = parser.parse_args()

eval_model = args.eval_model
judge_model = args.judge_model or eval_model
chunker = args.chunker
top_k = args.top_k
_safe_eval = eval_model.replace(':', '-').replace('/', '-')
_safe_judge = judge_model.replace(':', '-').replace('/', '-')
CELL = f'gen-{_safe_eval}_judge-{_safe_judge}' + (f'_chunker-{chunker}' if chunker else '') + (f'_topk-{top_k}' if top_k else '')

BINARY_LOG = f'./prediction/result/judge_debug_{CELL}.jsonl'
NUMERIC_LOG = f'./prediction/result/numeric_judge_debug_{CELL}.jsonl'
OUT_LOG = f'./prediction/result/combined_judgments_{CELL}.jsonl'

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

binary_rows = load_jsonl(BINARY_LOG)
numeric_rows = load_jsonl(NUMERIC_LOG)

if not binary_rows and not numeric_rows:
    print(f"No debug logs found for {CELL} (expected {BINARY_LOG} and/or {NUMERIC_LOG}).")
    print("Re-run compute_score_llm.py / compute_score_numeric.py with LARA_JUDGE_DEBUG=1 first.")
    raise SystemExit(0)

# group by tag, preserving within-tag order, so identically-worded questions in
# different tags/configs don't get cross-matched
def by_tag(rows):
    out = defaultdict(list)
    for r in rows:
        out[r['tag']].append(r)
    return out

binary_by_tag = by_tag(binary_rows)
numeric_by_tag = by_tag(numeric_rows)

all_tags = list(dict.fromkeys(list(binary_by_tag.keys()) + list(numeric_by_tag.keys())))

def _short(text, limit=500):
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"

combined = []
mismatches = 0
for tag in all_tags:
    b_list = binary_by_tag.get(tag, [])
    n_list = numeric_by_tag.get(tag, [])
    n = max(len(b_list), len(n_list))
    for i in range(n):
        b = b_list[i] if i < len(b_list) else None
        m = n_list[i] if i < len(n_list) else None
        if b and m and b['question'] != m['question']:
            mismatches += 1
        record = {
            "tag": tag,
            "question": (b or m)["question"],
            "expected": (b or m)["expected"],
            "got": (b or m)["got"],
            "binary_verdict": b["verdict"] if b else None,
            "binary_judge_raw": b["judge_raw"] if b else None,
            "numeric_score": m["score"] if m else None,
            "numeric_reason": m["reason"] if m else None,
            "numeric_binary_verdict": m["binary_verdict"] if m else None,
        }
        combined.append(record)

os.makedirs(os.path.dirname(OUT_LOG), exist_ok=True)
with open(OUT_LOG, 'w', encoding='utf-8') as f:
    for r in combined:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n{'=' * 70}\nCOMBINED JUDGMENTS: generator={eval_model}  judge={judge_model}\n{'=' * 70}")
for r in combined:
    print(
        "\n" + "-" * 70 + "\n"
        + f"[{r['tag']}]\n"
        + f"QUESTION:      {_short(r['question'], 300)}\n"
        + f"EXPECTED:      {_short(r['expected'])}\n"
        + f"GOT:           {_short(r['got'])}\n"
        + f"BINARY JUDGE:  {r['binary_verdict']}\n"
        + f"NUMERIC JUDGE: {r['numeric_score']}/10 -> {r['numeric_binary_verdict']}   ({_short(r['numeric_reason'], 200)})\n"
        + "-" * 70
    )
if mismatches:
    print(f"\nWARNING: {mismatches} rows had mismatched questions between the two logs "
          f"(binary and numeric logs are out of sync for this CELL) — re-run both with a fresh clear.")
print(f"\nCombined log written: {OUT_LOG}  ({len(combined)} questions)")
