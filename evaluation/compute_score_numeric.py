"""
Standalone numeric grader for LaRA predictions.

Separate from compute_score_llm.py on purpose (per aruniyen3's guidance: don't
modify LaRA's own scoring code, extract Q&A from it and grade independently).
Reads the same prediction/<model>/*.jsonl files LaRA's eval_full.py / eval_rag.py
already produce, asks a judge model for a 0-10 score instead of True/False, and
also derives a binary verdict (score >= --binary_threshold) so results stay
comparable to LaRA's own reported accuracy.

Usage mirrors compute_score_llm.py:
  python compute_score_numeric.py --eval_model qwen2.5:7b --judge_model qwen2.5:14b
"""
import argparse
import concurrent.futures
import csv
import json
import os
import re
import threading
import time

import pandas as pd
from openai import OpenAI

from eval_utils import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--eval_model', default='qwen2.5-7b-instruct', type=str, help='which model generated the predictions to read')
parser.add_argument('--judge_model', default=None, type=str, help='model that does the grading; defaults to --eval_model')
parser.add_argument('--binary_threshold', default=7, type=int, help='numeric score (0-10) at/above which a sample also counts as binary-correct')
parser.add_argument('--chunker', default=None, type=str,
                     help='Chonkie chunking strategy the RAG predictions were generated with '
                          '(token/sentence/recursive/semantic/fast). Only applies to rag_preds_*; '
                          'when set, full-context predictions are skipped since chunking does not apply there.')
args = parser.parse_args()

eval_model = args.eval_model
judge_model = args.judge_model or eval_model
BINARY_THRESHOLD = args.binary_threshold
chunker = args.chunker

_safe_eval = eval_model.replace(':', '-').replace('/', '-')
_safe_judge = judge_model.replace(':', '-').replace('/', '-')
CELL = f'gen-{_safe_eval}_judge-{_safe_judge}' + (f'_chunker-{chunker}' if chunker else '')

DEBUG = os.environ.get("LARA_JUDGE_DEBUG") == "1"
MAX_WORKERS = int(os.environ.get("LARA_WORKERS", "4"))
DEBUG_LOG_PATH = f'./prediction/result/numeric_judge_debug_{CELL}.jsonl'
_debug_lock = threading.Lock()

grand_score_sum = 0.0
grand_scored = 0
grand_err = 0

def _short(text, limit=500):
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"

def log_judgement(tag, sample, score, reason, judge_raw):
    verdict = "ERROR (judge call failed)" if score is False else (
        f"{score}/10 -> {'True' if score >= BINARY_THRESHOLD else 'False'} (threshold {BINARY_THRESHOLD})"
    )
    block = (
        "\n" + "-" * 70 + "\n"
        + f"[{tag}]\n"
        + f"QUESTION:   {_short(sample.get('question', ''), 300)}\n"
        + f"EXPECTED:   {_short(sample.get('ground_truth', ''))}\n"
        + f"GOT:        {_short(sample.get('prediction', ''))}\n"
        + f"JUDGE SAID: {_short(judge_raw, 300)}\n"
        + f"REASON:     {_short(reason, 300)}\n"
        + f"VERDICT:    {verdict}\n"
        + "-" * 70
    )
    record = {
        "tag": tag,
        "question": sample.get("question", ""),
        "expected": sample.get("ground_truth", ""),
        "got": sample.get("prediction", ""),
        "judge_raw": judge_raw,
        "score": score,
        "reason": reason,
        "binary_verdict": None if score is False else (score >= BINARY_THRESHOLD),
    }
    with _debug_lock:
        print(block)
        os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def call_gpt(model, messages, retry_num=5, retry_interval=5):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )
    for _ in range(retry_num):
        try:
            completion = client.chat.completions.create(model=model, messages=messages)
            text = completion.choices[0].message.content
            if isinstance(text, str) and text:
                return text
        except Exception:
            time.sleep(retry_interval)
            print(f'retry after {retry_interval} seconds')
    return False

SCORE_RE = re.compile(r'SCORE:\s*(\d+)', re.IGNORECASE)
FALLBACK_INT_RE = re.compile(r'\b(10|[0-9])\b')

def parse_score(response):
    m = SCORE_RE.search(response)
    if not m:
        m = FALLBACK_INT_RE.search(response)
    if not m:
        return None
    val = int(m.group(1))
    return max(0, min(10, val))

RUBRIC = """Score the AI assistant's answer against the groundtruth answer on a 0-10 scale:
- 10: fully correct and complete, conveys the same information as the groundtruth (rephrasing is fine).
- 7-9: correct on the core point(s), missing only minor detail or slightly imprecise.
- 4-6: partially correct; gets some but not all key facts, or is vague/incomplete.
- 1-3: mostly incorrect but contains a fragment of relevant/correct information.
- 0: completely incorrect, irrelevant, or contradicts the groundtruth.

Judge factual correctness and completeness relative to the groundtruth, not wording or phrasing.
An answer that says the same thing in different words should score as high as one that matches verbatim."""

RUBRIC_HALLU = """Score the AI assistant's answer on a 0-10 scale for whether it correctly handles a
potentially unanswerable question:
- 10: matches the groundtruth's stance exactly, either correctly stating the text does not contain
  the answer, or giving a fully correct answer when it does.
- 7-9: correct stance, minor imprecision.
- 4-6: hedges or is partially correct/incomplete about whether the text supports an answer.
- 1-3: mostly wrong stance but shows some relevant awareness.
- 0: confidently fabricates an answer not supported by the text, or flatly contradicts the groundtruth's stance."""

def get_score_one_llm(pred, label, query, query_type):
    rubric = RUBRIC_HALLU if query_type == 'hallu' else RUBRIC
    prompt = f'''{rubric}

[Query] {query}

[Groundtruth Answer] {label}

[AI Assistant's Answer] {pred}

Respond in exactly this format:
SCORE: <integer 0-10>
REASON: <one sentence>'''
    msg = [
        {"role": "system", "content": "You are a careful grader scoring AI answers against a groundtruth on a 0-10 scale."},
        {"role": "user", "content": prompt},
    ]
    for _ in range(20):
        response = call_gpt(judge_model, msg)
        if not response:
            time.sleep(5)
            continue
        score = parse_score(response)
        if score is None:
            time.sleep(2)
            continue
        reason_m = re.search(r'REASON:\s*(.*)', response, re.IGNORECASE | re.DOTALL)
        reason = reason_m.group(1).strip() if reason_m else ""
        return score, reason, response
    return False, "", ""

def process_example(sample, query_type):
    score, reason, judge_raw = get_score_one_llm(
        pred=sample['prediction'], label=sample['ground_truth'],
        query=sample['question'], query_type=query_type,
    )
    return score, reason, judge_raw, sample

query_type_list = ['location', 'reasoning', 'comp', 'hallu']
context_type_list = ['book', 'financial', 'paper']
length_list = ['32k', '128k']

save_all_path = f'./prediction/result/numeric_{CELL}_all.jsonl'
save_order_path = f'./prediction/result/numeric_{CELL}_order.jsonl'

# chunking only applies to the RAG path; when --chunker is set, skip full-context entirely
rag_or_full_list = ['rag'] if chunker else ['rag', 'full']

for rag_or_full in rag_or_full_list:
    for context_length in length_list:
        for query_type in query_type_list:
            for context_type in context_type_list:
                chunker_tag = f'{chunker}_' if (chunker and rag_or_full == 'rag') else ''
                check = f'{rag_or_full}_{eval_model}_{chunker_tag}{context_length}_{context_type}_{query_type}'
                data_path = f'./prediction/{eval_model}/{rag_or_full}_preds_{eval_model}_{chunker_tag}{context_length}_{context_type}_{query_type}.jsonl'
                if not os.path.exists(data_path):
                    continue

                print("\n============================================================")
                if os.path.exists(save_all_path):
                    with open(save_all_path, 'r') as f:
                        check_str = f.read()
                    if check in check_str:
                        print(f"{check} is already evaluated.")
                        continue

                print(f"current data is {check}")

                score_sum = 0.0
                cnt_scored = 0
                score_location = {}
                cnt_location = {}
                binary_true = 0

                data = load_data(data_path)
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = [executor.submit(process_example, sample, query_type) for sample in data]
                    for future in futures:
                        score, reason, judge_raw, sample = future.result()
                        if DEBUG:
                            log_judgement(check, sample, score, reason, judge_raw)
                        if score is not False:
                            score_sum += score
                            cnt_scored += 1
                            if score >= BINARY_THRESHOLD:
                                binary_true += 1
                            if query_type in ['location', 'reasoning']:
                                co = sample['context_order']
                                score_location[co] = score_location.get(co, 0) + score
                                cnt_location[co] = cnt_location.get(co, 0) + 1

                total_examples = len(data)
                err_all = total_examples - cnt_scored
                mean_score_10 = (score_sum / cnt_scored) if cnt_scored else 0.0
                mean_score_01 = mean_score_10 / 10.0
                binary_acc = (binary_true / cnt_scored) if cnt_scored else 0.0

                grand_score_sum += score_sum
                grand_scored += cnt_scored
                grand_err += err_all

                with open(save_all_path, 'a') as f:
                    f.write(f'{check}: mean_score={mean_score_10:.3f}, mean_score_norm={mean_score_01:.4f}, binary_accuracy={binary_acc:.4f}, cnt:{cnt_scored}\n')
                if query_type in ['location', 'reasoning']:
                    with open(save_order_path, 'a') as f:
                        f.write(f'{check}:\n')
                        for loc in score_location:
                            f.write(f'\tlocation {loc}: mean_score: {score_location[loc]/cnt_location[loc]:.3f}, cnt: {cnt_location[loc]}\n')

                print(f"[{check}]  mean_score: {mean_score_10:.2f}/10   binary_accuracy: {binary_acc:.4f}   scored: {cnt_scored}/{total_examples}   errors: {err_all}")

grand_total = grand_scored + grand_err
print("\n" + "=" * 70)
print(f"CELL: generator={eval_model}  judge={judge_model}  binary_threshold={BINARY_THRESHOLD}/10")
print(f"TOTAL SCORED: {grand_total}")
print(f"  SCORED: {grand_scored}")
print(f"  ERRORS: {grand_err}")
if grand_scored:
    print(f"  MEAN SCORE: {grand_score_sum / grand_scored:.3f} / 10")
if DEBUG:
    print(f"Per-question debug log: {DEBUG_LOG_PATH}")
print("=" * 70 + "\n")

# ---- CSV + summary, same shape as compute_score_llm.py's output ----
input_file = save_all_path
output_csv = f'./prediction/result/numeric_{CELL}_all.csv'

if os.path.exists(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Task Name', 'MeanScoreNorm', 'BinaryAccuracy'])
        for line in lines:
            line = line.strip()
            if not line:
                continue
            task_name, rest = line.split(': ', 1)
            parts = dict(p.split('=') for p in rest.split(', ') if '=' in p and not p.startswith('cnt'))
            csvwriter.writerow([task_name, parts.get('mean_score_norm', ''), parts.get('binary_accuracy', '')])

    df = pd.read_csv(output_csv)
    sub_tasks = ["location", "reasoning", "comp", "hallu"]
    results = {'full32k': {}, 'rag32k': {}, 'full128k': {}, 'rag128k': {}}

    for sub_task in sub_tasks:
        for length in ['32k', '128k']:
            for rf in ['full', 'rag']:
                filtered_df = df[df['Task Name'].str.contains(f"{rf}_") & df['Task Name'].str.contains(sub_task)
                & df['Task Name'].str.contains(length)]
                mean_val = filtered_df['MeanScoreNorm'].mean()
                bin_val = filtered_df['BinaryAccuracy'].mean()
                results[rf+length][sub_task] = {
                    'numeric_score_pct': round(float(mean_val) * 100, 2) if pd.notna(mean_val) else 0.0,
                    'binary_accuracy_pct': round(float(bin_val) * 100, 2) if pd.notna(bin_val) else 0.0,
                }

    print(json.dumps(results, ensure_ascii=False, indent=4))
