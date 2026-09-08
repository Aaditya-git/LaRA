from eval_utils import (
    load_data,
    dump_jsonl
)
import re
import string
from collections import Counter
import json
from tqdm import tqdm
import requests
import time
import os
import concurrent.futures
import csv
import dashscope
import pandas as pd
from openai import OpenAI
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--eval_model', default='qwen2.5-7b-instruct', type=str, help='which model generated the predictions to read')
parser.add_argument('--judge_model', default=None, type=str, help='model that does the grading; defaults to --eval_model')
parser.add_argument('--chunker', default=None, type=str,
                     help='Chonkie chunking strategy the RAG predictions were generated with '
                          '(token/sentence/recursive/semantic/fast). Only applies to rag_preds_*; '
                          'when set, full-context predictions are skipped since chunking does not apply there.')
parser.add_argument('--top_k', default=None, type=int,
                     help='number of chunks retrieved (rerank_size) the RAG predictions were generated with; '
                          'must match the --top_k passed to eval_rag.py for the same run.')

args = parser.parse_args()

eval_model = args.eval_model
judge_model = args.judge_model or eval_model
chunker = args.chunker
top_k = args.top_k

# distinct result-file tag per (generator, judge[, chunker][, top_k]) cell so runs don't overwrite each other
_safe_eval = eval_model.replace(':', '-').replace('/', '-')
_safe_judge = judge_model.replace(':', '-').replace('/', '-')
CELL = f'gen-{_safe_eval}_judge-{_safe_judge}' + (f'_chunker-{chunker}' if chunker else '') + (f'_topk-{top_k}' if top_k else '')

import threading

# --- Debug / visibility knobs -------------------------------------------------
# LARA_JUDGE_DEBUG=1 : print each judge decision (question / expected / got /
#                      judge's raw reply / verdict) and append it to a jsonl file.
# LARA_WORKERS=N     : judge calls in parallel (default 4). Use 1 for clean,
#                      non-interleaved debug output.
DEBUG = os.environ.get("LARA_JUDGE_DEBUG") == "1"
MAX_WORKERS = int(os.environ.get("LARA_WORKERS", "4"))
DEBUG_LOG_PATH = f'./prediction/result/judge_debug_{CELL}.jsonl'
_debug_lock = threading.Lock()

# running totals across every task, printed as a grand total at the end
grand_true = 0
grand_false = 0
grand_err = 0

def _short(text, limit=500):
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"

def log_judgement(tag, sample, score, judge_raw):
    if score is False:
        verdict = "ERROR (judge call failed)"
    elif score == 1.0:
        verdict = "1.0 (True)"
    else:
        verdict = "0.0 (False)"
    block = (
        "\n" + "-" * 70 + "\n"
        + f"[{tag}]\n"
        + f"QUESTION:   {_short(sample.get('question', ''), 300)}\n"
        + f"EXPECTED:   {_short(sample.get('ground_truth', ''))}\n"
        + f"GOT:        {_short(sample.get('prediction', ''))}\n"
        + f"JUDGE SAID: {_short(judge_raw, 300)}\n"
        + f"VERDICT:    {verdict}\n"
        + "-" * 70
    )
    record = {
        "tag": tag,
        "question": sample.get("question", ""),
        "expected": sample.get("ground_truth", ""),
        "got": sample.get("prediction", ""),
        "judge_raw": judge_raw,
        "verdict": verdict,
    }
    with _debug_lock:
        print(block)
        os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

api_key = ""
org_id = ""

def call_qwen(
    model,
    messages,
    retry_num: int = 10,
    retry_interval: int = 1,
    ):
    dashscope.api_key = api_key
    for _ in range(retry_num):
        response = dashscope.Generation.call(
            model=model,
            messages=messages,
            enable_search=False,
        )
        print(response)
        try:
            text = response.output.text
            if isinstance(text, str) and text:
                return text
        except:
            time.sleep(retry_interval)
    return False

def call_gpt(model, messages, retry_num=5, retry_interval=5):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", api_key),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        organization=org_id or None,
    )
    for _ in range(retry_num):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            text = completion.choices[0].message.content
            if isinstance(text, str) and text:
                return text
        except:
            time.sleep(retry_interval)
            print(f'retry after {retry_interval} seconds')
    return False

def get_score_one_llm(pred, label, query, query_type, model='qwen-max-allinone') -> float:
    if query_type == 'hallu':
        prompt = f'''You need to help me determine whether an AI model is hallucinating. I will provide you with a question, a ground truth answer, and a prediction from the model. If the prediction from the model and the ground truth answer are basically consistent, and it is determined that this question is not mentioned in the text, then the model is deemed not to be hallucinating and the answer is considered correct. If it is correct, you should only output True; if it is incorrect, only output False.

[Query] {query}

[Groundtruth Answer] {label}

[AI Assistant's Answer] {pred}

Now, start your judgement:'''
    else:
        prompt = f'''I will provide you with a question and its groundtruth answer, as well as an answer from an AI assistant. You need to judge whether the AI assistant's answer is correct based on the groundtruth answer. If it is correct, you should only output True; if it is incorrect, only output False.

[Query] {query}

[Groundtruth Answer] {label}

[AI Assistant's Answer] {pred}

Now, start your judgment:'''
    msg = [
        {
            "role": "system",
            "content": "You are a discriminator that judges whether the predictions to questions are correct.",
        },
        {"role": "user", "content": prompt},
    ]
    for _ in range(20):
        try:
            # response = call_qwen(model=model, messages=msg)
            response = call_gpt(judge_model, msg)
            if not response:
                return False, ""
            else:
                if 'true' in response.lower():
                    return 1.0, response
                else:
                    return 0.0, response
        except:
            time.sleep(5)
    return False, ""

def process_example(sample, query_type):
    pred = sample['prediction']
    query = sample['question']
    label = sample['ground_truth']
    score, judge_raw = get_score_one_llm(pred=pred, label=label, query=query, query_type=query_type)
    return score, judge_raw, sample     

query_type_list = ['location', 'reasoning', 'comp', 'hallu']
context_type_list = ['book', 'financial', 'paper']
length_list = ['32k', '128k']

save_all_path = f'./prediction/result/{CELL}_all.jsonl'
save_order_path = f'./prediction/result/{CELL}_order.jsonl'

# chunking/top_k only apply to the RAG path; when either is set, skip full-context entirely
rag_or_full_list = ['rag'] if (chunker or top_k) else ['rag', 'full']

for rag_or_full in rag_or_full_list:
    for context_length in length_list:
        for query_type in query_type_list:
            for context_type in context_type_list:
                chunker_tag = f'{chunker}_' if (chunker and rag_or_full == 'rag') else ''
                topk_tag = f'top{top_k}_' if (top_k and rag_or_full == 'rag') else ''
                check = f'{rag_or_full}_{eval_model}_{chunker_tag}{topk_tag}{context_length}_{context_type}_{query_type}'
                data_path = f'./prediction/{eval_model}/{rag_or_full}_preds_{eval_model}_{chunker_tag}{topk_tag}{context_length}_{context_type}_{query_type}.jsonl'
                if not os.path.exists(data_path):
                    continue  # no predictions generated for this config; skip silently
                print("\n============================================================")
                if os.path.exists(save_all_path):
                    with open(save_all_path, 'r') as f:
                        check_str = f.read()
                    if check in check_str:
                        print(f"{check} is already evaluated.")
                        continue

                print(f"current data is {check}")

                score_all = 0.0
                cnt_all = 0
                score_location = {}
                cnt_location = {}

                data = load_data(data_path)
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = [executor.submit(process_example, sample, query_type) for sample in data]
                    for future in futures:
                        score, judge_raw, sample = future.result()
                        if DEBUG:
                            log_judgement(check, sample, score, judge_raw)
                        if score is not False:
                            score_all += score
                            cnt_all += 1
                            if query_type in ['location', 'reasoning']:
                                if sample['context_order'] in score_location:
                                    score_location[sample['context_order']] += score
                                    cnt_location[sample['context_order']] += 1
                                else:
                                    score_location[sample['context_order']] = score
                                    cnt_location[sample['context_order']] = 1                         
                total_examples = len(data)
                err_all = total_examples - cnt_all
                n_true = int(round(score_all))
                n_false = cnt_all - n_true
                score_all_avg = (score_all / cnt_all) if cnt_all else 0.0
                grand_true += n_true
                grand_false += n_false
                grand_err += err_all
                with open(save_all_path, 'a') as f:
                    f.write(f'{check}: {score_all_avg}, cnt:{cnt_all}\n')
                if query_type in ['location', 'reasoning']:
                    with open(save_order_path, 'a') as f:
                        for loc in score_location:
                            acc = score_location[loc] / cnt_location[loc]
                            f.write(f'{check}:\n')
                            f.write(f'\tlocation {loc}: accuracy: {acc}, cnt: {cnt_location[loc]}\n')
                print(f"[{check}]  TRUE: {n_true}/{cnt_all}   FALSE: {n_false}/{cnt_all}   ERRORS: {err_all}   ->  accuracy: {score_all_avg:.4f}")

grand_total = grand_true + grand_false + grand_err
print("\n" + "=" * 70)
print(f"CELL: generator={eval_model}  judge={judge_model}")
print(f"TOTAL JUDGED: {grand_total}")
print(f"  TRUE:   {grand_true}")
print(f"  FALSE:  {grand_false}")
print(f"  ERRORS: {grand_err}")
if DEBUG:
    print(f"Per-question debug log: {DEBUG_LOG_PATH}")
print("=" * 70 + "\n")

# Save evaluation results into csv format

input_file = save_all_path
output_csv = f'./prediction/result/{CELL}_all.csv'

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Task Name', 'Accuracy'])

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # split on the first ': ' only, so a colon inside the model name
        # (e.g. qwen2.5:7b) does not break parsing
        task_name, rest = line.split(': ', 1)
        accuracy = float(rest.split(',')[0].strip())
        csvwriter.writerow([task_name, accuracy])

df = pd.read_csv(output_csv)
sub_tasks = ["location", "reasoning", "comp", "hallu"]
results = {'full32k': {}, 'rag32k': {}, 'full128k': {}, 'rag128k': {}}

for sub_task in sub_tasks:
    for length in ['32k', '128k']:
        for rf in ['full', 'rag']:
            filtered_df = df[df['Task Name'].str.contains(f"{rf}_") & df['Task Name'].str.contains(sub_task)
            & df['Task Name'].str.contains(length)]
            mean_val = filtered_df['Accuracy'].mean()
            # mean() of an empty selection is NaN (a plain float); treat missing configs as 0.0
            avg_score = round(float(mean_val) * 100, 2) if pd.notna(mean_val) else 0.0
            results[rf+length][sub_task] = avg_score

results['rag32k']['overall'] = round((results['rag32k']['location'] + results['rag32k']['reasoning'] + results['rag32k']['comp'] + results['rag32k']['hallu']) / 4, 2)
results['rag128k']['overall'] = round((results['rag128k']['location'] + results['rag128k']['reasoning'] + results['rag128k']['comp'] + results['rag128k']['hallu']) / 4, 2)
results['full32k']['overall'] = round((results['full32k']['location'] + results['full32k']['reasoning'] + results['full32k']['comp'] + results['full32k']['hallu']) / 4, 2)
results['full128k']['overall'] = round((results['full128k']['location'] + results['full128k']['reasoning'] + results['full128k']['comp'] + results['full128k']['hallu']) / 4, 2)


print(json.dumps(results, ensure_ascii=False, indent=4))
