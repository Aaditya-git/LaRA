"""
Curate the reasoning/comp question subset for the top_k retrieval-count study
(see RESUME.md section on "this week's task"). Not a random/first-N slice --
picks the questions that actually stress RAG retrieval, reusing the Task A
analysis (relative spread for comp; "full"-document reasoning questions):

- comp: LaRA's comp_parts field gives the 2 segments a question needs. Rank by
  relative spread = |a-b| / (segments_in_doc - 1) and keep the highest-spread
  questions per doc type -- these are the ones where the 2 needed chunks are
  farthest apart in the document, the closest thing to "several chunks from
  all over" that 32k docs offer (see RESUME.md Task A).
- reasoning: most reasoning questions have a single-segment context_order
  (0/1/2...), but a subset is tagged context_order == "full", meaning the
  question needs the whole document, not one chunk -- the real RAG-stress
  case. Keep those first; only top up with diverse single-segment questions
  where a doc type has no "full" questions at all (paper, at 32k).

Output: datasets/query_topk_study/32k_<context_type>_<query_type>.jsonl,
one curated file per (context_type, query_type), read by eval_rag.py via
LARA_QUERY_DIR=../datasets/query_topk_study.
"""
import json
import os
import collections

SRC_DIR = 'datasets/query'
OUT_DIR = 'datasets/query_topk_study'
CONTEXT_TYPES = ['book', 'financial', 'paper']
N_PER_CELL = 15

os.makedirs(OUT_DIR, exist_ok=True)


def load(ctype, qtype):
    with open(f'{SRC_DIR}/32k_{ctype}_{qtype}.jsonl') as f:
        return [json.loads(l) for l in f]


def segs_per_file(ctype):
    """segment count per file = 1 + max index referenced by any question about it"""
    counts = collections.defaultdict(int)
    for qt in ['location', 'reasoning', 'hallu']:
        for row in load(ctype, qt):
            co = row['context_order']
            if isinstance(co, int):
                counts[row['file']] = max(counts[row['file']], co + 1)
    for row in load(ctype, 'comp'):
        counts[row['file']] = max(counts[row['file']], max(row['comp_parts']) + 1)
    return counts


def select_reasoning(ctype, n):
    rows = load(ctype, 'reasoning')
    full_rows = [r for r in rows if isinstance(r['context_order'], str)]
    if len(full_rows) >= n:
        return full_rows[:n], len(full_rows)
    rest = [r for r in rows if isinstance(r['context_order'], int)]
    by_order = collections.defaultdict(list)
    for r in rest:
        by_order[r['context_order']].append(r)
    picked = list(full_rows)
    orders = sorted(by_order)
    i = 0
    while len(picked) < n and any(by_order[o] for o in orders):
        o = orders[i % len(orders)]
        if by_order[o]:
            picked.append(by_order[o].pop(0))
        i += 1
    return picked[:n], len(full_rows)


def select_comp(ctype, n):
    rows = load(ctype, 'comp')
    segs = segs_per_file(ctype)

    def rel_spread(r):
        a, b = r['comp_parts']
        s = segs[r['file']]
        return abs(a - b) / (s - 1) if s > 1 else 0.0

    rows.sort(key=rel_spread, reverse=True)
    return rows[:n], [round(rel_spread(r), 2) for r in rows[:n]]


print(f"{'ctype':<10} {'reasoning (full/total picked)':<32} {'comp (spread range of picked)'}")
for ctype in CONTEXT_TYPES:
    r_rows, n_full = select_reasoning(ctype, N_PER_CELL)
    with open(f'{OUT_DIR}/32k_{ctype}_reasoning.jsonl', 'w') as f:
        for row in r_rows:
            f.write(json.dumps(row) + '\n')

    c_rows, spreads = select_comp(ctype, N_PER_CELL)
    with open(f'{OUT_DIR}/32k_{ctype}_comp.jsonl', 'w') as f:
        for row in c_rows:
            f.write(json.dumps(row) + '\n')

    print(f"{ctype:<10} {f'{n_full}/{len(r_rows)} full':<32} {min(spreads):.2f}-{max(spreads):.2f}")

print(f"\nWrote curated question sets to {OUT_DIR}/")
