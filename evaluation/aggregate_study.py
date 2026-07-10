#!/usr/bin/env python3
"""Aggregate the 2x2 study into a comparison matrix + judge-agreement analysis.

Reads the per-cell result files written by compute_score_llm.py and produces:
  - a 2x2 accuracy matrix (per task type + overall)
  - judge agreement: for the SAME answers, how often the weak (7b) and strong (14b)
    judge agree, and which way they disagree (too strict vs too lenient)
  - prediction/result/STUDY_matrix.csv  (raw material for the PPT)
"""
import json
import os
import csv

RESULT_DIR = "prediction/result"
WEAK = "qwen2.5:7b"
STRONG = "qwen2.5:14b"
QTYPES = ["location", "reasoning", "comp", "hallu"]

CELLS = [
    ("weak gen  / weak judge",   WEAK,   WEAK),
    ("weak gen  / strong judge", WEAK,   STRONG),
    ("strong gen / weak judge",  STRONG, WEAK),
    ("strong gen / strong judge", STRONG, STRONG),
]


def safe(m):
    return m.replace(":", "-").replace("/", "-")


def cell_tag(gen, judge):
    return f"gen-{safe(gen)}_judge-{safe(judge)}"


def read_scores(gen, judge):
    """-> {qtype: (accuracy_0_1, n)}"""
    path = f"{RESULT_DIR}/{cell_tag(gen, judge)}_all.jsonl"
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        tag, rest = line.split(": ", 1)
        acc = float(rest.split(",")[0])
        n = int(rest.split("cnt:")[1]) if "cnt:" in rest else 0
        qtype = tag.split("_")[-1]
        out[qtype] = (acc, n)
    return out


def read_debug(gen, judge):
    """-> {question: 1|0|None} verdict per question"""
    path = f"{RESULT_DIR}/judge_debug_{cell_tag(gen, judge)}.jsonl"
    rows = {}
    if not os.path.exists(path):
        return rows
    for line in open(path):
        r = json.loads(line)
        v = r["verdict"]
        rows[r["question"]] = 1 if v.startswith("1.0") else (0 if v.startswith("0.0") else None)
    return rows


# ---- accuracy matrix ----
print("\n=== 2x2 ACCURACY MATRIX  (RAG, book, 32k, %) ===")
print("cell".ljust(26) + "".join(q[:9].ljust(11) for q in QTYPES) + "overall")
matrix = {}
for name, gen, judge in CELLS:
    sc = read_scores(gen, judge)
    row = name.ljust(26)
    accs = []
    for q in QTYPES:
        if q in sc:
            a = sc[q][0] * 100
            accs.append(a)
            row += f"{a:5.1f}".ljust(11)
        else:
            row += "  -".ljust(11)
    overall = sum(accs) / len(accs) if accs else 0.0
    row += f"{overall:5.1f}"
    matrix[name] = (sc, overall)
    print(row)

# ---- judge agreement (the trustworthiness signal) ----
print("\n=== JUDGE AGREEMENT  (same answers: weak 7b judge vs strong 14b judge) ===")
for label, gen in [("weak generator (7b)", WEAK), ("strong generator (14b)", STRONG)]:
    w = read_debug(gen, WEAK)
    s = read_debug(gen, STRONG)
    common = [q for q in w if q in s and w[q] is not None and s[q] is not None]
    if not common:
        print(f"  {label}: no overlapping judged questions yet")
        continue
    n = len(common)
    agree = sum(1 for q in common if w[q] == s[q])
    too_strict = sum(1 for q in common if w[q] == 0 and s[q] == 1)   # weak=False, strong=True
    too_lenient = sum(1 for q in common if w[q] == 1 and s[q] == 0)  # weak=True, strong=False
    print(f"  {label}: {n} Qs | agree {agree}/{n} ({100*agree/n:.0f}%) | "
          f"weak too STRICT (F vs T): {too_strict} | weak too LENIENT (T vs F): {too_lenient}")

# ---- CSV for the deck ----
os.makedirs(RESULT_DIR, exist_ok=True)
csv_path = f"{RESULT_DIR}/STUDY_matrix.csv"
with open(csv_path, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["cell", "generator", "judge"] + QTYPES + ["overall"])
    for name, gen, judge in CELLS:
        sc, ov = matrix[name]
        wr.writerow([name, gen, judge]
                    + [f"{sc[q][0]*100:.1f}" if q in sc else "" for q in QTYPES]
                    + [f"{ov:.1f}"])
print(f"\nMatrix CSV written: {csv_path}")
