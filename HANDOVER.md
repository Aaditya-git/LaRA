# HANDOVER — LaRA fork

Everything you need to understand this repo and run it, written for someone who has never
seen it before. Author context: this is a fork of the [LaRA benchmark](https://arxiv.org/abs/2502.09977)
maintained by Aaditya Bhilegaonkar + Sarvesh Jere (ASU FURI) for Prof. Jia Zou.
Last updated: 2026-09-08.

## Table of contents

0. [30-second summary](#0-30-second-summary)
1. [Concepts you need before any of this makes sense](#1-concepts-you-need-before-any-of-this-makes-sense)
2. [Quick start — run something in 5 minutes](#2-quick-start--run-something-in-5-minutes)
3. [The pipeline: how one question turns into one score](#3-the-pipeline-how-one-question-turns-into-one-score)
4. [Repo map — every file, what it does](#4-repo-map--every-file-what-it-does)
5. [How to run every scenario](#5-how-to-run-every-scenario)
6. [How to read the output](#6-how-to-read-the-output)
7. [What research has been done in this fork, and its status](#7-what-research-has-been-done-in-this-fork-and-its-status)
8. [Open items / what to do next](#8-open-items--what-to-do-next)

---

## 0. 30-second summary

LaRA answers one question: **when an LLM has to answer a question about a long document,
is it better to (a) paste the whole document into the prompt, or (b) retrieve just the
relevant pieces and paste those?** This repo runs that comparison and grades the answers.

This fork got that comparison running for free using **Ollama** (a local LLM server) instead
of paid APIs, fixed a pile of bugs that stopped it running at all, and then used it to answer
three follow-up research questions for Prof. Zou. Section 7 covers what those were and where
they landed; everything before that is "how the machine works."

---

## 1. Concepts you need before any of this makes sense

| Term | Meaning here |
|---|---|
| **Long-Context (LC) / "Full"** | Paste the *entire* document into the prompt and ask the question. Simple, but expensive and limited by how much text the model can hold at once. Code: `eval_full.py`. |
| **RAG (Retrieval-Augmented Generation)** | Split the document into small pieces ("chunks"), find the chunks most relevant to the question, and paste *only those* into the prompt. Code: `eval_rag.py`. |
| **Chunking** | The step that splits a document into chunks before retrieval. Different strategies (split by token count, by sentence, by semantic similarity, etc.) chunk differently, and that can change RAG's accuracy. Code: `chunkers.py`. |
| **Retrieval / rerank** | RAG doesn't hand every chunk to the model — it searches for the most relevant ones (a mix of keyword search and embedding-similarity search), then reranks and keeps only the top `k` ("top_k"). Code: `search/`. |
| **Generator / `eval_model`** | The LLM that reads the prompt (whole doc or retrieved chunks) and writes an answer. |
| **Judge** | A *second* LLM call that grades the generator's answer against the known-correct answer. This is not a simple string match — the judge reads both answers and decides if they mean the same thing. This turns out to be the trickiest part of the whole system (see §7.1). |
| **Task types** | Every question belongs to one of four types: `location` (find a specific fact), `reasoning` (compute/derive something), `comp` (compare two things, needs 2 document sections), `hallu` (the answer isn't in the document at all — tests whether the model admits that instead of making something up). |
| **Context types** | The kind of document: `book` (fiction), `paper` (academic paper), `financial` (a company's financial filing). |
| **Context length** | `32k` or `128k` — roughly how many tokens the source document is. |

**One question, concretely** (from `datasets/query/32k_book_reasoning.jsonl`):
```json
{"type": "novelette_fake", "level": "reasoning", "length": "32k", "file": "A Christmas Carol.txt",
 "context_order": 1,
 "question": "What lesson does Hartman learn from observing Brimstone's Christmas Eve party?",
 "answer": "Hartman realizes that despite spending only a small amount of money, ... generosity and kindness."}
```
`context_order` says WHERE in the document the answer lives (`0`=start third, `1`=middle third,
`2`=end third, `"full"`=spread across the whole thing) — this is a ground-truth label baked in
when the dataset was built, not something any chunker computes. `comp` questions instead carry
`comp_parts`, e.g. `[0, 2]`, meaning the answer needs BOTH the start-third and end-third of the
document combined.

---

## 2. Quick start — run something in 5 minutes

```bash
# 1. One-time setup (skip if venv/ already has these + Ollama is already pulled)
cd ~/LaRA
./venv/bin/pip install -r requirements.txt
ollama serve &                    # starts the local model server (if not already running)
ollama pull qwen2.5:7b            # ~4.7GB — this will act as both answer-writer and judge

# 2. Run the smoke test: 5 questions per config, generate -> judge -> merge, all in one command
cd evaluation
LARA_LIMIT=5 bash run_pipeline.sh
```

What happens: it generates answers for 3 small configs (edit the `CONFIGS` array inside
`run_pipeline.sh` to change which ones), judges each answer on BOTH a binary True/False scale
and a 0–10 graded scale, and prints a summary. You'll see output like:

```
[full_qwen2.5:7b_32k_book_reasoning]  TRUE: 3/5   FALSE: 2/5   ERRORS: 0   ->  accuracy: 0.6000
```

Full run transcript is saved to `evaluation/logs/run_<timestamp>_<model>.log`. Per-config
results land in `evaluation/prediction/result/`. See §6 for what every output file means.

That's the whole loop. Everything else in this doc is either "how to run it differently"
(§5) or "what we found when we did" (§7).

---

## 3. The pipeline: how one question turns into one score

```
 datasets/query/32k_book_reasoning.jsonl   (2,326 questions total, across all files)
            │
            │  eval_full.py  OR  eval_rag.py     (§4.2)
            ▼
 ┌──────────────────────────────────────────────────────────┐
 │  RAG path only (eval_rag.py):                             │
 │  raw document text                                        │
 │      → chunkers.get_nodes(text, strategy)     chunk it    │
 │      → search/simpleHybridSearcher.py         retrieve    │
 │        (vector search + BM25 keyword search,              │
 │         fused, then reranked down to top_k chunks)         │
 │      → eg["context"] = the retrieved chunks                │
 └──────────────────────────────────────────────────────────┘
            │
            │  prompt.py builds the final prompt (context + question)
            │  call_gpt() sends it to the generator model (Ollama/OpenAI-compatible)
            ▼
 evaluation/prediction/<model>/{full,rag}_preds_....jsonl
   one line per question: {question, ground_truth, prediction, ...}
            │
            │  compute_score_llm.py   (binary True/False judge)
            │  compute_score_numeric.py  (0-10 graded judge)      ← both read the SAME file
            ▼
 evaluation/prediction/result/{CELL}_all.csv        per-config accuracy
 evaluation/prediction/result/numeric_{CELL}_all.csv per-config numeric score
            │
            │  merge_judgments.py   (optional: join both scales per-question)
            │  aggregate_*.py       (pivot many configs into one comparison table)
            ▼
 evaluation/prediction/result/*_comparison.csv   +   slide decks (*.html)
```

`{CELL}` is a filename tag built from the generator, judge, chunker, and top_k used, so
different runs never overwrite each other. See §6.1 for exactly how it's built.

---

## 4. Repo map — every file, what it does

### 4.1 Root

| File | What it is |
|---|---|
| `README.md` | The **original upstream** project description (paper link, citation, license). Not touched by this fork — for what LaRA *is*, not how this fork runs it. |
| `HANDOVER.md` | This file. |
| `requirements.txt` | Python dependencies for `./venv`. (`chonkie` and `pandas` are extra packages this fork added — see §5.1's setup command; they're installed ad hoc, not yet added to this file.) |
| `query_gen_fin.py` | Upstream's script that originally *generated* the financial-document questions using GPT. You will not need this unless you want to mint brand-new financial QA pairs. Not part of the eval pipeline. |
| `select_stress_questions.py` | Written in this fork. Picks the 15 hardest/most RAG-stressing questions per (document type, task type) — used only by the top_k study (§7.4). Reads `datasets/query/`, writes `datasets/query_topk_study/`. |
| `scripts/run_eval.sh`, `scripts/compute_score.sh` | Upstream's original driver scripts. Superseded by `evaluation/run_*.sh` (§4.3) — kept for reference, not part of this fork's workflow. |

### 4.2 `datasets/`

| Path | What's in it |
|---|---|
| `datasets/32k/<book\|paper\|financial>/` | The raw source documents at ~32k tokens each. |
| `datasets/128k/<book\|paper\|financial>/` | The raw source documents at ~128k tokens each. |
| `datasets/query/{length}_{context_type}_{task}.jsonl` | The 2,326 question files. One JSON object per line — see §1 for the field meanings. This is the **original, untouched dataset**. |
| `datasets/query_topk_study/` | A curated *subset* (15 questions per context_type × task) built by `select_stress_questions.py`, used only for the top_k study (§7.4). The original `datasets/query/` is never modified. |

### 4.3 `evaluation/` — this is where everything actually runs

**Generation (turn a question into a predicted answer):**

| File | Purpose | Reads | Writes |
|---|---|---|---|
| `eval_full.py` | Long-Context mode: paste the whole document into the prompt. | `datasets/query/*.jsonl`, `datasets/{32k,128k}/...` | `prediction/<model>/full_preds_*.jsonl` |
| `eval_rag.py` | RAG mode: chunk, retrieve, prompt with only the retrieved chunks. | same, plus calls `chunkers.py` + `search/` | `prediction/<model>/rag_preds_*.jsonl` |
| `eval_full_open.py` / `eval_rag_open.py` | Same idea but load HuggingFace model weights directly via `transformers` instead of calling an API. **Not used in this fork** — heavier on memory, and the weights aren't in the repo. Ollama through `eval_full.py`/`eval_rag.py` is the path actually used here. |
| `eval_utils.py` | Shared helpers: `load_data` (read a `.jsonl`), `create_msgs`/`create_prompt` (fill in `prompt.py`'s templates), `dump_jsonl` (write predictions out). Used by every `eval_*.py`. |
| `prompt.py` | The literal prompt text templates — one per context_type, one set for Full mode and one for RAG mode. |
| `chunkers.py` | RAG chunking adapter, added in this fork. `get_nodes(text, strategy)` runs one of [Chonkie](https://github.com/chonkie-inc/chonkie)'s chunkers (`token`/`sentence`/`recursive`/`semantic`/`fast`) and wraps the output as objects the retriever understands. |
| `model_cache.py` | Added in this fork to fix an out-of-memory bug: caches the embedding model and reranker once per process instead of reloading them for every single question. |

**Retrieval (used only by `eval_rag.py`), in `evaluation/search/`:**

| File | Purpose |
|---|---|
| `baseSearcher.py` | Abstract base class: defines `.process(query)` = retrieve, then rerank. |
| `simpleHybridSearcher.py` | The actual retrieval logic. Builds a vector-similarity retriever (embeddings via `model_cache.py`) and a BM25 keyword retriever, and a reranker that narrows everything down to the final `top_k` chunks that go in the prompt. |
| `simpleHybridRetriever.py` | Combines the vector-search results and BM25 results into one deduplicated list (the "hybrid" part). |
| `evaluation/query_engine/` | Thin wrapper so `search/` doesn't need to import `llama_index` directly everywhere. |

**Scoring (turn a predicted answer into a score):**

| File | Purpose | Reads | Writes |
|---|---|---|---|
| `compute_score_llm.py` | The **binary** judge (True/False), upstream's original scoring method, patched to work locally. | `prediction/<model>/*.jsonl` | `prediction/result/{CELL}_all.csv` etc. |
| `compute_score_numeric.py` | The **0–10 graded** judge, added in this fork. Deliberately a *separate* script from the binary one, so upstream's own scoring path is never touched — it just reads the same prediction files. | same | `prediction/result/numeric_{CELL}_all.csv` etc. |
| `merge_judgments.py` | Joins the binary and numeric judge's per-question debug logs into one file, so you can see both verdicts for the same question side by side. Only useful if you ran with `LARA_JUDGE_DEBUG=1`. | `judge_debug_*.jsonl` + `numeric_judge_debug_*.jsonl` | `combined_judgments_*.jsonl` |

**Aggregation (turn many per-config CSVs into one comparison table):**

| File | Produces |
|---|---|
| `aggregate_study.py` | `prediction/result/STUDY_matrix.csv` — the 2×2 generator×judge study (§7.1). |
| `aggregate_chunker_results.py` | `prediction/result/chunker_comparison.csv` — chunking-strategy comparison (§7.5). |
| `aggregate_topk_results.py` | `prediction/result/topk_comparison.csv` — retrieval-depth comparison (§7.4). |

**Orchestration (run everything with one command):**

| File | What it runs end-to-end |
|---|---|
| `run_pipeline.sh` | General-purpose driver: generate → verify → judge (both scales) → merge, for a hand-edited list of configs. **Start here for any new/manual run** — see §2 and §5.2. |
| `run_study.sh` | The full 2×2 generator×judge study (§7.1). |
| `run_chunker_study.sh` | The full chunking-strategy comparison (§7.5). |
| `run_topk_study.sh` | The full retrieval-depth (top_k) study (§7.4). |
| `cleanup_study.sh` | Deletes a study's output files so you can rerun it from scratch. |

**Everything else in `evaluation/`:**

| Path | What it is |
|---|---|
| `LaRA_judge_study_deck.html`/`.pdf`, `LaRA_chunker_study_deck.html`/`.pdf`, `LaRA_topk_study_deck.html` | Self-contained slide decks presenting each study's results (open the `.html` directly in a browser). |
| `prediction/<model>/` | Generated answers. One `.jsonl` per config. Predictions for the paper's original baseline models (`gpt-4o`, `claude-3-5-sonnet`, etc.) ship in git; anything generated locally via Ollama is gitignored. |
| `prediction/result/` | Every scoring/aggregation output — gitignored (regenerated locally). See §6 for the naming scheme. |

---

## 5. How to run every scenario

### 5.1 One-time setup

```bash
cd ~/LaRA
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install chonkie[semantic]     # extra dep this fork added, not yet in requirements.txt

ollama serve                    # leave running (or run as a systemd service)
ollama pull qwen2.5:7b          # generator, and a decent "weak" judge — ~4.7GB
ollama pull qwen2.5:14b         # a much better judge — ~9GB (see §7.1 for why this matters)
```

This machine (ASU server): 2× A10 GPUs (46GB VRAM total), 125GB RAM, ~1.4TB disk — both
`qwen2.5:14b` and 128k-token full-context prompts fit comfortably. If you're ever on a
smaller machine, cap context length at 32k and expect small local judges to under-report
accuracy (§7.1) — treat those numbers as a lower bound, not ground truth.

### 5.2 Environment variables reference

| Variable | What it controls | Typical value |
|---|---|---|
| `OPENAI_BASE_URL` | Where the OpenAI-compatible client sends requests | `http://localhost:11434/v1` (Ollama) |
| `OPENAI_API_KEY` | Any non-empty string — Ollama ignores it | `ollama` |
| `OLLAMA_NUM_CTX` | Ollama's context window. **Must be ≥ your prompt size or it silently truncates.** | `8192` for RAG, `32768`/`131072` for Full mode at 32k/128k |
| `LARA_LIMIT` | Only process the first N questions per config — use for smoke tests | `5`–`15`; unset = all questions |
| `LARA_WORKERS` | Thread pool size for parallel generation/judging | `4`+ (this server has headroom) |
| `LARA_JUDGE_DEBUG` | `1` = also write a per-question trace of every judge decision | set whenever you want to inspect *why* something scored what it did |
| `LARA_QUERY_DIR` | Which question directory `eval_rag.py` reads from | `../datasets/query` (default), or `../datasets/query_topk_study` for the top_k study |
| `LARA_EMBED_MODEL` / `LARA_RERANK_MODEL` | Override the RAG retrieval models | defaults: `BAAI/bge-small-en-v1.5` / `BAAI/bge-reranker-base` |

CLI flags (not env vars): `--eval_model` (which model generates/is being scored),
`--judge_model` (grade with a *different*, usually stronger, model than the generator —
defaults to `--eval_model` if omitted), `--chunker` (RAG only), `--top_k` (RAG only).

### 5.3 Run one config by hand

```bash
cd evaluation

# Generate — RAG mode, token chunker, retrieve top 5 chunks
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama OLLAMA_NUM_CTX=8192 \
  ../venv/bin/python eval_rag.py \
  --chunker token --top_k 5 \
  --query_type reasoning --context_type book --context_length 32k \
  --eval_model qwen2.5:7b

# Score it — binary scale
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
  ../venv/bin/python compute_score_llm.py \
  --eval_model qwen2.5:7b --judge_model qwen2.5:14b --chunker token --top_k 5

# Score it — numeric scale
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
  ../venv/bin/python compute_score_numeric.py \
  --eval_model qwen2.5:7b --judge_model qwen2.5:14b --chunker token --top_k 5
```

Swap `eval_rag.py` for `eval_full.py` (drop `--chunker`/`--top_k`, they don't apply) to run
Long-Context mode instead. `--query_type` is one of `location`/`reasoning`/`comp`/`hallu`;
`--context_type` is one of `book`/`paper`/`financial`; `--context_length` is `32k`/`128k`.

### 5.4 `run_pipeline.sh` — the "do everything for one run" script

This is the script to reach for by default, and the one to demo if someone asks "how do I
run this project?" It's the **generate → judge → merge** loop from §3, driven by a short
settings block you edit at the top of the file.

**Settings block** (top of `run_pipeline.sh`), all overridable from the command line:
```bash
MODEL="qwen2.5:7b"   # the Ollama model — used as BOTH the answer-writer and the judge
LIMIT=5              # questions per config (5 = smoke test; "" = every question, slow)
DEBUG=1              # 1 = log every judge decision to a file you can inspect afterward
WORKERS=1            # how many questions to judge in parallel

CONFIGS=(
  "full 32k paper reasoning"
  "full 32k book reasoning"
  "rag  32k book reasoning"
)
```
Each `CONFIGS` line is `mode length context_type query_type`:
`mode` = `full` (whole document in the prompt) or `rag` (retrieve chunks first) · `length` =
`32k`/`128k` · `context_type` = `book`/`paper`/`financial` · `query_type` =
`location`/`reasoning`/`comp`/`hallu`. This list is *what gets tested* — edit it, save, rerun.

**What the script does, in order, and which file it calls for each step:**

| Step | What happens | File it calls |
|---|---|---|
| 0. Checks | Confirms the venv exists, Ollama answers on `:11434`, and `MODEL` is pulled. Fails fast with a clear message instead of limping partway through. | — |
| 1. Clear old outputs | Deletes any previous prediction/result files for this exact `MODEL`, so the run starts clean. | — |
| 2. Generate | For each `CONFIGS` line, calls `eval_full.py` or `eval_rag.py` (per its `mode`) with the matching `--context_length/--context_type/--query_type/--eval_model`. This is where the LLM actually writes answers. | `eval_full.py` / `eval_rag.py` |
| 3. Verify | Prints a line-count for every prediction file produced; aborts if generation produced nothing, before wasting time grading it. | — |
| 4. Judge (binary) | Sends each (question, correct answer, model's answer) to the judge, asks for plain True/False. | `compute_score_llm.py` |
| 5. Judge (numeric) | Same idea, but asks for a 0–10 score against a rubric instead of True/False — the graded scale from Task B (§7.3), since binary was too harsh on partially-right answers. | `compute_score_numeric.py` |
| 6. Merge | Stitches the binary and numeric judge's per-question notes into one file, so both verdicts for the same question sit side by side. | `merge_judgments.py` |

```
run_pipeline.sh
   ├── eval_full.py / eval_rag.py      → writes prediction/<model>/*.jsonl
   ├── compute_score_llm.py            → binary True/False grading
   ├── compute_score_numeric.py        → 0-10 graded scoring
   └── merge_judgments.py              → combines both judges' notes per question
```

Two separate scoring scripts is intentional, not redundant: `compute_score_llm.py` is the
benchmark's original scoring method, kept untouched for comparability with the published
paper; `compute_score_numeric.py` was added in this fork because a hard True/False was
marking correct-but-differently-worded answers as wrong (§7.1, §7.3).

**How to run it:**
```bash
cd ~/LaRA/evaluation
ollama serve &                  # if not already running
ollama pull qwen2.5:7b          # if not already pulled
LARA_LIMIT=5 bash run_pipeline.sh
```

**What you'll see, and where it lands:**
- Live in the terminal: a banner per step, then per-config lines like
  `[full_qwen2.5:7b_32k_book_reasoning]  TRUE: 3/5  FALSE: 2/5  ERRORS: 0  -> accuracy: 0.6000`
- Saved transcript: `evaluation/logs/run_<timestamp>_<model>.log`
- Results: `evaluation/prediction/result/gen-<model>_judge-<model>_all.csv` (binary) and
  `numeric_gen-<model>_judge-<model>_all.csv` (0–10 scale)
- Per-question detail (open this if someone asks "why did it get that one wrong?"):
  `evaluation/prediction/result/combined_judgments_gen-<model>_judge-<model>.jsonl`

`MODEL` plays both generator and judge here — for a genuinely independent judge (a stronger
model grading a weaker one, which §7.1 shows matters a lot), use the manual commands in §5.3
with separate `--eval_model`/`--judge_model`, or `run_study.sh` in §5.5.

### 5.5 Run one of the three full studies

Each script is idempotent — it skips any config whose output already exists, so a failed
run can just be re-launched.

```bash
cd evaluation

bash run_study.sh                                          # 2x2 generator x judge study
bash run_chunker_study.sh                                  # chunking-strategy comparison
python ../select_stress_questions.py && bash run_topk_study.sh   # top_k retrieval-depth study
```

Override scope with env vars, e.g.:
```bash
LARA_LIMIT=30 CHUNKERS="token semantic" bash run_chunker_study.sh
REASONING_TOP_KS="5 10 20" bash run_topk_study.sh
```

---

## 6. How to read the output

### 6.1 The `{CELL}` filename tag

Every scoring script computes a tag from its settings and stamps every file it writes with
it, so different runs never collide:

```
CELL = gen-{generator}_judge-{judge}[_chunker-{chunker}][_topk-{top_k}]
```
e.g. `gen-qwen2.5-7b_judge-qwen2.5-14b_chunker-token_topk-5` (`:` and `/` in model names get
turned into `-` so they're filesystem-safe).

### 6.2 Where things land

| File | Contents |
|---|---|
| `prediction/result/{CELL}_all.csv` | One row per config: `Task Name, Accuracy` (binary, 0.0–1.0). |
| `prediction/result/numeric_{CELL}_all.csv` | One row per config: `Task Name, MeanScoreNorm, BinaryAccuracy` (numeric, normalized 0–1, plus a derived binary verdict at the `--binary_threshold`, default 7/10). |
| `prediction/result/{CELL}_order.jsonl` | Binary accuracy broken down by `context_order` (0/1/2/full) — only for `location`/`reasoning`. |
| `prediction/result/judge_debug_{CELL}.jsonl` / `numeric_judge_debug_{CELL}.jsonl` | One record per question: the question, expected answer, got answer, the judge's raw reply, and the verdict. Only written when you ran with `LARA_JUDGE_DEBUG=1`. **This is the file to open when a score looks surprising.** |
| `prediction/result/combined_judgments_{CELL}.jsonl` | Binary + numeric verdicts for the same question, side by side. Written by `merge_judgments.py`. |
| `prediction/result/STUDY_matrix.csv` / `chunker_comparison.csv` / `topk_comparison.csv` | The final, presentable comparison table for each of the three studies (§7). |

---

## 7. What research has been done in this fork, and its status

### 7.1 Getting it running + the judge-strength finding (done)

The upstream repo didn't run at all out of the box: hardcoded API key, a code path that left
a variable undefined for any model name that wasn't `gpt`/`qwen`, missing Python imports,
retrieval models referenced that aren't shipped in the repo, and a reranker library
incompatible with current `transformers` that hung forever. All fixed (see git history on
`eval_full.py`, `eval_rag.py`, `compute_score_llm.py`, `search/simpleHybridSearcher.py`).

Central finding, from a 2×2 generator×judge study (`run_study.sh`, deck:
`LaRA_judge_study_deck.html`): **a small local judge model (7B) is an unreliable grader.**
It marks correct-but-differently-worded answers as False. Swapping the judge from 7B to 14B,
on the exact same set of answers, moved measured accuracy by up to +17.5 points — more than
swapping the *generator* moved it. Worse, the weak judge scored two different-quality
generators as a tie, while the strong judge correctly told them apart. **Conclusion: answer
generation can run on a small local model, but grading needs a strong judge** — this box uses
`qwen2.5:14b`, bump to something larger if VRAM allows.

### 7.2 Task A — "which questions really stress RAG?" (analysis done, no code deliverable yet)

The professor's ask: find the questions that genuinely require pulling information from
several places scattered across the document, not just one nearby chunk. Computed over all
2,326 questions:

| Task | Count | Segments the answer needs |
|---|---:|---|
| location | 763 | 1 |
| hallu | 608 | 1 |
| reasoning | 606 | 1 |
| **comp** | 349 (15%) | **exactly 2** |

Only `comp` needs more than one document segment (via its `comp_parts` field), and nothing in
LaRA ever needs 3+. Of those 349, only ~50 (≈2% of the whole benchmark) have their two needed
segments far apart (`relative_spread = |a−b| / (segments−1) ≥ 3`, a metric that's comparable
across documents of different lengths, unlike raw index distance).

**Bottom line to give the professor**: LaRA cannot really test "several chunks from all over
the document" — the closest it offers is `comp` questions ranked by relative spread, and that
analysis directly fed the question-curation in the top_k study (§7.4). Genuinely multi-hop
questions would need a dataset beyond LaRA. **Still undecided**: whether to package this as a
standalone script, a written report, or both — confirm with Aaditya before building it.

### 7.3 Task B — numeric judge scale (built)

`compute_score_numeric.py` grades each answer 0–10 against a rubric (factual correctness +
completeness, explicitly told to ignore wording differences), instead of a hard True/False.
Kept as a fully separate script from `compute_score_llm.py` so upstream's binary scoring path
is untouched. Also derives a binary verdict at `--binary_threshold` (default: score ≥ 7) so
results stay comparable to the paper's reported binary accuracy.

Every study since (chunker study, top_k study) judges with both scales, and the gap between
them is consistent: the numeric scale reads several points higher than binary in every case —
the same judge-collapse behavior found in §7.1, now shown to hold regardless of chunking
strategy or retrieval depth.

### 7.4 Retrieval chunk-count (`top_k`) study (done, 2026-08-19)

Ask: benchmark only `comp` and `reasoning`; keep `comp` at top_k=2 (per §7.2, it never needs
more); sweep top_k for `reasoning` across 5/10/15; use genuinely RAG-stressing questions
instead of the first N in the file; produce a slide deck.

Results (90 curated questions, generator `qwen2.5:7b`, judge `qwen2.5:14b`):

| top_k | task | binary accuracy | numeric accuracy |
|---|---|---|---|
| 5 | reasoning | 53.3% | 59.8% |
| 10 | reasoning | 53.3% | 61.6% |
| 15 | reasoning | 57.8% | 64.2% |
| 2 (fixed) | comp | 28.9% | 37.1% |

Comp's low score is a **retrieval** problem: 9 of 11 zero-scored comp answers explicitly say
the retrieved chunks don't contain the needed info — expected, since these questions were
deliberately picked for maximum chunk-to-chunk distance and top_k=2 leaves zero margin for a
miss. Reasoning's near-zero tail is a **generation/arithmetic** problem, not retrieval — one
financial "compute the change between two report dates" question scored 0, 0, 1 at top_k
5/10/15; tripling the retrieved context did not fix it. Full table:
`evaluation/prediction/result/topk_comparison.csv`. **Not yet done**: the deck hasn't been
sent to the professor.

### 7.5 Chonkie chunking-strategy comparison (done, 2026-08-05)

Ask: does the *strategy* used to chunk a document (not just how many chunks you retrieve)
change RAG accuracy? Swapped in each of [Chonkie](https://github.com/chonkie-inc/chonkie)'s 5
chunkers (`token`/`sentence`/`recursive`/`semantic`/`fast`) via the `chunkers.py` seam, kept
everything downstream (embedding, hybrid retrieval, rerank, generation, judging) identical.

Bounded first run (5 questions × 3 doc types × 4 tasks × 5 chunkers = 60 questions): `token`
ranked first on both scales (63.3% binary / 68.3% numeric overall); `fast` showed the widest
binary-vs-numeric gap. Directional, not statistically tight — full table:
`evaluation/prediction/result/chunker_comparison.csv`. **Not yet done**: a summary email for
the professor was drafted but never sent (recipient addresses unconfirmed).

---

## 8. Open items / what to do next

1. **Task A deliverable** (§7.2) — decide script vs. written report vs. both, with Aaditya/the
   professor, before building further.
2. **Send the pending write-ups** — chunker study email (§7.5) and top_k deck (§7.4) were
   never sent to the professor.
3. **Untested interaction**: chunker strategy × top_k depth together — each was swept
   independently so far, never both at once.
4. **Widen sample sizes** — all three studies ran on small `LARA_LIMIT` samples (5–15 per
   cell); results are directional, not statistically tight. Confirm which axis the professor
   wants deepened before spending compute on it.
5. **128k full-context** is unblocked by this server's RAM but hasn't been exercised by any of
   the three studies yet — everything so far is scoped to 32k.
