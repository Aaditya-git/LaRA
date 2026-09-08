# HANDOVER — LaRA fork (Aaditya + Sarvesh, ASU FURI, Prof. Jia Zou)

Single source of truth for this repo, replacing the older RESUME.md / PROGRESS_NOTES.md /
JUDGE_ISSUE_WALKTHROUGH.md / CHUNKER_INTEGRATION_SCOPE.md / RUNNING_LOCALLY.md /
RUNNING_ON_SERVER.md / evaluation/STUDY_RESULTS.md (their content is folded in below; those
files have been removed). `README.md` is kept as-is — it's the original upstream project
description (paper, citation, license).

Last updated: 2026-09-08.

---

## 1. What this repo is

[LaRA](https://arxiv.org/abs/2502.09977) is a benchmark that compares two ways of answering
questions over a long document with an LLM:

- **Full / Long-Context (LC)**: paste the whole document into the prompt.
- **RAG**: chunk the document, retrieve the top-k chunks relevant to the question, paste
  only those into the prompt.

It ships 2,326 question/answer pairs over 32k- and 128k-token documents (`book`, `paper`,
`financial`), across four task types (`location`, `reasoning`, `comp`, `hallu`), and grades
every generated answer with an LLM-as-judge against a ground-truth answer.

This fork got the upstream pipeline running locally with **Ollama** instead of paid APIs,
found and fixed several bugs that blocked it from running at all, and then used it to
investigate research questions from Prof. Jia Zou. Four pieces of work sit on top of the base
pipeline, in the order they happened — see §5 for what each one is and its current status:

1. Get the pipeline running locally at all (bug fixes, Ollama routing).
2. **Task A**: which LaRA questions genuinely stress RAG (need chunks from all over the doc)?
3. **Task B**: replace the binary True/False judge with a graded numeric score.
4. Chonkie chunker comparison (separate ask): does chunking *strategy* affect RAG accuracy?
5. Retrieval chunk-count (`top_k`) study (separate ask): does retrieving more chunks help?

## 2. Repo map

```
LaRA
├── README.md              upstream project description (paper, citation) — unchanged
├── HANDOVER.md             this file
├── requirements.txt        pip deps (paper's original list; a couple of extras — chonkie,
│                           pandas — get installed ad hoc, see §6)
├── query_gen_fin.py        upstream script that GENERATED the financial QA pairs in
│                           datasets/query/ via GPT. Not part of the eval pipeline; only
│                           relevant if you need to mint new financial questions.
├── select_stress_questions.py   curates the RAG-stressing question subset used by the
│                           top_k study (§5.4). Reads datasets/query/, writes
│                           datasets/query_topk_study/.
├── scripts/                 upstream's original driver scripts (run_eval.sh,
│                           compute_score.sh) — superseded by evaluation/run_*.sh below,
│                           kept for reference, not used in this fork's workflow.
├── datasets/
│   ├── 32k/, 128k/          the raw documents (book .txt, paper .md, financial .txt),
│   │                        one subfolder per context_type.
│   ├── query/                the 2,326 question files, `{length}_{context_type}_{task}.jsonl`.
│   │                        See §3 for the fields.
│   └── query_topk_study/     curated subset (15 Q per context_type × task) for the top_k
│                           study only, produced by select_stress_questions.py.
├── figs/                    images used by README.md.
└── evaluation/               everything that actually runs. See §4.
```

### `evaluation/` in detail

```
evaluation/
├── eval_full.py             Generate answers, Full/Long-Context mode (whole doc in prompt).
├── eval_rag.py               Generate answers, RAG mode (chunk → retrieve → prompt).
├── eval_full_open.py         Same as eval_full.py but loads HF weights via transformers
├── eval_rag_open.py          instead of calling an API/Ollama. NOT used in this fork
│                            (memory-heavy, weights not in repo) — see §6 note.
├── eval_utils.py             load_data / create_msgs / create_prompt / dump_jsonl — shared
│                            by every eval_*.py script.
├── prompt.py                 the actual prompt templates (full_templates / rag_templates),
│                            one per context_type. eval_utils.py fills these in.
├── chunkers.py                RAG chunking adapter: get_nodes(text, strategy) → llama-index
│                            TextNodes, using Chonkie's 5 chunkers (token/sentence/
│                            recursive/semantic/fast). Added in this fork (§5.5).
├── model_cache.py             process-wide singleton cache for the embedding model + reranker
│                            (added in this fork to fix an OOM — see §5.3 note).
├── search/
│   ├── baseSearcher.py        abstract BaseSearcher: .process(query) = retrieve then rerank.
│   ├── simpleHybridSearcher.py  SimpleHybridSearcher(config, nodes): builds a vector
│   │                          retriever (VectorStoreIndex + BGE embeddings, via
│   │                          model_cache) and a BM25 retriever, fuses them, reranks with
│   │                          SentenceTransformerRerank (BGE reranker). This is the actual
│   │                          RAG retrieval logic; `rerank_size` here is the RAG top_k.
│   └── simpleHybridRetriever.py  SimpleHybridRetriever: dedupes + concatenates the vector
│                              and BM25 result lists (the "hybrid" in the name).
├── query_engine/              thin wrapper re-exporting llama-index's RetrieverQueryEngine
│                            / BaseQueryEngine so search/ doesn't import llama_index directly.
├── compute_score_llm.py       LLM-as-judge, BINARY True/False scale (upstream's original
│                            scoring, patched — see §5.1 and §5.3).
├── compute_score_numeric.py   LLM-as-judge, 0–10 GRADED scale (added in this fork, §5.3
│                            "Task B"). Standalone from compute_score_llm.py on purpose —
│                            reads the same prediction files, writes separate output, so
│                            upstream's own binary scoring path is never touched.
├── merge_judgments.py         joins the per-question debug logs from the binary and numeric
│                            judges into one combined_judgments_*.jsonl (side-by-side view,
│                            used for the slide decks).
├── aggregate_study.py         builds prediction/result/STUDY_matrix.csv for the 2×2
│                            generator×judge study (§5.1).
├── aggregate_chunker_results.py  builds prediction/result/chunker_comparison.csv (§5.5).
├── aggregate_topk_results.py     builds prediction/result/topk_comparison.csv (§5.4).
├── run_pipeline.sh             general-purpose local pipeline driver: generate → judge
│                            (both scales) → merge, for a hand-edited list of configs.
│                            This is the one to reach for on a fresh box / new smoke test.
├── run_study.sh                 orchestrates the 2×2 generator×judge study end to end (§5.1).
├── run_chunker_study.sh          orchestrates the chunker comparison end to end (§5.5).
├── run_topk_study.sh              orchestrates the top_k study end to end (§5.4).
├── cleanup_study.sh              deletes a study's prediction/result outputs to rerun clean.
├── LaRA_judge_study_deck.html/.pdf     slide deck, 2×2 judge study (§5.1).
├── LaRA_chunker_study_deck.html/.pdf   slide deck, chunker comparison (§5.5).
├── LaRA_topk_study_deck.html            slide deck, top_k study (§5.4, HTML only).
├── prediction/<model>/          generated answers, one .jsonl per (mode, model, chunker,
│                            top_k, length, context_type, task) config.
└── prediction/result/           everything compute_score_*.py / aggregate_*.py write:
    ├── {CELL}_all.jsonl / _all.csv     per-config binary accuracy (CELL = gen/judge/
    │                                  chunker/top_k tag, see §4.2).
    ├── numeric_{CELL}_all.jsonl/.csv   per-config numeric mean score + derived binary.
    ├── {CELL}_order.jsonl               binary accuracy broken down by context_order
    │                                  (0/1/2/full) for location & reasoning tasks.
    ├── judge_debug_{CELL}.jsonl / numeric_judge_debug_{CELL}.jsonl   per-question judge
    │                                  traces (only written when LARA_JUDGE_DEBUG=1).
    ├── combined_judgments_{CELL}.jsonl   binary + numeric verdicts merged per question.
    └── {STUDY,chunker,topk}_comparison.csv / STUDY_matrix.csv   the final comparison
                                          tables each study script produces.
```

`prediction/` and `prediction/result/` for Ollama-generated runs are gitignored (see
`.gitignore`) — only the paper's original baseline predictions ship in git. Everything
described above under `prediction/` is regenerated locally when you run the scripts.

## 3. Dataset shape (`datasets/query/*.jsonl`)

One JSON object per line. Common fields: `type`, `level`, `length`, `file` (which document),
`question`, `answer` (ground truth). Task-specific field:

- **`location` / `reasoning` / `hallu`**: `context_order` — where the answer's evidence sits
  in the document. `0` = start third, `1` = middle third, `2` = end third, `"full"` = spread
  across the whole document. This is a **ground-truth label baked in at dataset-creation
  time** (based on character/token position in the original continuous document) — it is
  *not* produced by any chunker or retriever at eval time.
- **`comp`**: `comp_parts` — a pair of segment indices `[a, b]` the answer must combine
  (e.g. `[0, 2]` = start + end). This is the only task with a genuinely multi-segment
  requirement; see §5.2 for what we found out about it.

## 4. How the pipeline connects, end to end

### 4.1 The two stages

1. **Generate.** `eval_full.py` or `eval_rag.py` reads a `datasets/query/*.jsonl` file,
   builds a prompt per question (`eval_utils.create_msgs` + `prompt.py`), calls the model
   (`call_gpt`, which despite the name talks to any OpenAI-compatible endpoint — real OpenAI,
   DashScope/Qwen, or local Ollama, chosen via `OPENAI_BASE_URL`), and writes one
   `{full,rag}_preds_*.jsonl` file into `prediction/<model>/`.

   For RAG specifically: `eval_rag.py::process_example()` reads the raw document text,
   calls `chunkers.get_nodes(text, strategy)` to chunk it, hands the chunks to
   `SimpleHybridSearcher` (vector + BM25 hybrid retrieval, then rerank down to `top_k`
   chunks), and stuffs the retrieved chunks into the prompt as `eg["context"]`.

2. **Score.** `compute_score_llm.py` (binary) and/or `compute_score_numeric.py` (0–10) read
   the prediction file(s), send each (question, ground truth, prediction) triple to a judge
   model, and write aggregate accuracy to `prediction/result/`. `merge_judgments.py`
   optionally joins the two scales' per-question logs into one file. The `aggregate_*.py`
   scripts then pivot those per-config CSVs into one final comparison table per study.

### 4.2 The `CELL` naming convention

Every scoring script computes a tag: `gen-{eval_model}_judge-{judge_model}` plus
`_chunker-{chunker}` and/or `_topk-{top_k}` when those flags are set. All of that config's
output files (`{CELL}_all.csv`, `judge_debug_{CELL}.jsonl`, etc.) are named with this tag, so
different generator/judge/chunker/top_k combinations never overwrite each other and can be
diffed side by side. `:` and `/` in model names are sanitized to `-` (Ollama tags like
`qwen2.5:7b` would otherwise break filenames).

### 4.3 Filename convention for everything else

`{length}_{context_type}_{query_type}` throughout, e.g. `32k_book_reasoning`. RAG prediction
filenames also carry the chunker and top_k:
`rag_preds_{eval_model}_{chunker}_top{top_k}_{length}_{context_type}_{query_type}.jsonl`.

## 5. Status of each piece of work

### 5.1 Base pipeline + judge-strength question (done)

The upstream repo didn't run out of the box (hardcoded API key, undefined-variable bug for
non-gpt/qwen model names, missing imports, GTE models not shipped, an incompatible reranker
that hung forever, no auto-created output dirs). All fixed; see git history on `eval_full.py`,
`eval_rag.py`, `compute_score_llm.py`, `search/simpleHybridSearcher.py` for the specifics.

Central finding, confirmed by a 2×2 generator×judge study (`run_study.sh` →
`aggregate_study.py` → `prediction/result/STUDY_matrix.csv`, deck:
`LaRA_judge_study_deck.html`): **a small local judge (7B) is unreliable** — it fails
correct-but-reworded answers, and swapping judge 7b→14b moved accuracy by up to +17.5 points
on identical answers, more than swapping the generator did. On the 3 shared task types, the
weak judge scored the 7b and 14b generators as a tie; the strong judge correctly separated
them. **Practical rule going forward: generation can be local/cheap, but grading needs a
strong judge** (this box: `qwen2.5:14b`, or larger if VRAM allows).

### 5.2 Task A — which questions genuinely stress RAG (analysis done; no code deliverable yet)

The professor's ask: find questions that need "several chunks from all over the document."
Computed over all 2,326 questions:

| Task      | Count | Segments needed | Spread across doc? |
|-----------|------:|------------------|---------------------|
| location  | 763   | 1                | No |
| hallu     | 608   | 1                | No |
| reasoning | 606   | 1                | No |
| **comp**  | 349 (15%) | **exactly 2** | Sometimes |

Only `comp` needs 2 segments (`comp_parts`); nothing in LaRA needs 3+. Of the 349 comp
questions, only ~50 (≈2%) have `comp_parts` at relative spread ≥3 (using
`relative_spread = |a−b| / (segments−1)`, since raw spread isn't comparable across docs with
different segment counts — 32k docs have ~3 segments, 128k up to 10).

**Honest bottom line**: LaRA cannot really test "several chunks from all over the doc"; the
best it offers is comp questions ranked by relative spread. Generating genuinely multi-hop
questions would require going beyond LaRA's own dataset. This analysis fed directly into the
top_k study's question curation (§5.4) but was never turned into its own standalone
script/report — **that deliverable form is still an open decision** if the professor wants it
separately.

### 5.3 Task B — numeric judge scale (built)

Replaced/augmented the binary True/False judge with a 0–10 rubric-based judge:
`compute_score_numeric.py`, kept fully separate from `compute_score_llm.py` (binary scoring
is untouched; numeric reads the same prediction files and writes its own output). Rubric
scores factual correctness/completeness, not wording (see the `RUBRIC` / `RUBRIC_HALLU`
constants in that file). Also derives a binary verdict at `--binary_threshold` (default 7/10)
so numeric results stay comparable to the paper's reported binary accuracy.

Design decisions made: 0–10 integer scale (not 0–1 continuous or 5-point Likert); binary kept
*alongside* numeric, not replaced; strong judge required (same finding as §5.1); scale applies
to all 4 task types with a task-specific rubric variant for `hallu`.

Every study since (chunker, top_k) judges with **both** scales and reports the gap: numeric
consistently reads several points higher than binary — the same judge-collapse phenomenon as
§5.1, now shown to persist across chunking strategy and retrieval depth too.

Infra note: a process-wide model cache (`model_cache.py`) was added because the RAG eval used
to construct a fresh embedding model + reranker *per question* across the thread pool, which
OOM-killed a 16GB machine. It now loads each model once and shares it across threads/questions.

### 5.4 Retrieval chunk-count (top_k) study — done, 2026-08-19

Ask: benchmark only `comp` + `reasoning` (comp fixed at top_k=2, since §5.2 shows it never
needs more than 2 segments); sweep top_k for reasoning (5/10/15); use RAG-stressing questions,
not first-N file order; end with a deck.

Built: `select_stress_questions.py` (curates 15 Q per context_type×task — comp by highest
relative spread, reasoning prioritizing `context_order=="full"`), `--top_k` flag threaded
through `eval_rag.py` / `compute_score_llm.py` / `compute_score_numeric.py` /
`merge_judgments.py`, `run_topk_study.sh` (orchestration), `aggregate_topk_results.py`,
`LaRA_topk_study_deck.html`. Chunker held fixed at `token` (the §5.5 winner) so top_k is the
only new variable. Scope: 32k only.

Results (90 curated questions, gen `qwen2.5:7b`, judge `qwen2.5:14b`):

| top_k | task | binary | numeric |
|---|---|---|---|
| 5  | reasoning | 53.3% | 59.8% |
| 10 | reasoning | 53.3% | 61.6% |
| 15 | reasoning | 57.8% | 64.2% |
| 2 (fixed) | comp | 28.9% | 37.1% |

Comp's low score traces to retrieval misses (9/11 zero-scored comp answers explicitly say the
retrieved chunks lack the needed info — expected, since these questions were curated for
*widest* chunk distance and top_k=2 leaves no margin for a miss). Reasoning's near-zero tail
traces to arithmetic across two document sections, not retrieval depth (one such question
scored 0, 0, 1 at top_k 5/10/15 — more chunks didn't fix it). Full table:
`evaluation/prediction/result/topk_comparison.csv`.

**Not yet done**: deck not sent to the professor. Natural next step: widen N per cell, test
top_k × chunker interaction (untested so far).

### 5.5 Chonkie chunker comparison — done, 2026-08-05

Separate ask: swap LaRA's fixed `SentenceSplitter` chunking for each of Chonkie's 5 chunkers
(token/sentence/recursive/semantic/fast) and compare RAG accuracy, without pulling in
ChunkResearch's MongoDB/LangGraph/Langfuse scaffolding — only the chunking step crosses over.

Built: `chunkers.py` (single seam, `get_nodes(text, strategy)`), `--chunker` flag through
`eval_rag.py` / both scoring scripts / `merge_judgments.py`, `run_chunker_study.sh`,
`aggregate_chunker_results.py`, `LaRA_chunker_study_deck.html/.pdf`. chunk_size=600/
overlap=100 held constant across strategies (recursive/semantic/fast have no overlap knob in
Chonkie — a real strategy difference, not a bug). Kept LaRA's own LLM-as-judge (both scales);
explicitly did **not** switch to embedding-similarity scoring, to keep chunking strategy the
only variable under test.

Bounded first run (`LARA_LIMIT=5`, 32k only, 60 unique questions): `token` chunker ranked
first on both scales (63.3% binary / 68.3% numeric overall); `fast` showed the widest
binary-vs-numeric gap. Small-n, directional, not a tight ranking. Full table:
`evaluation/prediction/result/chunker_comparison.csv`.

**Not yet done**: an email summarizing this for the professor was drafted but never sent
(recipient addresses unconfirmed); deck + logs were copied to Aaditya's Mac for him to send.

## 6. How to run

### 6.1 One-time setup

```bash
# Python deps live in ./venv
cd LaRA
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install chonkie[semantic]   # not in requirements.txt; installed ad hoc,
                                            # run_chunker_study.sh auto-installs it if missing

# Ollama serves an OpenAI-compatible API on localhost:11434
ollama serve                 # leave running (or run as a service)
ollama pull qwen2.5:7b       # generator / weak judge, ~4.7GB
ollama pull qwen2.5:14b      # strong judge, ~9GB (needed for trustworthy scores, see §5.1)
```

Current machine (ASU server): 2× A10 (46GB VRAM total), 125GB RAM, ~1.4TB disk — 128k
full-context and a 14b+ judge both fit comfortably. The Mac this started on (16GB M2) could
not do either; if you're ever back on a small machine, cap context at 32k and judge with
whatever fits, but treat the resulting scores as a lower bound (§5.1).

### 6.2 Environment variables (used throughout)

| Var | Purpose | Typical value here |
|---|---|---|
| `OPENAI_BASE_URL` | point the OpenAI client at Ollama | `http://localhost:11434/v1` |
| `OPENAI_API_KEY` | any non-empty string; Ollama ignores it | `ollama` |
| `OLLAMA_NUM_CTX` | Ollama's context window; must cover the prompt or it silently truncates | `8192` for RAG, `32768`/`131072` for Full at 32k/128k |
| `LARA_LIMIT` | only run the first N examples per config (smoke test) | `5`–`15`, unset for a full run |
| `LARA_WORKERS` | thread pool size for generation/judging | `4`+ (server has headroom; the old Mac needed `1`) |
| `LARA_EMBED_MODEL` / `LARA_RERANK_MODEL` | override the RAG retrieval models | defaults: `BAAI/bge-small-en-v1.5` / `BAAI/bge-reranker-base` |
| `LARA_JUDGE_DEBUG` | `1` to print + log every judge decision to `judge_debug_*.jsonl` | set for anything you want a per-question trace of |
| `LARA_QUERY_DIR` | which query dir `eval_rag.py` reads from | `../datasets/query` (default) or `../datasets/query_topk_study` |
| `--judge_model` (CLI flag, not env) | grade with a different model than the generator | e.g. `--eval_model qwen2.5:7b --judge_model qwen2.5:14b` |

### 6.3 Quickest smoke test

```bash
cd evaluation
LARA_LIMIT=5 bash run_pipeline.sh
```

This is the general-purpose driver: edit the `CONFIGS` array at the top of
`run_pipeline.sh` to choose which (mode, length, context_type, query_type) combinations to
run, then it generates, verifies prediction counts, judges with both scales, and merges the
logs — all in one command, with a full transcript saved to `evaluation/logs/`.

### 6.4 Running a single config by hand

```bash
cd evaluation
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama OLLAMA_NUM_CTX=8192 \
  ../venv/bin/python eval_rag.py --chunker token --top_k 5 \
  --query_type reasoning --context_type book --context_length 32k --eval_model qwen2.5:7b

OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
  ../venv/bin/python compute_score_llm.py --eval_model qwen2.5:7b --judge_model qwen2.5:14b --chunker token --top_k 5
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
  ../venv/bin/python compute_score_numeric.py --eval_model qwen2.5:7b --judge_model qwen2.5:14b --chunker token --top_k 5
```

Drop `--chunker`/`--top_k` for a plain run (default chunker = `sentence`, default top_k = 5);
use `eval_full.py` instead of `eval_rag.py` (no `--chunker`/`--top_k` flags there) for
Long-Context mode.

### 6.5 Reproducing one of the three studies

Each study script is self-contained and skips work already done (re-run safely after a
partial failure):

```bash
cd evaluation
bash run_study.sh                 # 2x2 generator x judge study        -> STUDY_matrix.csv
bash run_chunker_study.sh         # chunking-strategy comparison        -> chunker_comparison.csv
python ../select_stress_questions.py && bash run_topk_study.sh   # top_k retrieval-depth study -> topk_comparison.csv
```

Override scope inline, e.g. `LARA_LIMIT=30 CHUNKERS="token semantic" bash run_chunker_study.sh`
or `REASONING_TOP_KS="5 10 20" bash run_topk_study.sh`. Each script prints the env vars it's
using and where its outputs land at the end.

### 6.6 Reading results

- Per-config accuracy: `prediction/result/{CELL}_all.csv` (binary) and
  `numeric_{CELL}_all.csv` (numeric mean + derived binary). See §4.2 for the `CELL` tag.
- Final comparison tables: `STUDY_matrix.csv` / `chunker_comparison.csv` /
  `topk_comparison.csv`.
- Per-question detail (why a specific answer scored what it did): run with
  `LARA_JUDGE_DEBUG=1`, then read `judge_debug_{CELL}.jsonl` / `numeric_judge_debug_{CELL}.jsonl`,
  or `combined_judgments_{CELL}.jsonl` after running `merge_judgments.py` for both side by side.
- Slide decks (self-contained HTML, open in a browser): `LaRA_judge_study_deck.html`,
  `LaRA_chunker_study_deck.html`, `LaRA_topk_study_deck.html`.

## 7. Open items / where to pick up

1. Task A (§5.2) has an analysis but no agreed deliverable form (script vs. written report vs.
   both) — confirm with Aaditya/the professor before building anything further.
2. Chunker study email (§5.5) and top_k deck (§5.4) were never sent to the professor.
3. Untested interaction: chunker × top_k together (each was swept independently).
4. All three studies ran on small `LARA_LIMIT` samples (5–15 per cell) — directional, not
   statistically tight. Widening N is the natural next step for any of them, pending
   feedback on which axis the professor cares about most.
5. 128k full-context runs are unblocked by this server but haven't been exercised by any of
   the three studies yet (all scoped to 32k so far).
