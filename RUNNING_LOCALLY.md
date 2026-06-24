# Running LaRA locally (Mac, no paid API key)

This document explains how to run the LaRA benchmark fully locally using Ollama,
why each change was made, and the exact commands to reproduce a run.

The original repo assumes you call OpenAI / Qwen cloud APIs and that you have local
GTE embedding/rerank models on disk. Neither is true on a plain Mac, so we route the
LLM calls to a local Ollama server and swap the retrieval models for small BGE ones
that auto-download from HuggingFace.

## 0. The mental model

A run has two stages:

1. **Generate predictions** with one of the eval scripts. Each writes one `.jsonl`
   of answers into `evaluation/prediction/<model>/`.
   - `eval_full.py`  = Long-Context: feed the WHOLE document to the model.
   - `eval_rag.py`   = RAG: chunk the document, retrieve the top-k chunks, feed only those.
   - (`eval_full_open.py` / `eval_rag_open.py` run HF weights via `transformers` directly.
      We do NOT use these on Mac; Ollama is easier and lighter. See note at the bottom.)
2. **Score** the predictions with `compute_score_llm.py`, an LLM-as-judge that marks
   each answer True/False against the ground truth and writes accuracy to
   `evaluation/prediction/result/`.

Filename convention everywhere: `{length}_{context_type}_{task}` e.g. `32k_book_reasoning`.
Tasks: `location`, `reasoning`, `comp`, `hallu`. Context types: `book`, `paper`, `financial`.

## 1. One-time setup

```bash
# Ollama serves a local, OpenAI-compatible API on http://localhost:11434/v1
brew install ollama          # if not already installed
ollama serve                 # leave running in a terminal (or it runs as a service)
ollama pull llama3.2         # ~2GB, the 3B model we answer/judge with

# Python deps already live in ./venv (Python 3.14). Nothing else to install:
# the BGE retrieval models download automatically on first RAG run.
```

## 2. Environment variables we use

| Var | Meaning |
|-----|---------|
| `OPENAI_BASE_URL=http://localhost:11434/v1` | point the OpenAI client at Ollama |
| `OPENAI_API_KEY=ollama` | any non-empty string; Ollama ignores it |
| `OLLAMA_NUM_CTX=32768` | raise Ollama's context window (default ~4k would truncate long prompts) |
| `LARA_LIMIT=3` | (optional) only run the first N examples, for a quick smoke test |
| `LARA_WORKERS=1` | (optional, RAG only) threads; 1 avoids loading multiple rerankers on 16GB RAM |

## 3. Smoke test (proven working)

Run from the `evaluation/` directory. We use 32k only; 128k will not fit in 16GB RAM.

```bash
cd evaluation

# --- Full (long-context) ---
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
OLLAMA_NUM_CTX=32768 LARA_LIMIT=3 \
../venv/bin/python eval_full.py \
  --query_type reasoning --context_type book --context_length 32k --eval_model llama3.2

# --- RAG ---
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
OLLAMA_NUM_CTX=8192 LARA_LIMIT=3 LARA_WORKERS=1 \
../venv/bin/python eval_rag.py \
  --query_type reasoning --context_type book --context_length 32k --eval_model llama3.2

# --- Score both ---
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
../venv/bin/python compute_score_llm.py --eval_model llama3.2
```

Outputs:
- `evaluation/prediction/llama3.2/full_preds_*.jsonl` and `rag_preds_*.jsonl`
- `evaluation/prediction/result/llama3.2_all.jsonl` and `llama3.2_all.csv` (accuracy)

## 4. Scaling up

Drop `LARA_LIMIT` to run all examples of a config. Loop over tasks/context types
(still 32k only) by changing `--query_type` and `--context_type`. The scripts skip a
config whose output file already exists, so you can re-run safely.

## 5. Known limitation: scores are unreliable with a 3B model

In the smoke test, both Full and RAG scored 0.0. That is mostly because llama3.2:3b is
a weak **judge**: it returns "False" even for reasonable answers. The 3B model is also a
weak **answerer**. The pipeline is correct; the numbers are not meaningful research
results. To get usable numbers, point the judge (and optionally the answerer) at a
stronger model: pull e.g. `qwen2.5:7b` and pass `--eval_model qwen2.5:7b`, or restore the
real OpenAI/Qwen API path by setting a real key and removing `OPENAI_BASE_URL`.

## 6. What was changed from upstream and why

- `evaluation/eval_full.py`, `evaluation/eval_rag.py`
  - `call_gpt` now reads `OPENAI_API_KEY` / `OPENAI_BASE_URL` from the environment, so the
    same code talks to OpenAI or to local Ollama.
  - Inject `num_ctx` (via `OLLAMA_NUM_CTX`) so long prompts are not silently truncated.
  - Dispatch: any non-`qwen` model name now routes to the OpenAI-compatible path
    (previously a name without `gpt`/`qwen` left `response` undefined).
  - `LARA_LIMIT` to subset examples for quick runs.
- `evaluation/eval_rag.py`
  - The retrieval embedding/rerank models defaulted to local GTE paths that do not exist
    in this repo. Replaced with `BAAI/bge-small-en-v1.5` and `BAAI/bge-reranker-base`
    (auto-download, CPU-friendly), overridable via `LARA_EMBED_MODEL` / `LARA_RERANK_MODEL`.
  - `LARA_WORKERS` to control thread count.
- `evaluation/search/simpleHybridSearcher.py`
  - Replaced `FlagEmbeddingReranker` with `SentenceTransformerRerank`. FlagEmbedding 1.4.0
    is incompatible with the installed transformers (`XLMRobertaTokenizer has no attribute
    prepare_for_model`) and hangs in an infinite retry loop. SentenceTransformerRerank loads
    the same BGE reranker via sentence-transformers.
- `evaluation/compute_score_llm.py`
  - Added missing `import pandas as pd` and `from openai import OpenAI` (both would crash).
  - `call_gpt` reads env (same as above) so the judge runs on Ollama.
  - Skip any prediction file that does not exist (so partial runs score cleanly).
  - Fixed the final aggregation crash on empty groups (`mean().item()` on a NaN float).

## Note on the `*_open.py` scripts

`eval_full_open.py` / `eval_rag_open.py` load HuggingFace weights from `../models/` via
`transformers` with `device_map="auto"`. On a 16GB M2 this is slow and memory-fragile,
and the weights are not in the repo. Using Ollama through the API scripts above is the
recommended local path. If you ever want the `_open` route, download the weights into
`../models/` first and expect long runtimes.
