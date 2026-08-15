# Running LaRA on the ASU server (2x A10, 125GB RAM)

Companion to `RUNNING_LOCALLY.md`, which documents the Mac (16GB, memory-survival)
setup. Keep that file as-is for history; this one is for the ASU box, where several
of the Mac's constraints no longer apply.

## 1. One-time setup

```bash
# Ollama serves a local, OpenAI-compatible API on http://localhost:11434/v1
ollama serve                  # leave running, or run as a service
ollama pull qwen2.5:7b        # ~4.7GB, generator / weak judge
ollama pull qwen2.5:14b       # ~9GB, strong judge (fits easily in 46GB VRAM total)

# Python deps live in ./venv (Python 3.12). BGE retrieval models auto-download on
# first RAG run, same as the Mac.
```

## 2. Environment variables

| Var | Mac value | Server value | What it does |
|-----|-----------|---------------|---------------|
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | same | points the OpenAI client at Ollama (switch to `http://localhost:8000/v1` if we move to vLLM) |
| `OPENAI_API_KEY` | `ollama` | same | any non-empty string; Ollama ignores it |
| `OLLAMA_NUM_CTX` | `32768` (Full) / `8192` (RAG) | match the context length being run, up to `131072` for 128k Full runs | raises Ollama's context window so long prompts aren't truncated |
| `LARA_WORKERS` | `1` | `4` or higher | thread count; the Mac capped this at 1 to avoid loading multiple rerankers on 16GB RAM, the server has headroom |
| `LARA_LIMIT` | `3` | same, optional | only run the first N examples, for a smoke test |
| `LARA_EMBED_MODEL` / `LARA_RERANK_MODEL` | defaults | same, optional | override the RAG retrieval models |
| `LARA_JUDGE_DEBUG` | unset | `1`, optional | print + log each judge decision to `evaluation/prediction/result/judge_debug_<CELL>.jsonl` |

`--eval_model`: use `qwen2.5:7b` (and `qwen2.5:14b` or larger) instead of `llama3.2`.
`compute_score_llm.py` also takes `--judge_model` (defaults to `--eval_model` if omitted)
to grade with a different, stronger model than the one that generated the answers.

## 3. Smoke test

```bash
cd evaluation

# --- Full (long-context), 32k ---
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
OLLAMA_NUM_CTX=32768 \
../venv/bin/python eval_full.py \
  --query_type reasoning --context_type book --context_length 32k --eval_model qwen2.5:7b

# --- Full (long-context), 128k — could not run this on the Mac ---
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
OLLAMA_NUM_CTX=131072 \
../venv/bin/python eval_full.py \
  --query_type reasoning --context_type book --context_length 128k --eval_model qwen2.5:7b

# --- RAG ---
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
OLLAMA_NUM_CTX=8192 LARA_WORKERS=4 \
../venv/bin/python eval_rag.py \
  --query_type reasoning --context_type book --context_length 32k --eval_model qwen2.5:7b

# --- Score, judged by the stronger 14b model ---
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
../venv/bin/python compute_score_llm.py --eval_model qwen2.5:7b --judge_model qwen2.5:14b
```

Drop `LARA_LIMIT` (or don't set it) to run the full example set. Add `LARA_JUDGE_DEBUG=1`
before the score command to get the per-question trace instead of just the aggregate
accuracy.

## 4. What's different from the Mac, and why it matters

- **128k now fits.** The 16GB Mac couldn't hold a 128k KV-cache; the server's 125GB RAM
  and 2x A10 (46GB VRAM total) can. Use `OLLAMA_NUM_CTX=131072` for those runs.
- **A real judge is affordable.** `qwen2.5:14b` (9GB) fits comfortably alongside the
  generator model. The Mac was stuck at 3B/7B judges, which the 2x2 study
  (`evaluation/STUDY_RESULTS.md`) showed materially understate accuracy and can even
  fail to rank generators correctly.
- **Concurrency headroom.** `LARA_WORKERS` can go well above 1 without reranker-thrashing
  OOMs (see `evaluation/STUDY_RESULTS.md` section 6 for what that failure looked like
  on 16GB).
- **`--judge_model` decouples generator and judge**, which is what the 2x2 study and any
  future strong-judge run rely on. Not documented in `RUNNING_LOCALLY.md` since it wasn't
  needed there.
