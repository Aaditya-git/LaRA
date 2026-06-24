# LaRA benchmark: local run progress, decisions, and open issues

Author: Aaditya Bhilegaonkar
Machine: MacBook Air, Apple M2, 16 GB RAM

## 1. Goal

Get the LaRA benchmark running so we can compare RAG against Long-Context (LC) LLMs.
The benchmark has two generation modes (Full = whole document in context, RAG =
retrieve top-k chunks) and an LLM-as-judge scoring step that grades each answer
against the ground truth.

## 2. Key decisions

1. **Run locally with open-source models (Ollama) instead of the OpenAI/Qwen cloud APIs.**
   Reason: the OpenAI API key gets exhausted very quickly when generating answers over
   32k-128k token contexts. Local models cost nothing per call.
2. **Use the API-style scripts (`eval_full.py`, `eval_rag.py`) pointed at Ollama**, not the
   `*_open.py` transformers scripts. Ollama is lighter on a Mac and reuses the cleaner code;
   the `*_open.py` route needs model weights on disk and is memory-heavy.
3. **Models tried:** `llama3.2:3b` (fast, weak) and `qwen2.5:7b` (slower, stronger).
4. **Scope locally = 32k only.** 128k full-context does not fit in 16 GB RAM (see issue 1 below).
5. **Keep the paper's published baseline predictions**; write our local runs to new folders.

## 3. What we did (steps)

1. Installed/used Ollama; pulled `llama3.2` and `qwen2.5:7b`.
2. Repointed the eval scripts to Ollama's OpenAI-compatible endpoint using env vars
   (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OLLAMA_NUM_CTX`).
3. Fixed several code issues that blocked any run (see section 5).
4. Swapped the missing GTE retrieval models for small BGE models that auto-download.
5. Ran the smoke test (32k / book / reasoning) for Full and RAG with both models, then scored.

How to run is documented separately in `RUNNING_LOCALLY.md`.

## 4. The main blocker for results: the local judge is too weak

This is the most important finding.

The generation step works well. Both Full and RAG produce coherent, on-topic answers,
and many of them clearly match the ground truth. Example (qwen2.5:7b, RAG):

- Question: "What lesson does Hartman learn from observing Brimstone's Christmas Eve party?"
- Ground truth: generosity and happiness come from simple gestures, not large sums of money;
  employers can make employees' lives joyful or miserable through their treatment.
- Our answer: "despite spending only a small amount of money, Brimstone brings great joy and
  gratitude to many people, which contrasts with his own miserly ways ... value the importance
  of generosity and kindness."

That answer is essentially correct. But the scoring step uses an LLM-as-judge, and the local
judge model (tested both `llama3.2:3b` and `qwen2.5:7b`) returns "False" on it, repeatedly and
deterministically. As a result, every score in the smoke test comes out as 0.0.

**Why this happens:** the LaRA paper judges with GPT-4o / Qwen-Max (very large models). Small
local models are not reliable graders for this nuanced True/False judgment; they default to
"False" when the wording differs from the ground truth, even when the meaning matches.

**Impact on results:** the accuracy numbers we get locally are not trustworthy. They understate
performance badly (0.0 across the board) and cannot be used to compare RAG vs LC. The pipeline
is correct end to end; only the grader is the bottleneck.

**Options to fix the judge:**
- (a) Use a strong judge via API for the scoring step only. Judge prompts are short (just the
  question, ground truth, and answer), so this is far cheaper than generating answers and would
  not exhaust a key as fast. This needs an API key.
- (b) Use a much larger local judge, which our 16 GB Mac cannot run.
- Generation can stay fully local either way; only grading needs the stronger model.

## 5. Code issues found and fixed

The upstream repo did not run out of the box. Fixes made (see `RUNNING_LOCALLY.md` for detail):
- Eval scripts hardcoded a cloud API key and could leave the response undefined for non-gpt/non-qwen
  model names. Now read endpoint/key from env and route any model to the OpenAI-compatible path.
- Ollama defaults to a small (~4k) context window, which silently truncates long prompts. We raise
  it via `OLLAMA_NUM_CTX` so the "full context" test is actually full.
- RAG referenced local GTE embedding/rerank models that are not in the repo. Swapped for small BGE
  models (`bge-small-en-v1.5`, `bge-reranker-base`) that auto-download and run on CPU.
- The RAG reranker (`FlagEmbeddingReranker`) is incompatible with the installed transformers version
  and hung in an infinite retry loop. Replaced with `SentenceTransformerRerank`.
- The scorer crashed on missing imports (`pandas`, `OpenAI`), did not handle partial runs, crashed on
  the final aggregation, and its result parsing breaks when the model name contains a colon
  (e.g. `qwen2.5:7b`). These are being cleaned up.
- Scripts did not create their own output directory, causing a crash after all answers were generated.

## 6. Hardware limitation: 128k does not fit on the Mac

The 16 GB M2 MacBook Air cannot run the Full-Context (whole document) test at 128k context. The
key-value cache for a 128k-token prompt alone exceeds available RAM. We can run 32k locally, but
128k needs more memory. This likely means we need access to ASU's machines for the 128k portion.

## 7. Status summary

- Local generation pipeline (Full and RAG, 32k): working for `llama3.2:3b` and `qwen2.5:7b`.
- Answers look good qualitatively.
- Scoring is blocked by weak local judges, so accuracy numbers are not yet meaningful.
- 128k requires more RAM than the Mac has.

## 8. What we need to move forward

1. A proprietary API key (OpenAI or Qwen) to use as a strong judge for scoring. This is the
   cheaper use of the key (short prompts), so it should last much longer than using it for
   generation.
2. Access to a larger machine (e.g. ASU compute) to run the 128k Full-Context evaluations.
