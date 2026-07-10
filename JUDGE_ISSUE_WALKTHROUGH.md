# LaRA local run: the 0.0 problem, explained end-to-end

Meeting notes / walkthrough. Read top to bottom; every code reference points to an
exact file and line you can open in VS Code.

**One-line summary:** Our generation pipeline works and produces good answers. Every
accuracy number comes out `0.0` because the *scoring* step uses a weak local
LLM-as-judge that says "False" on almost everything. The fix is to strengthen the
judge, not to rebuild the pipeline.

---

## 1. What LaRA is actually measuring

LaRA compares two ways of answering a question about a long document:

- **Full context (LC):** put the *whole document* in the model's context window and ask.
- **RAG:** *retrieve* the most relevant chunks and put only those in context, then ask.

The research question is: for long documents, when is RAG as good as (or better than)
just feeding the whole thing? To answer that we need a trustworthy accuracy number for
each mode. That number is where we're stuck.

**The data (in `datasets/`):**
- 2 lengths: `32k`, `128k` tokens
- 3 document types: `book`, `paper`, `financial`
- 4 question types: `location`, `reasoning`, `comp` (comparison), `hallu` (hallucination)
- Each question ships with a human `answer` = the **ground truth**.

---

## 2. The pipeline has three stages

```
  STAGE 1                STAGE 2                        STAGE 3
  Dataset        ->      Generation            ->       Scoring
  (question +            (LLM writes an                 (LLM judge grades
   ground truth)          answer = "prediction")         prediction vs truth)

  datasets/              eval_full.py / eval_rag.py     compute_score_llm.py
                         -> prediction/*.jsonl          -> result/*.csv
       WORKS                    WORKS                        BROKEN  <-- problem is here
```

### Stage 2 — Generation (this works)

Two scripts, one per mode. Both load the document, build a prompt, call an LLM, and
write out records containing `question`, `prediction`, and `ground_truth`.

**Full context — `evaluation/eval_full.py`**
- `process_example()` (`eval_full.py:81`) loads the entire document as the context
  (`eval_full.py:86`), builds the prompt with `create_msgs`, and calls the model.

**RAG — `evaluation/eval_rag.py`**
- `process_example()` (`eval_rag.py:89`) does the retrieval work:
  - Chunk the document with `SentenceSplitter` (`eval_rag.py:97`) — chunk size 600,
    overlap 100.
  - Embed chunks with a small BGE model (`eval_rag.py:104`).
  - Hybrid search + rerank, keep top 5 chunks (`SimpleHybridSearcher`, `eval_rag.py:125-126`;
    `rerank_size: 5` at `eval_rag.py:119`).
  - Stitch those 5 chunks into the context (`eval_rag.py:127-130`) and ask the model.
- **This is the exact seam** where the planned chunker-strategy experiment plugs in
  (see `CHUNKER_INTEGRATION_SCOPE.md`).

**How the model is called (both scripts):** `call_gpt()` (`eval_rag.py:65`,
`eval_full.py:56`). Locally it talks to **Ollama** through an OpenAI-compatible
endpoint set by env vars (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OLLAMA_NUM_CTX`). So
"gpt" here just means "the OpenAI-style API" — the actual model is our local
`qwen2.5:7b` or `llama3.2`.

**Prompt templates:** `evaluation/prompt.py` (one template per document type, for full
vs rag).

**Output:** `evaluation/prediction/<model>/<mode>_preds_<model>_<len>_<ctx>_<type>.jsonl`.
Each line has `question`, `prediction`, `ground_truth`. **These answers look good** —
e.g. our qwen2.5:7b answer about SC-LSTM correctly says it "learns from data rather
than relying on predefined rules," which matches the ground truth.

### Stage 3 — Scoring (this is broken)

`evaluation/compute_score_llm.py` reads each prediction file and grades every answer
with **another LLM call** — the "LLM-as-judge."

---

## 3. How the judge actually decides True / False

There is **no math, no similarity score, no cosine** in the grader. Judging is just one
more LLM call. Two parts:

**(a) Build a judge prompt** — `get_score_one_llm()` (`compute_score_llm.py:71`).
The standard prompt (`compute_score_llm.py:83-91`) is literally:

```
System: You are a discriminator that judges whether the predictions to questions are correct.

User: I will provide you with a question and its groundtruth answer, as well as an
answer from an AI assistant. You need to judge whether the AI assistant's answer is
correct based on the groundtruth answer. If it is correct, you should only output
True; if it is incorrect, only output False.

[Query]                <the question>
[Groundtruth Answer]   <the reference answer>
[AI Assistant's Answer] <our model's answer>

Now, start your judgment:
```

(The `hallu` question type gets a slightly different prompt — `compute_score_llm.py:73-81`.)
Note the judge only ever sees the **question + ground truth + answer**. It never sees
the 32k/128k document. So judging is a *short* task, even though generation was huge.

**(b) Turn the judge's text into a number** — `compute_score_llm.py:106-109`:

```python
if 'true' in response.lower():
    return 1.0
else:
    return 0.0
```

That's the whole scoring rule. `True` → 1.0, anything else → 0.0.

**(c) Aggregate** — a task's accuracy is just the average of those 1.0/0.0 values
across all its questions (`score_all / cnt_all`, `compute_score_llm.py:168`), then
written to a CSV and summary JSON (`compute_score_llm.py:181-221`).

---

## 4. Why every number comes out 0.0

Two failure modes stack on top of each other, and **both point at the judge model**:

**Failure mode 1 — weak reasoning, so it grades too literally.**
Deciding "does this answer *mean the same thing* as the ground truth?" requires real
comprehension. A strong model abstracts to meaning. A weak 3B/7B model latches onto
surface wording. Since our answer always paraphrases the ground truth (and the ground
truth is often much longer and more detailed), the weak judge sees "different words /
missing details" and concludes **incorrect -> False**. It lacks the calibration to say
"close enough, the key point is present."

**Failure mode 2 — the code defaults to False, and weak models trigger that default.**
Look again at `compute_score_llm.py:106-109`: it returns 1.0 *only if* the literal
string `true` appears. Weak models are bad at following "only output True/False" — they
ramble, hedge, explain, or even leak other languages (we saw qwen2.5:7b emit Chinese in
its answers). When the judge replies *"Well, it's mostly aligned but omits X, so..."*
and never cleanly says "True," the code silently scores it **0.0**. So any messy or
uncertain verdict is counted as wrong by default.

**Net effect:** the weak judge marks almost every answer False, the numerator stays
near zero, and `average -> 0.0` on every task.

---

## 5. The key insight: binary scoring is NOT the bug

Natural objection: *"If each score is a hard 1.0 or 0.0, how can a stronger judge
help?"*

Because the **binary is per question**, but the **metric is the average over hundreds
of questions**, which is continuous. Example, a 200-question task:

| Judge behavior | # marked True | Task accuracy |
|---|---|---|
| Weak judge (says False on nearly all) | 4 | 4/200 = **0.02** ("our 0.0") |
| Strong judge (says True where deserved) | 124 | 124/200 = **0.62** |

Same binary scoring. The only thing that changed is *how many the judge got right* —
i.e. the judge's strength. That sets the fraction, and the fraction is the score.

**Proof this is right:** the LaRA paper uses this *exact* binary True/False scoring,
with GPT-4o / Qwen-Max as the judge, and reports real numbers (~50-70% on many tasks).
Those numbers are literally baked into the baseline folders in
`evaluation/prediction/gpt-4o/`, `qwen2.5-72b-instruct/`, etc. If binary scoring forced
everything to 0.0, the paper could never report 62%. Binary isn't broken — our judge is.

**Where the binary *does* matter (later):** for a single genuinely half-right answer,
1.0/0.0 is lossy — even a strong judge must pick one side. If, after we install a
strong judge, we find it's unfairly zeroing partially-correct answers, *then* switching
to graded/partial credit earns its place. That's a Step 2 decision, made from evidence,
not now.

---

## 6. What is and isn't the problem (so we don't chase ghosts)

**Already fixed (this is why it runs at all)** — from the git history:
- Hardcoded cloud API key removed; endpoint/key now read from env.
- Missing GTE retrieval models swapped for auto-downloading BGE models (`eval_rag.py:40-41`).
- Reranker infinite-loop replaced (`FlagEmbeddingReranker` -> `SentenceTransformerRerank`).
- Scorer crashes on missing imports / partial runs / colon-in-model-name fixed
  (`compute_score_llm.py:195-199`).
- Ollama's small default context window raised via `OLLAMA_NUM_CTX` so long prompts
  aren't silently truncated (`eval_full.py:63-65`).

**NOT the problem:** retrieval, chunking, prompts, generation. Answers are good.

**THE problem:** the judge model is too weak. Concentrated at
`compute_score_llm.py:71-109` (prompt + binary parse) and, more fundamentally, the
`--eval_model` we hand the judge.

**Separate hard limit (not the judge):** the 128k full-context run does not fit in the
Mac's 16 GB RAM. 32k runs locally; 128k needs a bigger machine (ASU compute).

---

## 7. The plan (in priority order)

**Step 1 — Change the judge (do this first).**
The judge only handles short prompts, so we can afford a much bigger model than we could
for generation. Swap the judge to `qwen2.5:14b` (fits 16 GB for short judging prompts;
32B would not). No code change — just point `--eval_model` at it via the existing env
route. Then re-score the answers we already generated and watch the average climb off
zero. A strong judge fixes *both* failure modes at once: it reasons well enough to
accept a correct paraphrase, and it follows "only output True" cleanly so the parser
catches it.

**Step 2 — Only if 14B is still too harsh: tweak the scoring code.**
This is where we'd move off the hard 1.0/0.0 at `compute_score_llm.py:106-109` — e.g.
have the judge reason then emit a verdict we parse robustly (instead of a substring
default-to-False), or award partial credit. We decide the exact change based on *how*
14B fails, not preemptively. (14B is the ceiling that fits 16 GB, so if it's still not
enough, code is genuinely the next lever.)

**Still outstanding regardless:** 128k full-context needs ASU compute.

---

## 8. Likely meeting questions (and the answers)

- **"Is the pipeline wrong?"** No. Generation and retrieval work; answers are good. Only
  the grader is failing.
- **"Why not just use GPT-4o as the judge like the paper?"** That needs a paid API key,
  and we're deliberately keeping this local/free. A strong *local* judge (qwen2.5:14b)
  is the first thing to try.
- **"If scoring is binary, why does the judge's strength matter?"** The metric is the
  average over hundreds of questions, which is continuous; the judge's accuracy sets
  that fraction. See section 5.
- **"How will we know the new judge is actually trustworthy?"** We can sanity-check it
  against the paper's baseline predictions (the `gpt-4o/` etc. folders already contain
  strong-model answers the paper scored) — a good judge should rate those highly and
  roughly reproduce the paper's numbers.
- **"What about 128k?"** Doesn't fit 16 GB RAM; needs ASU compute. 32k runs locally now.

---

## Quick VS Code open list

| What | File:line |
|---|---|
| Judge prompt (standard) | `evaluation/compute_score_llm.py:83` |
| Judge prompt (hallu) | `evaluation/compute_score_llm.py:73` |
| Judge model call | `evaluation/compute_score_llm.py:102` |
| **The binary 1.0/0.0 parse** | `evaluation/compute_score_llm.py:106` |
| Accuracy = average | `evaluation/compute_score_llm.py:168` |
| RAG retrieval seam | `evaluation/eval_rag.py:89-130` |
| Full-context loader | `evaluation/eval_full.py:81-90` |
| Local model call (Ollama) | `evaluation/eval_rag.py:65` / `eval_full.py:56` |
| Prompt templates | `evaluation/prompt.py` |
</content>
</invoke>
