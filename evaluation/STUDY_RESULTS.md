# Can we trust a small local LLM-as-judge? A 2×2 generator × judge study

**Author:** Aaditya Bhilegaonkar · **Machine:** MacBook Air, Apple M2, 16 GB RAM
**Setting:** LaRA benchmark, RAG mode, `book` document, `32k` context, N=10 questions per task type

> **Run complete** (2026-07-08). All cells below are from the live local run. The single
> exception is `qwen2.5:7b / comp`, which failed on a transient memory issue and is `n/a`.

---

## 1. The question

Our earlier finding (see `PROGRESS_NOTES.md`, `JUDGE_ISSUE_WALKTHROUGH.md`) was that the
LaRA **generation** pipeline works well locally, but **scoring** is the bottleneck: the
LaRA paper grades answers with a very large judge (GPT-4o / Qwen-Max), and a weak local
judge tends to say "False" even when the answer is right — collapsing every accuracy
number to 0.0.

This study asks the practical follow-up: **how much does the judge's size actually change
the verdict?** If a 7B judge and a 14B judge mostly agree, we can grade cheaply and locally.
If they diverge, local accuracy numbers cannot be trusted and we need a strong (API) judge.

## 2. The design — a 2×2

We vary two things independently and hold everything else fixed:

- **Generator** (who writes the answer): `qwen2.5:7b` (weak) vs `qwen2.5:14b` (strong)
- **Judge** (who grades the answer): `qwen2.5:7b` (weak) vs `qwen2.5:14b` (strong)

Fixed: mode = RAG, document = book, context = 32k, N = 10 questions × 4 task types
(`location`, `reasoning`, `comp` = comparison, `hallu` = hallucination detection).

Answers are generated **once per generator**, then graded by **both** judges, so the two
judges see identical answers. That isolates the judge's effect.

```
                    JUDGE: 7b (weak)      JUDGE: 14b (strong)
  GEN: 7b  (weak)   cell 1                cell 2
  GEN: 14b (strong) cell 3                cell 4
```

## 3. Results — accuracy matrix (RAG · book · 32k · % correct)

| Generator → Judge          | location | reasoning | comp | hallu | overall |
|----------------------------|:--------:|:---------:|:----:|:-----:|:-------:|
| 7b gen  / 7b judge         | 70.0 | 40.0 | n/a  | 90.0  | 66.7 |
| 7b gen  / 14b judge        | 60.0 | 50.0 | n/a  | 90.0  | 66.7 |
| 14b gen / 7b judge         | 60.0 | 40.0 | 10.0 | 100.0 | 52.5 |
| 14b gen / 14b judge        | 60.0 | 80.0 | 50.0 | 90.0  | 70.0 |

_Overalls for the 7b-generator rows average 3 task types (comp excluded); the 14b-generator
rows average all 4. For an apples-to-apples generator comparison, use the 3 shared task types
(location, reasoning, hallu): under the **strong (14b) judge**, 14b gen = **76.7%** vs 7b gen =
**66.7%**; under the **weak (7b) judge**, both tie at **66.7%**._

> **Note on `comp` (7b generator):** the `qwen2.5:7b / comp` generation cell did not
> complete due to a transient memory issue during the run, so its two cells are left blank
> (`n/a`). The remaining three task types are unaffected.

**How to read it:**
- Compare **rows 1 vs 2** (and **3 vs 4**): same answers, different judge → *the judge effect*.
- Compare **rows 1 vs 3** (and **2 vs 4**): same judge, different generator → *the generator effect*.
- If swapping the judge moves accuracy more than swapping the generator, the score is
  telling us more about the grader than about the model under test.

## 4. Results — judge agreement (the trustworthiness signal)

For the **same** set of answers, how often do the 7b and 14b judges return the same verdict?

| Answers from | # questions | agree | 7b too STRICT (7b=F, 14b=T) | 7b too LENIENT (7b=T, 14b=F) |
|--------------|:-----------:|:-----:|:---------------------------:|:----------------------------:|
| 7b generator  | 30 | 28 (**93%**) | 1 | 1 |
| 14b generator | 40 | 31 (**78%**) | **8** | 1 |

- **agree** = fraction of answers both judges grade identically. High = the cheap judge is safe.
- **too STRICT** = the weak judge fails an answer the strong judge passes — the exact failure
  mode we suspected (rejects correct answers on wording differences).
- **too LENIENT** = the weak judge passes an answer the strong judge fails.

## 5. Takeaways

1. **The judge moves the score more than the generator does.** For the *identical* set of 14b
   answers, swapping the judge from 7b → 14b raises overall accuracy from **52.5% → 70.0%**
   (+17.5 pts) — driven almost entirely by `reasoning` (40 → 80) and `comp` (10 → 50). Nothing
   about the model under test changed; only the grader did.

2. **The weak judge is asymmetrically too strict, and it gets worse on harder answers.** On the
   14b generator's longer, more elaborate answers, the 7b and 14b judges agree only **78%** of
   the time, and the disagreement is lopsided: **8 cases where the 7b judge failed an answer the
   14b judge passed**, vs just 1 the other way. On the shorter 7b answers, agreement is **93%**
   (1 strict / 1 lenient). So the weak judge's unreliability scales with answer length/complexity
   — exactly the failure mode behind the earlier all-`0.0` result.

3. **A weak judge can invert the benchmark's own conclusion.** On the 3 shared task types, the
   weak judge rates the 7b and 14b generators as a **tie (66.7% each)** — it cannot tell the
   stronger model apart. The strong judge correctly separates them (**14b 76.7% > 7b 66.7%**).
   A benchmark exists to rank models; a weak local judge erases the ranking.

4. **Practical rule.** Grading locally with a small judge is only safe for short/simple answers.
   For the real RAG-vs-LC comparison, generation can stay fully local, but **grading needs a
   strong (API) judge** on the short scoring prompts — the cheap use of a key. Local accuracy
   with a small judge should be read as a **lower bound**, not a true score.

## 6. Why this run is possible at all (infra note)

The 2×2 study previously **OOM-killed** this 16 GB machine: the RAG eval loaded a fresh
embedding model + reranker for *every* question across 8 worker threads (~1.4 GB each,
93 embed + 46 reranker loads in one partial run), stacked on top of Ollama's 9 GB resident
`qwen2.5:14b`. Fix: a process-wide model cache (`model_cache.py`) loads each retrieval model
**once** and shares it across all questions and threads, plus lower generate concurrency
(`LARA_WORKERS=2`). Peak dropped from ~20 GB+ to ~9 GB (Ollama) + ~1.4 GB (models) — fits 16 GB.

## 7. Limitations

- **Single slice:** RAG / book / 32k only. No Full-Context (LC) comparison here, and no
  128k — the 128k KV-cache does not fit in 16 GB (needs ASU compute).
- **Both judges are local and small.** Neither 7b nor 14b is the paper's GPT-4o/Qwen-Max
  grader; agreement between them bounds *consistency*, not *ground-truth correctness*.
- **N = 10 per task type** (40 questions per generator) — directional, not statistically tight.


2×2 accuracy matrix (RAG · book · 32k · % correct):

┌─────────────────────┬──────────┬───────────┬──────┬───────┬─────────┐
│  Generator → Judge  │ location │ reasoning │ comp │ hallu │ overall │
├─────────────────────┼──────────┼───────────┼──────┼───────┼─────────┤
│ 7b gen / 7b judge   │    70    │    40     │ n/a  │  90   │  66.7   │
├─────────────────────┼──────────┼───────────┼──────┼───────┼─────────┤
│ 7b gen / 14b judge  │    60    │    50     │ n/a  │  90   │  66.7   │
├─────────────────────┼──────────┼───────────┼──────┼───────┼─────────┤
│ 14b gen / 7b judge  │    60    │    40     │  10  │  100  │  52.5   │
├─────────────────────┼──────────┼───────────┼──────┼───────┼─────────┤
│ 14b gen / 14b judge │    60    │    80     │  50  │  90   │  70.0   │
└─────────────────────┴──────────┴───────────┴──────┴───────┴─────────┘