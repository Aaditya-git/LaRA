# RESUME — read this first when starting work on this repo

**Purpose of this file:** onboarding for Claude (or a new collaborator) picking up the LaRA
work on the ASU server. It states where we are, the two tasks the professor asked for, the
exact code that needs changing, and the design decisions that are still open. Read it, then
read `PROGRESS_NOTES.md`, `STUDY_RESULTS.md`, and `JUDGE_ISSUE_WALKTHROUGH.md` for depth.

---

## 1. One-paragraph context

This is a fork of LaRA (Long-context vs RAG benchmark). We (Aaditya + Sarvesh, ASU FURI, for
Prof. Jia Zou) got it running locally on a 16 GB Mac, fixed the many things that stopped it
running, and discovered the **LLM-as-judge is the bottleneck**: a small local judge returns
"False" on correct-but-reworded answers and collapses accuracy toward 0. A 2×2 generator×judge
study confirmed the judge moves the score more than the model under test does. We are now moving
to an **ASU server (125 GB RAM, ~1.4 TB free disk)** which unblocks (a) 128k full-context runs
that never fit on the Mac, and (b) a strong judge.

## 2. Two tasks from the professor (both open)

### Task A — Identify the questions that genuinely stress RAG
The professor's observation: most LaRA questions are answerable from a small number of chunks,
so they don't stress retrieval. He asked us to find the questions that "really test RAG by
requiring several chunks from all over the documents."

**What the data actually shows** (computed over all 2,326 questions in `datasets/query/`):

| Task      | Count | Segments needed | Spread across doc? |
|-----------|------:|-----------------|--------------------|
| location  | 763   | 1               | No                 |
| hallu     | 608   | 1               | No                 |
| reasoning | 606   | 1               | No                 |
| **comp**  | 349 (15%) | **exactly 2** | Sometimes        |

- Only `comp` questions require 2 segments (field `comp_parts`, a pair of segment indices).
  Everything else has a single `context_order`.
- **LaRA has NO question needing 3+ segments.** Multi-chunk difficulty is structurally capped at 2.
- `comp_parts` spread |a−b| distribution: spread 1 (adjacent) = 184, spread 2 = 115,
  **spread ≥3 = only 50 questions** (~2% of the benchmark). Those 50 are the genuinely
  "from all over the document" cases.
- Docs have different segment counts (**32k docs ≈ 3 segments; 128k docs up to 10**), so raw
  spread is not comparable across docs. The honest ranking metric is
  **relative spread = (b−a) / (segments−1)**.

**Honest bottom line to give the professor:** LaRA can't really test "several chunks from all
over"; the best it offers is comp questions with a large relative spread. We can isolate and
rank that subset, but generating genuinely multi-hop questions would require going beyond LaRA.

**Deliverable form: not yet decided** — options are (1) a reusable script that tags every
question with relative-spread and emits the RAG-hard subset, (2) a written analysis for the
professor, or (3) both. Confirm with Aaditya before building.

### Task B — Replace the binary True/False judge with a numerical scale
The professor wants graded scores instead of True/False, because complex answers are partially
correct and a hard 0/1 loses that. He pasted three real examples: one questionable "True" and
two harsh "False"s on answers that were largely right. This matches our own judge-weakness finding.

**Exactly where the code lives:**
- Judge function: `evaluation/compute_score_llm.py` → `get_score_one_llm()` (~lines 130–171).
  It builds a prompt asking for "True/False" and parses the verdict at **line 165**:
  `if 'true' in response.lower(): return 1.0 else 0.0`.
- Aggregation (`compute_score_llm.py` ~lines 211–249) **already averages floats** into per-task
  "accuracy", so a fractional/graded score flows through with almost no aggregation change.
- The only binary-hardcoded spots: the True/False prompt + parse (line ~132–168), and the
  `n_true = int(round(score_all))` TRUE/FALSE counting (~line 235) which becomes meaningless
  with graded scores and must be handled.

**Design decisions still OPEN (do not just build — brainstorm these with Aaditya first):**
1. **Scale:** continuous 0.0–1.0, integer 1–5 Likert, 0–10, or a multi-dimension rubric
   (e.g. correctness + completeness + faithfulness averaged)?
2. **Backward compatibility:** the paper reports binary accuracy. Do we keep binary alongside
   the graded score (for leaderboard comparability) or replace it?
3. **Judge model:** graded scoring is *harder* for a weak judge than binary. Likely requires a
   strong (API) judge; local small models proved unreliable even for binary.
4. **Per task type:** hallu and location are close to binary by nature; graded scoring mainly
   helps reasoning and comp. Decide whether grading applies everywhere or only where it helps.

## 3. Environment notes for running here
- **RAM 125 GB, disk ~1.4 TB free** — 128k full-context now fits (was impossible on the 16 GB Mac).
- **Shared machine** — be considerate with large model loads; namespace our output folders.
- **Ollama models present locally on the Mac** were `qwen2.5:7b`, `llama3.2`, `nomic-embed-text`.
  `qwen2.5:14b` was deleted to free disk — re-pull if a 14b cell is needed. On the server, check
  what's installed (`ollama list`) and what GPU exists (`nvidia-smi`) before choosing models.
- Local run instructions: `RUNNING_LOCALLY.md`. OOM fix already in place
  (`evaluation/model_cache.py` + `LARA_WORKERS`).

## 4. Suggested first moves on the server
1. Confirm environment: `nvidia-smi`, `nproc`, `ollama list`, Python/venv, `pip install -r requirements.txt`.
2. Reproduce a small smoke run (32k / book / one task) to confirm the pipeline works here.
3. Then tackle Task A (agree deliverable form) and Task B (brainstorm the scale design) — in that order.

## 5. Existing docs map
- `PROGRESS_NOTES.md` — full local-run history, decisions, code fixes, blockers.
- `STUDY_RESULTS.md` (in `evaluation/`) — the 2×2 generator×judge study + conclusions.
- `JUDGE_ISSUE_WALKTHROUGH.md` — deep dive on the judge failure.
- `CHUNKER_INTEGRATION_SCOPE.md` — scoped (not built) plan to plug external chunkers into RAG.
- `RUNNING_LOCALLY.md` — how to run with Ollama.
