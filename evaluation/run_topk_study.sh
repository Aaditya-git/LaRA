#!/usr/bin/env bash
#
# Retrieval chunk-count (top_k) study for LaRA's RAG path.
#
# "This week" deliverable: only run reasoning + comp (the two task types the
# professor cares about), sweep the number of chunks retrieved after rerank
# (rerank_size in simpleHybridSearcher.py) for reasoning across 5/10/15,
# and keep comp fixed at top_k=2 -- comp questions only ever need 2 segments
# (see RESUME.md Task A), so sweeping top_k there would just be noise.
#
# Question set: not the raw dataset order -- select_stress_questions.py (run
# once, ahead of this script) curates 15 "neat"/RAG-stressing questions per
# (context_type, query_type): comp picked by highest relative comp_parts
# spread, reasoning picked by context_order == "full" (whole-document
# questions) with a diverse-context_order fallback where a doc type (paper)
# has no "full" questions. Output lives in datasets/query_topk_study/ and is
# read via LARA_QUERY_DIR, so the original datasets/query/ files are untouched.
#
# Chunker held fixed at "token" (the winner of the earlier chunker study) so
# top_k is the only new variable.
#
# Usage:
#   python ../select_stress_questions.py   # once, if datasets/query_topk_study/ is stale/missing
#   bash run_topk_study.sh
#   REASONING_TOP_KS="5 10 20" bash run_topk_study.sh   # override scope
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PY="$SCRIPT_DIR/../venv/bin/python"

GEN="${GEN:-qwen2.5:7b}"
JUDGE="${JUDGE:-qwen2.5:14b}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32k}"
WORKERS="${WORKERS:-4}"
CHUNKER="${CHUNKER:-token}"
CONTEXT_TYPES="${CONTEXT_TYPES:-book financial paper}"
REASONING_TOP_KS="${REASONING_TOP_KS:-5 10 15}"
COMP_TOP_K="${COMP_TOP_K:-2}"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:11434/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"
export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-8192}"
export LARA_QUERY_DIR="${LARA_QUERY_DIR:-../datasets/query_topk_study}"

banner() { echo; echo "=== $1 ==="; echo; }

banner "Top_k study: gen=$GEN judge=$JUDGE chunker=$CHUNKER length=$CONTEXT_LENGTH"
echo "reasoning top_ks: $REASONING_TOP_KS"
echo "comp top_k:       $COMP_TOP_K (fixed)"
echo "context_types:    $CONTEXT_TYPES"
echo "query dir:        $LARA_QUERY_DIR"

if [ ! -f "../datasets/query_topk_study/32k_book_reasoning.jsonl" ]; then
  banner "Curated question set missing -- run select_stress_questions.py first"
  (cd .. && "$PY" select_stress_questions.py)
fi

# ---- 1. generate: reasoning swept across top_k, comp fixed ----
banner "1. Generating RAG predictions"
for context_type in $CONTEXT_TYPES; do
  for top_k in $REASONING_TOP_KS; do
    out="./prediction/${GEN}/rag_preds_${GEN}_${CHUNKER}_top${top_k}_${CONTEXT_LENGTH}_${context_type}_reasoning.jsonl"
    if [ -f "$out" ]; then
      echo "  skip (exists): $out"
      continue
    fi
    echo "  --- reasoning  context=$context_type  top_k=$top_k ---"
    LARA_WORKERS="$WORKERS" \
      "$PY" eval_rag.py --chunker "$CHUNKER" --top_k "$top_k" \
        --query_type reasoning --context_type "$context_type" \
        --context_length "$CONTEXT_LENGTH" --eval_model "$GEN"
  done

  out="./prediction/${GEN}/rag_preds_${GEN}_${CHUNKER}_top${COMP_TOP_K}_${CONTEXT_LENGTH}_${context_type}_comp.jsonl"
  if [ -f "$out" ]; then
    echo "  skip (exists): $out"
  else
    echo "  --- comp  context=$context_type  top_k=$COMP_TOP_K (fixed) ---"
    LARA_WORKERS="$WORKERS" \
      "$PY" eval_rag.py --chunker "$CHUNKER" --top_k "$COMP_TOP_K" \
        --query_type comp --context_type "$context_type" \
        --context_length "$CONTEXT_LENGTH" --eval_model "$GEN"
  fi
done

# ---- 2. judge: binary + numeric, one call per top_k cell ----
banner "2. Judging each top_k (binary + numeric)"
for top_k in $REASONING_TOP_KS; do
  echo "  --- binary judge:  top_k=$top_k (reasoning) ---"
  LARA_JUDGE_DEBUG=1 LARA_WORKERS="$WORKERS" \
    "$PY" compute_score_llm.py --eval_model "$GEN" --judge_model "$JUDGE" --chunker "$CHUNKER" --top_k "$top_k"

  echo "  --- numeric judge: top_k=$top_k (reasoning) ---"
  LARA_JUDGE_DEBUG=1 LARA_WORKERS="$WORKERS" \
    "$PY" compute_score_numeric.py --eval_model "$GEN" --judge_model "$JUDGE" --chunker "$CHUNKER" --top_k "$top_k"

  echo "  --- merging per-question logs: top_k=$top_k ---"
  "$PY" merge_judgments.py --eval_model "$GEN" --judge_model "$JUDGE" --chunker "$CHUNKER" --top_k "$top_k"
done

echo "  --- binary judge:  top_k=$COMP_TOP_K (comp, fixed) ---"
LARA_JUDGE_DEBUG=1 LARA_WORKERS="$WORKERS" \
  "$PY" compute_score_llm.py --eval_model "$GEN" --judge_model "$JUDGE" --chunker "$CHUNKER" --top_k "$COMP_TOP_K"

echo "  --- numeric judge: top_k=$COMP_TOP_K (comp, fixed) ---"
LARA_JUDGE_DEBUG=1 LARA_WORKERS="$WORKERS" \
  "$PY" compute_score_numeric.py --eval_model "$GEN" --judge_model "$JUDGE" --chunker "$CHUNKER" --top_k "$COMP_TOP_K"

echo "  --- merging per-question logs: top_k=$COMP_TOP_K ---"
"$PY" merge_judgments.py --eval_model "$GEN" --judge_model "$JUDGE" --chunker "$CHUNKER" --top_k "$COMP_TOP_K"

# ---- 3. aggregate into one comparison table ----
banner "3. Building comparison table"
"$PY" aggregate_topk_results.py --eval_model "$GEN" --judge_model "$JUDGE" --chunker "$CHUNKER" \
  --reasoning_top_ks $REASONING_TOP_KS --comp_top_k "$COMP_TOP_K"

banner "DONE"
echo "Per-question logs:  prediction/result/combined_judgments_gen-*_chunker-${CHUNKER}_topk-*.jsonl"
echo "Comparison table:   prediction/result/topk_comparison.csv"
