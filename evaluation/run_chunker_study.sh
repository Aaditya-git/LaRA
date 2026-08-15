#!/usr/bin/env bash
#
# Chonkie chunker comparison for LaRA's RAG path.
#
# "This week" deliverable: swap LaRA's chunking step (previously llama-index's
# SentenceSplitter) for each of Chonkie's 5 chunkers in turn, keep everything else
# (BGE embed + BM25 hybrid retrieval + rerank + LLM inference) untouched, judge with
# LaRA's own LLM-as-judge (both binary and 0-10 numeric), and produce one comparison
# table: rows = chunker, columns = per-task accuracy.
#
# Scope for this run (bounded, not the full 2,326-question benchmark):
#   context_length = 32k only (chunking cost/behavior is what's under test, not
#                     long-context capacity)
#   context_type    = book, financial, paper (all 3)
#   query_type      = location, reasoning, comp, hallu (all 4)
#   LARA_LIMIT       = first N examples per (chunker, context_type, query_type) cell
#
# Usage:
#   bash run_chunker_study.sh
#   LARA_LIMIT=30 CHUNKERS="token semantic" bash run_chunker_study.sh   # override scope
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PY="$SCRIPT_DIR/../venv/bin/python"

GEN="${GEN:-qwen2.5:7b}"
JUDGE="${JUDGE:-qwen2.5:14b}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32k}"
LARA_LIMIT="${LARA_LIMIT:-15}"
WORKERS="${WORKERS:-4}"
CHUNKERS="${CHUNKERS:-token sentence recursive semantic fast}"
CONTEXT_TYPES="${CONTEXT_TYPES:-book financial paper}"
QUERY_TYPES="${QUERY_TYPES:-location reasoning comp hallu}"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:11434/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"
export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-8192}"

banner() { echo; echo "=== $1 ==="; echo; }

banner "Chunker study: gen=$GEN judge=$JUDGE length=$CONTEXT_LENGTH limit=$LARA_LIMIT"
echo "chunkers:      $CHUNKERS"
echo "context_types: $CONTEXT_TYPES"
echo "query_types:   $QUERY_TYPES"

if ! "$PY" -c "import chonkie" 2>/dev/null; then
  banner "Installing chonkie[semantic] (missing from venv)"
  uv pip install --python "$PY" "chonkie[semantic]"
fi

# ---- 1. generate: one eval_rag.py call per (chunker, context_type, query_type) ----
banner "1. Generating RAG predictions for each chunker"
for chunker in $CHUNKERS; do
  for context_type in $CONTEXT_TYPES; do
    for query_type in $QUERY_TYPES; do
      out="./prediction/${GEN}/rag_preds_${GEN}_${chunker}_${CONTEXT_LENGTH}_${context_type}_${query_type}.jsonl"
      if [ -f "$out" ]; then
        echo "  skip (exists): $out"
        continue
      fi
      echo "  --- chunker=$chunker  context=$context_type  query=$query_type ---"
      LARA_LIMIT="$LARA_LIMIT" LARA_WORKERS="$WORKERS" \
        "$PY" eval_rag.py --chunker "$chunker" \
          --query_type "$query_type" --context_type "$context_type" \
          --context_length "$CONTEXT_LENGTH" --eval_model "$GEN"
    done
  done
done

# ---- 2. judge: binary + numeric, one call per chunker (each script sweeps its own context/query loop) ----
# LARA_JUDGE_DEBUG=1 so every question's binary verdict + numeric score/reason is logged
# per-question (not just the aggregate accuracy) -- these logs are what get sent to the
# professor, merged below into one file showing both scales side by side.
banner "2. Judging each chunker (binary + numeric)"
unset LARA_LIMIT
for chunker in $CHUNKERS; do
  echo "  --- binary judge:  chunker=$chunker ---"
  LARA_JUDGE_DEBUG=1 LARA_WORKERS="$WORKERS" \
    "$PY" compute_score_llm.py --eval_model "$GEN" --judge_model "$JUDGE" --chunker "$chunker"

  echo "  --- numeric judge: chunker=$chunker ---"
  LARA_JUDGE_DEBUG=1 LARA_WORKERS="$WORKERS" \
    "$PY" compute_score_numeric.py --eval_model "$GEN" --judge_model "$JUDGE" --chunker "$chunker"

  echo "  --- merging binary + numeric per-question logs: chunker=$chunker ---"
  "$PY" merge_judgments.py --eval_model "$GEN" --judge_model "$JUDGE" --chunker "$chunker"
done

# ---- 3. aggregate into one comparison table ----
banner "3. Building comparison table"
"$PY" aggregate_chunker_results.py --eval_model "$GEN" --judge_model "$JUDGE" --chunkers $CHUNKERS

banner "DONE"
echo "Per-chunker CSVs:      prediction/result/gen-*_chunker-*_all.csv (binary), numeric_gen-*_chunker-*_all.csv (numeric)"
echo "Per-question logs:     prediction/result/combined_judgments_gen-*_chunker-*.jsonl (binary + numeric side by side, one file per chunker -- send these to the professor)"
echo "Comparison table:      prediction/result/chunker_comparison.csv"
