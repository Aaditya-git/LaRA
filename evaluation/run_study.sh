#!/usr/bin/env bash
#
# 2x2 STUDY:  generator {qwen2.5:7b, qwen2.5:14b}  x  judge {qwen2.5:7b, qwen2.5:14b}
# Fixed: mode=RAG, doc=book, length=32k, N=10 questions, all 4 task types.
#
# Generates predictions ONCE per generator, then judges each set with both judges
# (4 cells), then aggregates into a comparison matrix for the presentation.
#
# Usage:  bash run_study.sh          (LIMIT=10 default; override: LIMIT=5 bash run_study.sh)
#
set -uo pipefail

WEAK="qwen2.5:7b"
STRONG="qwen2.5:14b"
LIMIT="${LIMIT:-10}"
MODE="rag"; LEN="32k"; CTX="book"
QTYPES=(location reasoning comp hallu)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
PY="$SCRIPT_DIR/../venv/bin/python"
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-8192}"   # RAG context is small; 8192 is plenty and keeps 14b light

LOG_DIR="$SCRIPT_DIR/logs"; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/study_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
banner(){ printf '\n\033[1m==== %s ====\033[0m\n' "$*"; }
echo "Transcript: $LOG_FILE"
echo "WEAK=$WEAK  STRONG=$STRONG  LIMIT=$LIMIT  mode=$MODE doc=$CTX len=$LEN"

# ---- checks ----
banner "CHECKS"
[ -x "$PY" ] || { echo "ERROR: venv python missing at $PY"; exit 1; }
TAGS="$(curl -s http://localhost:11434/api/tags)" || { echo "ERROR: Ollama down (ollama serve)"; exit 1; }
for m in "$WEAK" "$STRONG"; do
  echo "$TAGS" | grep -q "$m" || { echo "ERROR: '$m' not installed. Pull it: ollama pull $m"; exit 1; }
done
echo "OK: both models present."

# ---- clean (fresh study) ----
banner "CLEAN"
rm -f prediction/"$WEAK"/*.jsonl prediction/"$STRONG"/*.jsonl
rm -f prediction/result/gen-*_all.jsonl prediction/result/gen-*_order.jsonl \
      prediction/result/gen-*_all.csv prediction/result/judge_debug_gen-*.jsonl \
      prediction/result/numeric_gen-*_all.jsonl prediction/result/numeric_gen-*_order.jsonl \
      prediction/result/numeric_gen-*_all.csv prediction/result/numeric_judge_debug_gen-*.jsonl \
      prediction/result/STUDY_matrix.csv
echo "cleared old predictions + study results."

# ---- generate once per generator ----
generate() {
  local GEN="$1"; export LARA_LIMIT="$LIMIT"
  # RAG generate loads local embed+rerank models per worker; keep it light so the
  # retrieval models + Ollama's resident LLM fit in 16GB (was defaulting to 8).
  export LARA_WORKERS="${LARA_WORKERS:-2}"
  banner "GENERATE with $GEN  ($MODE/$CTX/$LEN, N=$LIMIT, workers=$LARA_WORKERS)"
  for q in "${QTYPES[@]}"; do
    local qfile="../datasets/query/${LEN}_${CTX}_${q}.jsonl"
    [ -f "$qfile" ] || { echo "  skip $q (no query file)"; continue; }
    echo "  --- $GEN / $q ---"
    "$PY" "eval_${MODE}.py" --eval_model "$GEN" \
        --context_length "$LEN" --context_type "$CTX" --query_type "$q" \
        || echo "  WARN: generation failed for $GEN/$q"
  done
  unset LARA_LIMIT LARA_WORKERS
}

# ---- judge one cell (binary True/False, then numeric 0-10, separate script/output) ----
judge() {
  local GEN="$1" JUDGE="$2"
  banner "JUDGE  generator=$GEN  by  judge=$JUDGE  (binary)"
  LARA_JUDGE_DEBUG=1 LARA_WORKERS=4 \
    "$PY" compute_score_llm.py --eval_model "$GEN" --judge_model "$JUDGE"
  banner "SCORE  generator=$GEN  by  judge=$JUDGE  (numeric 0-10)"
  LARA_JUDGE_DEBUG=1 LARA_WORKERS=4 \
    "$PY" compute_score_numeric.py --eval_model "$GEN" --judge_model "$JUDGE"
  banner "MERGE  generator=$GEN  by  judge=$JUDGE  (binary + numeric combined log)"
  "$PY" merge_judgments.py --eval_model "$GEN" --judge_model "$JUDGE"
}

generate "$WEAK"
generate "$STRONG"

judge "$WEAK"   "$WEAK"      # cell 1: weak gen  / weak judge
judge "$WEAK"   "$STRONG"    # cell 2: weak gen  / strong judge
judge "$STRONG" "$WEAK"      # cell 3: strong gen / weak judge
judge "$STRONG" "$STRONG"    # cell 4: strong gen / strong judge

banner "AGGREGATE"
"$PY" aggregate_study.py

banner "STUDY DONE"
echo "Transcript: $LOG_FILE"
echo "Matrix:     prediction/result/STUDY_matrix.csv"
