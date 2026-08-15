#!/usr/bin/env bash
#
# LaRA local pipeline, from scratch:  check -> clear -> generate -> verify -> judge -> score
#
# Usage:
#   bash run_pipeline.sh
#
# Override any setting inline, e.g.:
#   MODEL=qwen2.5:14b LIMIT=10 WORKERS=4 DEBUG=0 bash run_pipeline.sh
#   LIMIT="" bash run_pipeline.sh          # LIMIT="" means ALL questions (slow!)
#
set -uo pipefail

# ------------------------------ settings (edit freely) ------------------------------
MODEL="${MODEL:-qwen2.5:7b}"     # Ollama model used for BOTH generation and judging
LIMIT="${LIMIT:-5}"              # questions per config (smoke test). "" = all questions
DEBUG="${DEBUG:-1}"              # 1 = print every judge decision + write debug log
WORKERS="${WORKERS:-1}"         # judge parallelism. 1 = readable blocks, 4 = faster

# Configs to run: "mode context_length context_type query_type"
#   mode           : full | rag
#   context_length : 32k | 128k          (128k needs lots of RAM)
#   context_type   : book | paper | financial
#   query_type     : location | reasoning | comp | hallu
CONFIGS=(
  "full 32k paper reasoning"
  "full 32k book reasoning"
  "rag  32k book reasoning"
  # uncomment to also test comparison & hallucination tasks:
  # "rag  32k book comp"
  # "rag  32k book hallu"
)
# ------------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PY="$SCRIPT_DIR/../venv/bin/python"

# same sanitizing + gen/judge tag compute_score_llm.py and compute_score_numeric.py use
# internally to name their output files (":" and "/" -> "-"). MODEL is used as both
# generator and judge here, so CELL_TAG covers both scripts' outputs.
SAFE_MODEL="${MODEL//:/-}"; SAFE_MODEL="${SAFE_MODEL//\//-}"
CELL_TAG="gen-${SAFE_MODEL}_judge-${SAFE_MODEL}"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:11434/v1}"  # local Ollama
export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"                      # dummy key
export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-32768}"                       # full context

banner(){ printf '\n\033[1m==== %s ====\033[0m\n' "$*"; }

# ---- full transcript logging (everything printed also goes to a file) ----
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S)_${MODEL//[:\/]/-}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Full transcript logging to: $LOG_FILE"

# ---- 0. prerequisites ----
banner "0. Checks"
[ -x "$PY" ] || { echo "ERROR: venv python not found at $PY"; exit 1; }
# check via the Ollama HTTP API (the `ollama` CLI may not be on bash's PATH)
TAGS="$(curl -s http://localhost:11434/api/tags)" \
  || { echo "ERROR: Ollama not reachable on :11434. Start it: ollama serve"; exit 1; }
echo "$TAGS" | grep -q "$MODEL" \
  || { echo "ERROR: model '$MODEL' not installed. Pull it: ollama pull $MODEL"; exit 1; }
echo "OK: venv python, Ollama, and model '$MODEL' are all present."
echo "MODEL=$MODEL  LIMIT=${LIMIT:-ALL}  DEBUG=$DEBUG  WORKERS=$WORKERS"

# ---- 1. clear old outputs (both gen and scoring skip existing files) ----
banner "1. Clearing old outputs for $MODEL"
rm -f prediction/"$MODEL"/*.jsonl
rm -f "prediction/result/${CELL_TAG}_all.jsonl" \
      "prediction/result/${CELL_TAG}_order.jsonl" \
      "prediction/result/${CELL_TAG}_all.csv" \
      "prediction/result/judge_debug_${CELL_TAG}.jsonl" \
      "prediction/result/numeric_${CELL_TAG}_all.jsonl" \
      "prediction/result/numeric_${CELL_TAG}_order.jsonl" \
      "prediction/result/numeric_${CELL_TAG}_all.csv" \
      "prediction/result/numeric_judge_debug_${CELL_TAG}.jsonl"
echo "cleared."

# ---- 2. generate answers ----
banner "2. Generating (${LIMIT:-ALL} questions/config)"
if [ -n "$LIMIT" ]; then export LARA_LIMIT="$LIMIT"; else unset LARA_LIMIT; fi
for cfg in "${CONFIGS[@]}"; do
  read -r mode len ctx qtype <<< "$cfg"
  qfile="../datasets/query/${len}_${ctx}_${qtype}.jsonl"
  if [ ! -f "$qfile" ]; then
    echo "  SKIP (no query file): $cfg -> $qfile"
    continue
  fi
  echo "  --- $mode / $len / $ctx / $qtype ---"
  "$PY" "eval_${mode}.py" --eval_model "$MODEL" \
      --context_length "$len" --context_type "$ctx" --query_type "$qtype" \
      || echo "  WARN: generation failed for: $cfg"
done

# ---- 3. verify predictions exist before scoring ----
banner "3. Prediction counts"
shopt -s nullglob
files=(prediction/"$MODEL"/*.jsonl)
if [ ${#files[@]} -eq 0 ]; then
  echo "ERROR: no prediction files produced. Aborting before scoring."; exit 1
fi
for f in "${files[@]}"; do printf "  %3d  %s\n" "$(wc -l < "$f")" "$(basename "$f")"; done

# ---- 4. judge / score (binary True/False, LaRA's own judge) ----
banner "4. Judging with $MODEL (binary)"
unset LARA_LIMIT
LARA_JUDGE_DEBUG="$DEBUG" LARA_WORKERS="$WORKERS" \
  "$PY" compute_score_llm.py --eval_model "$MODEL"

# ---- 5. numeric scoring (0-10 scale + derived binary, separate script/output) ----
banner "5. Scoring with $MODEL (numeric 0-10)"
LARA_JUDGE_DEBUG="$DEBUG" LARA_WORKERS="$WORKERS" \
  "$PY" compute_score_numeric.py --eval_model "$MODEL"

# ---- 6. merge both judging scales into one combined log (needs DEBUG=1 logs) ----
if [ "$DEBUG" = "1" ]; then
  banner "6. Merging binary + numeric judgments"
  "$PY" merge_judgments.py --eval_model "$MODEL"
else
  echo "DEBUG=0: skipping merge (no per-question logs to combine)."
fi

banner "DONE"
echo "Binary debug log:   prediction/result/judge_debug_${CELL_TAG}.jsonl"
echo "Numeric debug log:  prediction/result/numeric_judge_debug_${CELL_TAG}.jsonl"
echo "Combined debug log: prediction/result/combined_judgments_${CELL_TAG}.jsonl"
