#!/usr/bin/env bash
#
# Reclaim disk after the study. Removes the big 14b model (~9GB).
# KEEPS all results, debug logs, transcripts, and the matrix CSV for the PPT.
#
set -u
MODEL="qwen2.5:14b"

echo "This will remove '$MODEL' from Ollama (~9GB) and keep all study results."
read -r -p "Proceed? [y/N] " ans
[ "$ans" = "y" ] || [ "$ans" = "Y" ] || { echo "aborted."; exit 0; }

if command -v ollama >/dev/null 2>&1; then
  ollama rm "$MODEL"
else
  curl -s -X DELETE http://localhost:11434/api/delete -d "{\"name\":\"$MODEL\"}" \
    && echo "deleted '$MODEL' via API"
fi

echo
echo "Kept for the presentation:"
echo "  - prediction/result/STUDY_matrix.csv        (the 2x2 matrix)"
echo "  - prediction/result/gen-*_all.{jsonl,csv}   (per-cell scores)"
echo "  - prediction/result/judge_debug_gen-*.jsonl (every judge decision)"
echo "  - logs/study_*.log                          (full transcripts)"
echo
echo "Note: kept qwen2.5:7b (small) and the HF embed/rerank cache (reusable)."
echo "Disk reclaimed: ~9GB."
