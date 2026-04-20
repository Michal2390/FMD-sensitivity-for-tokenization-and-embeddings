#!/bin/bash
# Wait for current run_multi_genre_analysis.py to finish, then re-run with fixed MusicBERT
set -e

PROJ="/Users/michal.fereniec/PycharmProjects/FMD-sensitivity-for-tokenization-and-embeddings"
LOG="$PROJ/logs/5model_rerun.log"
CACHE="$PROJ/data/embeddings/cache"

echo "$(date) — Waiting for PID 11219 (run_multi_genre_analysis.py) to finish..."

# Wait for the current process to complete
while kill -0 11219 2>/dev/null; do
    sleep 30
done

echo "$(date) — Previous run finished. Preparing re-run..."

# Clear MusicBERT cache entries (so new model is used)
echo "Clearing MusicBERT cached embeddings..."
rm -f "$CACHE"/MusicBERT_*.npy "$CACHE"/MusicBERT_*_meta.json
echo "Cleared $(ls "$CACHE"/MusicBERT_* 2>/dev/null | wc -l) MusicBERT cache files"

# Back up previous results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
for f in "$PROJ/results/reports/multi_genre_fmd_results_5model.csv" \
         "$PROJ/results/reports/multi_genre_anova_results_5model.json"; do
    if [ -f "$f" ]; then
        cp "$f" "${f%.csv}_backup_${TIMESTAMP}.csv" 2>/dev/null || \
        cp "$f" "${f%.json}_backup_${TIMESTAMP}.json" 2>/dev/null
    fi
done

echo "$(date) — Starting re-run with fixed MusicBERT (manoskary/musicbert)..."
cd "$PROJ"
python -u run_multi_genre_analysis.py 2>&1 | tee "$LOG"

echo "$(date) — Re-run completed!"

