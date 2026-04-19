#!/bin/bash
# Auto-run all 4 analysis steps sequentially
set -e
cd /Users/michal.fereniec/PycharmProjects/FMD-sensitivity-for-tokenization-and-embeddings

echo "=== Waiting for run_multi_genre_analysis.py (PID 65295) to finish ==="
while kill -0 65295 2>/dev/null; do
    DONE=$(grep -c "→.*embeddings" logs/multi_genre_run.log 2>/dev/null || echo 0)
    echo "  [$(date +%H:%M)] Embedding extraction: $DONE / 192"
    sleep 60
done
echo "=== run_multi_genre_analysis.py FINISHED ==="
echo ""

# Check if it succeeded
if grep -q "MULTI-GENRE ANALYSIS COMPLETE" logs/multi_genre_run.log 2>/dev/null; then
    echo "✅ Multi-genre analysis completed successfully"
else
    echo "⚠️  Multi-genre may have failed, check logs/multi_genre_run.log"
    echo "Continuing anyway..."
fi

echo ""
echo "=== Step 3: Running Interaction Analysis ==="
python -u run_interaction_analysis.py 2>&1 | tee logs/interaction_run.log
echo ""

echo "=== Step 4: Running Cross-Dataset Validation ==="
python -u run_cross_dataset_validation.py --source midicaps 2>&1 | tee logs/cross_validation_run.log
echo ""

echo "=========================================="
echo "ALL ANALYSES COMPLETE"
echo "=========================================="
echo "Results:"
echo "  Multi-genre:  results/reports/lakh_multi/"
echo "  Interaction:  results/reports/lakh_multi/INTERACTION_MECHANISM_REPORT.md"
echo "  Cross-valid:  results/reports/cross_validation/"
echo "  Plots:        results/plots/paper/"

