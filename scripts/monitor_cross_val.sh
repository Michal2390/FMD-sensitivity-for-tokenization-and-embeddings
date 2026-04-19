#!/bin/bash
# Monitor cross-validation and update README when done
set -e
cd /Users/michal.fereniec/PycharmProjects/FMD-sensitivity-for-tokenization-and-embeddings

LOG=logs/cross_val_3model.log

echo "=== Monitoring 3-model cross-validation ==="
echo "Started: $(date)"

# Wait for process to finish
while ps aux | grep "run_cross_dataset_validation" | grep -v grep | grep -v monitor > /dev/null 2>&1; do
    DONE=$(grep -c "→.*embeddings" "$LOG" 2>/dev/null || echo 0)
    STEP=$(grep "Step\|Statistical\|COMPLETE\|Bootstrap\|FMD rows" "$LOG" 2>/dev/null | tail -1 || echo "starting...")
    echo "  [$(date +%H:%M)] Embeddings: $DONE/192 | $STEP"
    sleep 120
done

echo ""
echo "=== Cross-validation process finished at $(date) ==="

# Check if it completed successfully
if grep -q "CROSS-DATASET VALIDATION" "$LOG" && grep -q "Report:" "$LOG"; then
    echo "✅ Cross-validation completed successfully"

    # Extract key results for README update
    echo ""
    echo "=== KEY RESULTS ==="
    grep "Spearman\|spearman_rho\|η²\|Bootstrap\|✅" "$LOG" | tail -20
else
    echo "⚠️ Cross-validation may have failed"
    grep "Traceback\|Error" "$LOG" | tail -10
fi

echo ""
echo "Total runtime logged in: $LOG"
echo "Report: results/reports/cross_validation/CROSS_VALIDATION_REPORT.md"

