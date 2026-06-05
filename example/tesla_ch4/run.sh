#!/bin/bash
set -e

# CH4 dehydrogenation on Pt4 — minimal validation case
# Single iteration test: Train → Explore → Label
# Uses dummy computation commands since DeePMD/LAMMPS not installed locally

echo "================================================="
echo " CatFlow TESLA Validation — CH4 on Pt4"
echo "================================================="
echo ""

cd "$(dirname "$0")"

# Ensure omb is available
pip install oh-my-batch 2>/dev/null || true

# Step 0: Generate initial structures and training data
echo "=== Step 0: Setup ==="
bash 01-workflow/setup.sh

# Step 1-4: Single iteration
export ITER_NAME=000
echo ""
echo "=== Running iteration ${ITER_NAME} ==="
bash 01-workflow/iter.sh

# Verification
echo ""
echo "================================================="
echo " Validation Results"
echo "================================================="

ITER_DIR=10-workdir/iter-000
OK=0
FAIL=0

check() {
    if [ -f "$1" ]; then
        echo "  ✓ $2"
        OK=$((OK + 1))
    else
        echo "  ✗ $2 (missing: $1)"
        FAIL=$((FAIL + 1))
    fi
}

echo "Train:"
check "$ITER_DIR/00.train/000/frozen_model.pb" "frozen_model.pb"
check "$ITER_DIR/00.train/001/frozen_model.pb"  "model 1 frozen_model.pb"
check "$ITER_DIR/00.train/000/lcurve.out"       "lcurve.out"

echo "Explore:"
check "$(ls $ITER_DIR/01.model_devi/task.*/model_devi.out 2>/dev/null | head -1)" "model_devi.out"
check "$(ls $ITER_DIR/01.model_devi/task.500K.*/model_devi.out 2>/dev/null | head -1)" "model_devi.out (T=500K)"
explore_count=$(ls -d "$ITER_DIR/01.model_devi/task."*/ 2>/dev/null | wc -l)
echo "  ${explore_count} exploration tasks created"

echo "Label:"
LABEL_COUNT=$(ls -d "$ITER_DIR/02.fp/task."*/ 2>/dev/null | wc -l)
echo "  ${LABEL_COUNT} labeling tasks created"
check "$(ls $ITER_DIR/02.fp/task.*/output 2>/dev/null | head -1)" "CP2K output"

echo ""
echo "Total: $OK passed, $FAIL failed"
[ $FAIL -eq 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
