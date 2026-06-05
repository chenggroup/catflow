#!/bin/bash
set -e
cd "$(dirname "$0")"

# Dummy DeePMD training — generates expected output files
# Replace with: dp train input.json && dp freeze -o frozen_model.pb
# when DeePMD-kit is installed on HPC.

INPUT="input.json"
if [ ! -f "$INPUT" ]; then
    echo "ERROR: $INPUT not found"
    exit 1
fi

echo "[DUMMY] DeePMD training in $PWD"
echo "[DUMMY] Would run: dp train $INPUT"
sleep 1

# Create expected outputs (dummy)
touch frozen_model.pb
touch graph.pb

# Create a realistic lcurve.out (step rmse_trn rmse_e_trn rmse_f_trn)
# Format: step rmse_trn rmse_e_trn rmse_f_trn
STEPS=$(grep -o '"numb_steps": [0-9]*' "$INPUT" | grep -o '[0-9]*' || echo 200)
for step in $(seq 10 10 $STEPS); do
    rmse_f=$(python3 -c "print(${step} / ${STEPS} * 0.5 + 0.2)")
    rmse_e=$(python3 -c "print(${step} / ${STEPS} * 0.01)")
    echo "$step 0.1 $rmse_e $rmse_f" >> lcurve.out
done

echo "[DUMMY] Training done: frozen_model.pb, graph.pb, lcurve.out"
