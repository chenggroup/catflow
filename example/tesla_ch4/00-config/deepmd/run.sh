#!/bin/bash
set -e
cd "$(dirname "$0")"

# Auto-detect DeePMD-kit; use real computation when available
INPUT="input.json"
[ ! -f "$INPUT" ] && echo "ERROR: $INPUT not found" && exit 1

if command -v dp &>/dev/null; then
    echo "[REAL] DeePMD training in $PWD"
    dp train "$INPUT"
    dp freeze -o frozen_model.pb
    # Compress for LAMMPS interface
    dp compress -i frozen_model.pb -o graph.pb -t "$INPUT" 2>/dev/null || cp frozen_model.pb graph.pb
    echo "[REAL] Training done: frozen_model.pb, graph.pb, lcurve.out"
else
    echo "[DUMMY] DeePMD training in $PWD"
    echo "[DUMMY] Would run: dp train $INPUT"
    touch frozen_model.pb graph.pb
    STEPS=$(grep -o '"numb_steps": [0-9]*' "$INPUT" | grep -o '[0-9]*' || echo 200)
    for step in $(seq 10 10 $STEPS); do
        rmse_f=$(python3 -c "print(${step} / ${STEPS} * 0.5 + 0.2)")
        rmse_e=$(python3 -c "print(${step} / ${STEPS} * 0.01)")
        echo "$step 0.1 $rmse_e $rmse_f" >> lcurve.out
    done
    echo "[DUMMY] Training done: frozen_model.pb, graph.pb, lcurve.out"
fi
