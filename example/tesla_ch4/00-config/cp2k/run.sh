#!/bin/bash
set -e
cd "$(dirname "$0")"

# Auto-detect CP2K; use real computation when available
INPUT="${INPUT:-input.inp}"

if command -v cp2k.ssmp &>/dev/null; then
    echo "[REAL] CP2K calculation in $PWD"
    cp2k.ssmp -i "$INPUT" > output 2>&1
elif command -v cp2k.popt &>/dev/null; then
    echo "[REAL] CP2K (parallel) calculation in $PWD"
    cp2k.popt -i "$INPUT" > output 2>&1
else
    echo "[DUMMY] CP2K calculation in $PWD"
    echo "[DUMMY] Would run: cp2k.ssmp -i $INPUT"
    sleep 0.5
    cat > output << 'EOF'
 *****************************************
 **                 CP2K                **
 **    Dummy Single-Point Calculation   **
 *****************************************
 PROGRAM STARTED
 Total energy:      -42.000000000
 Run type:          ENERGY_FORCE
 PROGRAM ENDED
EOF
    touch cp2k-RESTART.wfn
    echo "[DUMMY] CP2K done: output"
fi
