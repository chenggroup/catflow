#!/bin/bash
set -e
cd "$(dirname "$0")"

# Dummy CP2K single-point calculation
# Replace with: cp2k.ssmp -i input.inp
# when CP2K is installed on HPC.

echo "[DUMMY] CP2K calculation in $PWD"
echo "[DUMMY] Would run: cp2k.ssmp -i input.inp"
sleep 0.5

# Create expected output file
cat > output << 'EOF'
 *****************************************
 **                 CP2K                **
 **    Dummy Single-Point Calculation   **
 *****************************************

 PROGRAM STARTED
 Total energy:      -42.000000000
 Run type:          ENERGY_FORCE
 DBCSR STATISTICS
  #######################################################################
 PROGRAM ENDED
EOF

touch cp2k-RESTART.wfn

echo "[DUMMY] CP2K done: output"
