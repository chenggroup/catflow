#!/bin/bash
set -e
cd "$(dirname "$0")"

# Auto-detect LAMMPS; use real computation when available
INPUT="${INPUT:-explore.in}"

if command -v lmp &>/dev/null; then
    echo "[REAL] LAMMPS simulation in $PWD"
    lmp -i "$INPUT"
elif command -v lmp_mpi &>/dev/null; then
    echo "[REAL] LAMMPS (MPI) simulation in $PWD"
    lmp_mpi -i "$INPUT"
elif command -v lmp_serial &>/dev/null; then
    echo "[REAL] LAMMPS (serial) simulation in $PWD"
    lmp_serial -i "$INPUT"
else
    echo "[DUMMY] LAMMPS simulation in $PWD"
    echo "[DUMMY] Would run: lmp -i $INPUT"
    NSTEPS=$(grep 'variable.*NSTEPS' "$INPUT" | grep -oP 'equal\s+\K[0-9]+' || echo 100)
    DT=$(grep '^timestep' "$INPUT" | awk '{print $2}' || echo 0.002)
    TEMP=$(grep 'variable.*TEMP' "$INPUT" | grep -oP 'equal\s+\K[0-9.]+' || echo 300)
    for step in $(seq 10 10 $NSTEPS); do
        if [ $step -eq 50 ] || [ $step -eq 60 ]; then
            devi_f="0.18"
        elif [ $step -eq 80 ]; then
            devi_f="0.50"
        else
            devi_f="0.05"
        fi
        echo "$(python3 -c "print($step * $DT)") 0.02 0.01 $devi_f" >> model_devi.out
    done
    mkdir -p traj
    for frame in $(seq 0 5); do touch "traj/${frame}.lammpstrj"; done
    echo "[DUMMY] MD done: model_devi.out, traj/*.lammpstrj"
fi
