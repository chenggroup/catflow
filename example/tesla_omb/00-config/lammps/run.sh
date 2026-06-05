#!/bin/bash
set -e
cd "$(dirname "$0")"

INPUT="${INPUT:-input.lammps}"
NP="${NP:-1}"

echo "Starting LAMMPS simulation in $PWD"
echo "Input: ${INPUT}, NP: ${NP}"

if [ "$NP" -gt 1 ] && command -v mpirun &>/dev/null; then
    mpirun -np "$NP" lmp -i "$INPUT"
elif command -v lmp &>/dev/null; then
    lmp -i "$INPUT"
elif command -v lmp_mpi &>/dev/null; then
    mpirun -np "$NP" lmp_mpi -i "$INPUT"
elif command -v lmp_serial &>/dev/null; then
    lmp_serial -i "$INPUT"
else
    echo "ERROR: LAMMPS not found"
    exit 1
fi

echo "Simulation complete: $PWD"
