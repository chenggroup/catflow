#!/bin/bash
set -e
cd "$(dirname "$0")"

# Dummy LAMMPS MD — generates expected output files
# Replace with: lmp -i explore.in
# when LAMMPS is installed on HPC.

echo "[DUMMY] LAMMPS simulation in $PWD"
echo "[DUMMY] Would run: lmp -i explore.in"
sleep 1

# Read step count from explore.in
NSTEPS=$(grep '^variable.*NSTEPS' explore.in | grep -oP 'equal\s+\K[0-9]+' || echo 100)
DT=$(grep '^timestep' explore.in | grep -oP '\S+$' || echo 0.002)
TEMP=$(grep '^variable.*TEMP' explore.in | grep -oP 'equal\s+\K[0-9.]+' || echo 300)

# Create model_devi.out (step max_devi_f max_devi_e max_devi_v)
for step in $(seq 10 10 $NSTEPS); do
    # Some frames have moderate deviation (candidates), most are low (accurate)
    if [ $step -eq 50 ] || [ $step -eq 60 ]; then
        devi_f="0.18"  # candidate
    elif [ $step -eq 80 ]; then
        devi_f="0.50"  # failed
    else
        devi_f="0.05"  # accurate
    fi
    echo "$(python3 -c "print($step * $DT)") 0.02 0.01 $devi_f" >> model_devi.out
done

# Create dummy trajectory files
mkdir -p traj
for frame in $(seq 0 5); do
    touch "traj/${frame}.lammpstrj"
done

echo "[DUMMY] MD done: model_devi.out, traj/*.lammpstrj"
