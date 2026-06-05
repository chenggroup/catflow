#!/bin/bash
set -e

# TESLA setup: prepare initial training data
# Converts AIMD trajectory or initial structures to dpdata format

WORK_DIR=./10-workdir
INIT_DATA_DIR="${WORK_DIR}/init_data"
mkdir -p "${WORK_DIR}" "${INIT_DATA_DIR}"

# Skip if already done
if [ -f "${WORK_DIR}/setup.done" ]; then
    echo "Setup already completed. Remove ${WORK_DIR}/setup.done to redo."
    exit 0
fi

# --- Generate sample initial data ---
# In a real workflow, this would read from AIMD trajectories.
# Here we create a minimal test structure for validation.

echo "Generating sample initial structures..."

python3 -c "
from ase.io import write
from ase.build import molecule

# Create a simple water molecule as test structure
atoms = molecule('H2O')
atoms.set_cell([10, 10, 10])
atoms.center()

# Write as XYZ for the workflow
write('${WORK_DIR}/init.xyz', atoms)
write('${WORK_DIR}/POSCAR', atoms, vasp5=True)

# Generate a few frames for testing
import numpy as np
frames = [atoms]
for i in range(10):
    a = atoms.copy()
    a.positions += np.random.randn(*a.positions.shape) * 0.05
    frames.append(a)
write('${WORK_DIR}/trajectory.xyz', frames)
"

# Convert initial structures to dpdata format
echo "Creating initial dataset..."
ai2-kit tool dpdata read "${WORK_DIR}/POSCAR" \
    --fmt 'vasp/poscar' \
    write "${INIT_DATA_DIR}" --fmt 'deepmd/npy' 2>/dev/null || {
    echo "ai2-kit not available, creating minimal dataset manually"
    mkdir -p "${INIT_DATA_DIR}/set.000"
    python3 -c "
import numpy as np
# Create minimal DeePMD dataset
np.save('${INIT_DATA_DIR}/set.000/coord.npy', np.random.randn(1, 9).astype(np.float32))
np.save('${INIT_DATA_DIR}/set.000/energy.npy', np.random.randn(1, 1).astype(np.float32))
np.save('${INIT_DATA_DIR}/set.000/force.npy', np.random.randn(1, 3, 3).astype(np.float32))
np.save('${INIT_DATA_DIR}/type.raw.npy', np.array([1, 1, 8], dtype=int))
np.save('${INIT_DATA_DIR}/type_map.raw', np.array(['H', 'H', 'O']))
echo '1 1 8' > ${INIT_DATA_DIR}/type.raw
echo 'H H O' > ${INIT_DATA_DIR}/type_map.raw
print('Minimal dataset created')
"
}

# Generate initial LAMMPS structure
echo "Generating initial LAMMPS configuration..."
mkdir -p "${WORK_DIR}/init_lmp"
python3 -c "
from ase.io import read, write
atoms = read('${WORK_DIR}/POSCAR')
write('${WORK_DIR}/init_lmp/conf.lmp', atoms, format='lammps-data')
"

# Mark setup complete
touch "${WORK_DIR}/setup.done"
echo ""
echo "Setup completed. Initial data prepared at:"
echo "  ${INIT_DATA_DIR}"
echo "  ${WORK_DIR}/init_lmp/conf.lmp"
echo ""
echo "Next: bash run.sh  (or ITER_NAME=000 bash 01-workflow/iter.sh)"
