#!/bin/bash
set -e

# Generate initial structures and training data for Pt4 + CH4
cd "$(dirname "$0")/.."

WORK_DIR=10-workdir
INIT_DATA_DIR="${WORK_DIR}/init_data"
mkdir -p "${WORK_DIR}" "${INIT_DATA_DIR}"

# Skip if already done
if [ -f "${WORK_DIR}/setup.done" ]; then
    echo "Setup already completed. Remove ${WORK_DIR}/setup.done to redo."
    exit 0
fi

echo "Generating initial structures for Pt4 + CH4..."

python3 - "${WORK_DIR}" "${INIT_DATA_DIR}" << 'PYEOF'
import sys, os, numpy as np
from ase.io import write
from ase import Atoms

work_dir = sys.argv[1]
init_data_dir = sys.argv[2]
os.makedirs(init_data_dir, exist_ok=True)

# --- Build Pt4 tetrahedral cluster ---
# Tetrahedron: center at (5,5,5), edge ~2.77 Å (Pt-Pt distance)
edge = 2.77
# Tetrahedron vertices
h = edge * np.sqrt(2/3)
v0 = np.array([5.0,    5.0,    5.0 + h/2])
v1 = np.array([5.0 + edge/np.sqrt(3), 5.0, 5.0 - h/2 + edge/(2*np.sqrt(3))])
v2 = np.array([5.0 - edge/(2*np.sqrt(3)), 5.0 + edge/2, 5.0 - h/2 - edge/(4*np.sqrt(3))])
v3 = np.array([5.0 - edge/(2*np.sqrt(3)), 5.0 - edge/2, 5.0 - h/2 - edge/(4*np.sqrt(3))])

# --- Place CH4 near Pt4 ---
# CH4: tetrahedral, C at center + 4 H
c_pos = np.array([5.0, 5.0 + 3.0, 5.0 + h/2 + 1.5])
ch_bond = 1.09
h_vecs = [
    c_pos + np.array([ch_bond, 0, 0]),
    c_pos + np.array([-ch_bond, 0, 0]),
    c_pos + np.array([0, ch_bond, 0]),
    c_pos + np.array([0, -ch_bond, 0]),
]

pos = np.vstack([v0, v1, v2, v3, c_pos, *h_vecs])
symbols = ['Pt']*4 + ['C'] + ['H']*4
atoms = Atoms(symbols=symbols, positions=pos)
atoms.set_cell([12.0, 12.0, 12.0])
atoms.center()
atoms.set_pbc([True, True, True])

write(f'{work_dir}/POSCAR', atoms, vasp5=True)
write(f'{work_dir}/init.xyz', atoms)

# --- Generate perturbed frames for initial dataset ---
print("Generating 10 perturbed frames for initial training set...")
np.random.seed(42)
frames = []
n_atoms = len(atoms)
for i in range(10):
    a = atoms.copy()
    a.positions += np.random.randn(n_atoms, 3) * 0.05
    frames.append(a)
write(f'{work_dir}/trajectory.xyz', frames)

# --- Create minimal DeePMD npy dataset ---
set_dir = os.path.join(init_data_dir, 'set.000')
os.makedirs(set_dir, exist_ok=True)

# Generate random coordinates, energies, forces for 10 frames
n_frames = 10
natom = 9  # Pt4 + C + 4H
coord = np.random.randn(n_frames, natom * 3).astype(np.float32)
energy = np.random.randn(n_frames, 1).astype(np.float32)
force = np.random.randn(n_frames, natom, 3).astype(np.float32)

np.save(os.path.join(set_dir, 'coord.npy'), coord)
np.save(os.path.join(set_dir, 'energy.npy'), energy)
np.save(os.path.join(set_dir, 'force.npy'), force)

# type_map: Pt=0, C=1, H=2
type_raw = np.array([0]*4 + [1] + [2]*4, dtype=int)
np.savetxt(os.path.join(init_data_dir, 'type.raw'), type_raw, fmt='%d')
with open(os.path.join(init_data_dir, 'type_map.raw'), 'w') as f:
    f.write('Pt\nC\nH\n')

print(f"Created {n_frames} frames at {set_dir}")

# --- Create LAMMPS conf.lmp ---
lmp_dir = os.path.join(work_dir, 'init_lmp')
os.makedirs(lmp_dir, exist_ok=True)
write(os.path.join(lmp_dir, 'conf.lmp'), atoms, format='lammps-data')
print(f"Created LAMMPS config at {lmp_dir}/conf.lmp")
PYEOF

# Mark done
touch "${WORK_DIR}/setup.done"
echo ""
echo "Setup complete!"
echo "  Structures: ${WORK_DIR}/POSCAR, ${WORK_DIR}/init.xyz"
echo "  Training data: ${INIT_DATA_DIR}/"
echo "  LAMMPS config: ${WORK_DIR}/init_lmp/conf.lmp"
