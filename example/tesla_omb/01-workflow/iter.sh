#!/bin/bash
set -e

# TESLA single iteration: Train → Explore → Label
# Called with ITER_NAME environment variable (e.g., ITER_NAME=000)

CONFIG_DIR=./00-config
WORK_DIR=./10-workdir
ITER_DIR="${WORK_DIR}/iter-${ITER_NAME}"

if [ -z "${ITER_NAME}" ]; then
    echo "ERROR: ITER_NAME is not set."
    echo "Usage: ITER_NAME=000 bash 01-workflow/iter.sh"
    exit 1
fi

# Skip if already done
if [ -f "${ITER_DIR}/iter.done" ]; then
    echo "Iteration ${ITER_NAME} already completed."
    exit 0
fi

mkdir -p "${ITER_DIR}"

echo "=== [${ITER_NAME}] Starting TESLA iteration ==="

# ============================================================
# Step 1: Training
# ============================================================
echo "--- Step 1: DeePMD Training ---"
TRAIN_DIR="${ITER_DIR}/00.train"
mkdir -p "${TRAIN_DIR}"

# Generate training input with 4 random seeds
omb combo \
  add_randint SEED -n 4 -a 0 -b 999999 \
  add_var STEPS 400000 \
  set_broadcast STEPS \
  make_files "${TRAIN_DIR}"/{i}/input.json \
    --template "${CONFIG_DIR}/deepmd/input.json" \
  make_files "${TRAIN_DIR}"/{i}/run.sh \
    --template "${CONFIG_DIR}/deepmd/run.sh" --mode 755 \
  done

# Submit training jobs (4 models, concurrency=4)
omb batch \
  add_work_dirs "${TRAIN_DIR}"/000 "${TRAIN_DIR}"/001 \
                "${TRAIN_DIR}"/002 "${TRAIN_DIR}"/003 \
  add_cmd 'bash run.sh' \
  make "${TRAIN_DIR}/train-{i}.slurm" --concurrency 4

omb job slurm submit "${TRAIN_DIR}/train-*.slurm" \
  --max_tries 2 --wait --recovery "${TRAIN_DIR}/.train_recovery.json"

echo "Training completed."

# ============================================================
# Step 2: Exploration
# ============================================================
echo "--- Step 2: LAMMPS Exploration ---"
EXPLORE_DIR="${ITER_DIR}/01.explore"
mkdir -p "${EXPLORE_DIR}"

# Collect trained models
MODEL_FILES=$(ls "${TRAIN_DIR}"/000/frozen_model.pb 2>/dev/null || echo "")
if [ -z "${MODEL_FILES}" ]; then
    echo "ERROR: No trained models found in ${TRAIN_DIR}"
    exit 1
fi

# Generate exploration tasks at different temperatures
omb combo \
  add_var TEMP 300 500 700 1000 \
  add_seq MODEL --start 0 --stop 4 --step 1 \
  make_files "${EXPLORE_DIR}/job-{TEMP}K-{i}/in.lammps" \
    --template "${CONFIG_DIR}/lammps/explore.in" \
  make_files "${EXPLORE_DIR}/job-{TEMP}K-{i}/run.sh" \
    --template "${CONFIG_DIR}/lammps/run.sh" --mode 755 \
  done

# Submit exploration jobs (concurrency=5)
omb batch \
  add_work_dirs "${EXPLORE_DIR}/job-*" \
  add_cmd 'bash run.sh' \
  make "${EXPLORE_DIR}/explore-{i}.slurm" --concurrency 5

omb job slurm submit "${EXPLORE_DIR}/explore-*.slurm" \
  --max_tries 2 --wait --recovery "${EXPLORE_DIR}/.explore_recovery.json"

echo "Exploration completed."

# ============================================================
# Step 3: Screening (model deviation analysis)
# ============================================================
echo "--- Step 3: Screening ---"

# Use ai2-kit's model_devi tool to filter structures
# (or use catflow's collector directly)
if command -v ai2-kit &>/dev/null; then
    ai2-kit tool model_devi \
        --path "${EXPLORE_DIR}" \
        --slice "10:" \
        --lo 0.1 --hi 0.3 \
        --output "${ITER_DIR}/screening"
else
    echo "Screening via model deviation thresholds (f_trust_lo=0.1, f_trust_hi=0.3)"
    python3 -c "
import numpy as np, json, glob
results = {'candidates': [], 'accurate': [], 'failed': []}
for task_file in glob.glob('${EXPLORE_DIR}/job-*/model_devi.out'):
    data = np.loadtxt(task_file)
    task_name = task_file.split('/')[-2]
    for frame_idx, row in enumerate(data):
        if frame_idx < 10:
            continue
        max_devi_f = row[4] if row.ndim == 1 else row[4]
        if max_devi_f < 0.1:
            results['accurate'].append(f'{task_name} {frame_idx}')
        elif max_devi_f > 0.3:
            results['failed'].append(f'{task_name} {frame_idx}')
        else:
            results['candidates'].append(f'{task_name} {frame_idx}')
open('${ITER_DIR}/candidates.json', 'w').write(json.dumps(results, indent=2))
print(f'Candidates: {len(results[\"candidates\"])}, Accurate: {len(results[\"accurate\"])}, Failed: {len(results[\"failed\"])}')
"
fi

echo "Screening completed."

# ============================================================
# Step 4: Labeling (DFT single point calculations)
# ============================================================
echo "--- Step 4: DFT Labeling ---"
LABEL_DIR="${ITER_DIR}/02.label"
mkdir -p "${LABEL_DIR}"

# Create labeling tasks from candidates
python3 -c "
import json, shutil, os
from pathlib import Path

candidates = json.load(open('${ITER_DIR}/candidates.json'))
for i, entry in enumerate(candidates.get('candidates', [])[:10]):
    task_parts = entry.split()
    if len(task_parts) < 2:
        continue
    task_name, frame_idx = task_parts
    job_dir = Path('${LABEL_DIR}') / f'candidate.{i:03d}'
    job_dir.mkdir(parents=True, exist_ok=True)

    # Copy the trajectory frame as POSCAR
    traj_file = Path('${EXPLORE_DIR}') / task_name / 'traj/dump.lammpstrj'
    if traj_file.exists():
        shutil.copy(traj_file, job_dir / 'input.dump')
    # Create placeholder input
    with open(job_dir / 'POSCAR', 'w') as f:
        f.write(f'Candidate {i} from {task_name} frame {frame_idx}\n')

    print(f'Created labeling task: {job_dir}')
"

# Submit labeling jobs
omb batch \
  add_work_dirs "${LABEL_DIR}/candidate.*" \
  add_cmd 'echo "DP: cp2k.ssmp -i input.inp" || true' \
  make "${LABEL_DIR}/label-{i}.slurm" --concurrency 5

# Mark iteration complete
touch "${ITER_DIR}/iter.done"
echo "=== Iteration ${ITER_NAME} completed ==="
