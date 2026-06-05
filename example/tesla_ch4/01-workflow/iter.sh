#!/bin/bash
set -e

# Single TESLA iteration: Train → Explore → Screen → Label
# CH4 dehydrogenation on Pt4
# Uses Python-based file generation (omb CLI not available on Python 3.13)

if [ -z "${ITER_NAME}" ]; then
    echo "ERROR: ITER_NAME is not set. Usage: ITER_NAME=000 bash iter.sh"
    exit 1
fi

cd "$(dirname "$0")/.."
CONFIG_DIR=00-config
WORK_DIR=10-workdir
ITER_DIR="${WORK_DIR}/iter-${ITER_NAME}"

mkdir -p "${ITER_DIR}"
echo "=== Iteration ${ITER_NAME}: Train → Explore → Label ==="

# ============================================================
# Step 1: Training
# ============================================================
echo ""
echo "--- Step 1: DeePMD Training ---"

TRAIN_DIR="${ITER_DIR}/00.train"
mkdir -p "${TRAIN_DIR}"

# Symlink initial data
ln -sf "$(pwd)/${WORK_DIR}/init_data" "${TRAIN_DIR}/data.init" 2>/dev/null || true

# Generate training tasks with Python (omb combo replacement)
python3 - "${TRAIN_DIR}" "${CONFIG_DIR}/deepmd/input.json" "${CONFIG_DIR}/deepmd/run.sh" << 'PYEOF'
import sys, os, json, shutil, random

train_dir = sys.argv[1]
input_tpl = sys.argv[2]
run_tpl = sys.argv[3]

num_models = 2
train_steps = 200

for m in range(num_models):
    model_dir = os.path.join(train_dir, f"{m:03d}")
    os.makedirs(model_dir, exist_ok=True)

    # Substitute @SEED@ and @STEPS@ in input.json
    seed = random.randint(0, 999999)
    with open(input_tpl) as f:
        content = f.read()
    content = content.replace("@SEED@", str(seed)).replace("@STEPS@", str(train_steps))
    with open(os.path.join(model_dir, "input.json"), "w") as f:
        f.write(content)

    # Copy run.sh
    shutil.copy(run_tpl, os.path.join(model_dir, "run.sh"))
    os.chmod(os.path.join(model_dir, "run.sh"), 0o755)
    print(f"  Generated model {m:03d} (seed={seed})")
PYEOF

echo "Generated 2 training tasks"

# Run training
for model_dir in "${TRAIN_DIR}"/{000,001}; do
    echo "  Running training in ${model_dir}..."
    (cd "${model_dir}" && bash run.sh) &
done
wait

# Verify training outputs
for model_dir in "${TRAIN_DIR}"/{000,001}; do
    ls "${model_dir}/frozen_model.pb" "${model_dir}/graph.pb" "${model_dir}/lcurve.out" 2>/dev/null
done
echo "Training complete."

# ============================================================
# Step 2: Exploration
# ============================================================
echo ""
echo "--- Step 2: LAMMPS Exploration ---"

EXPLORE_DIR="${ITER_DIR}/01.model_devi"
CONFS_DIR="${EXPLORE_DIR}/confs"
mkdir -p "${EXPLORE_DIR}" "${CONFS_DIR}"

# Copy LAMMPS conf from init data
cp "${WORK_DIR}/init_lmp/conf.lmp" "${CONFS_DIR}/conf.lmp"

# Symlink model graphs from training
for i in 0 1; do
    MODEL_DIR=$(printf "${TRAIN_DIR}/%03d" $i)
    ln -sf "$(pwd)/${MODEL_DIR}/graph.pb" "${EXPLORE_DIR}/graph.$(printf '%03d' $i).pb" 2>/dev/null || true
done

# Generate exploration tasks at 2 temperatures
python3 - "${EXPLORE_DIR}" "${CONFIG_DIR}/lammps/explore.in" "${CONFIG_DIR}/lammps/run.sh" "${CONFS_DIR}/conf.lmp" << 'PYEOF'
import sys, os, shutil, random

explore_dir = sys.argv[1]
explore_tpl = sys.argv[2]
run_tpl_arg = sys.argv[3]
conf_lmp = sys.argv[4]

temps = [300, 500]
task_idx = 0

for temp in temps:
    for rep in range(2):  # 2 replicas per temperature
        task_dir = os.path.join(explore_dir, f"task.{temp}K.{rep}")
        os.makedirs(task_dir, exist_ok=True)

        # Copy conf.lmp
        shutil.copy(conf_lmp, os.path.join(task_dir, "conf.lmp"))

        # Generate explore.in with @TEMP@ and @SEED@ substitution
        seed = random.randint(100000, 999999)
        with open(explore_tpl) as f:
            content = f.read()
        content = (content
            .replace("@TEMP@", str(temp))
            .replace("@SEED@", str(seed)))
        with open(os.path.join(task_dir, "explore.in"), "w") as f:
            f.write(content)

        # Copy run.sh
        shutil.copy(run_tpl_arg, os.path.join(task_dir, "run.sh"))
        os.chmod(os.path.join(task_dir, "run.sh"), 0o755)

        print(f"  Generated {task_dir} (T={temp}K, seed={seed})")
        task_idx += 1
PYEOF

echo "Generated exploration tasks"
ls -d "${EXPLORE_DIR}"/task.*/

# Run exploration
for task_dir in "${EXPLORE_DIR}"/task.*/; do
    echo "  Running explore in ${task_dir}..."
    (cd "${task_dir}" && bash run.sh) &
done
wait
echo "Exploration complete."

# ============================================================
# Step 3: Screening
# ============================================================
echo ""
echo "--- Step 3: Screening ---"

LABEL_DIR="${ITER_DIR}/02.fp"
mkdir -p "${LABEL_DIR}"

python3 - "${EXPLORE_DIR}" "${LABEL_DIR}" "${CONFIG_DIR}" << 'PYEOF'
import sys, os, json, glob, shutil
import numpy as np

explore_dir = sys.argv[1]
label_dir = sys.argv[2]
config_dir = sys.argv[3]

candidates = []
task_idx = 0
for task_dir in sorted(glob.glob(f"{explore_dir}/task.*/")):
    mdf = os.path.join(task_dir, "model_devi.out")
    if not os.path.exists(mdf):
        print(f"  WARNING: model_devi.out not found in {task_dir}")
        continue

    data = np.loadtxt(mdf)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    task_name = os.path.basename(task_dir.rstrip('/'))
    for frame_idx, row in enumerate(data):
        if frame_idx < 1:
            continue
        # Column 3 = max_devi_f (4th column)
        max_devi_f = row[3] if row.shape[0] > 3 else row[-1]

        if 0.1 < max_devi_f < 0.3:
            cand_dir = os.path.join(label_dir, f"task.{task_idx:03d}")
            os.makedirs(cand_dir, exist_ok=True)

            # Minimal VASP POSCAR
            with open(os.path.join(cand_dir, "POSCAR"), "w") as f:
                f.write(f"Pt4_CH4_candidate_{task_idx}\n")
                f.write("1.0\n")
                f.write("12.0 0.0 0.0\n")
                f.write("0.0 12.0 0.0\n")
                f.write("0.0 0.0 12.0\n")
                f.write("Pt C H\n")
                f.write("4 1 4\n")
                f.write("Direct\n")
                pos = [
                    (0.35,0.35,0.50), (0.35,0.50,0.35), (0.50,0.35,0.35), (0.42,0.42,0.42),
                    (0.50,0.50,0.65), (0.53,0.47,0.68), (0.47,0.53,0.68),
                    (0.50,0.50,0.60), (0.50,0.50,0.63),
                ]
                for p in pos:
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

            # CP2K input
            tpl_path = os.path.join(config_dir, "cp2k", "input.inp.template")
            if os.path.exists(tpl_path):
                with open(tpl_path) as src:
                    inp = src.read()
                # Write XYZ coordinates
                coord_lines = ["9"]
                coord_lines.append(f"Pt4_CH4 candidate {task_idx}")
                for i, p in enumerate(pos):
                    el = "Pt" if i < 4 else ("C" if i == 4 else "H")
                    x = p[0] * 12.0
                    y = p[1] * 12.0
                    z = p[2] * 12.0
                    coord_lines.append(f"{el} {x:.6f} {y:.6f} {z:.6f}")
                with open(os.path.join(cand_dir, "coord.xyz"), "w") as f:
                    f.write("\n".join(coord_lines))

                coord_str = "\n".join("    " + line for line in coord_lines)
                inp = inp.replace("@COORD_CONTENT@", coord_str)
                with open(os.path.join(cand_dir, "input.inp"), "w") as f:
                    f.write(inp)

            # Copy dummy run.sh
            shutil.copy(os.path.join(config_dir, "cp2k", "run.sh"),
                        os.path.join(cand_dir, "run.sh"))
            os.chmod(os.path.join(cand_dir, "run.sh"), 0o755)

            candidates.append(f"{task_name} frame_{frame_idx} max_devi_f={max_devi_f:.3f}")
            task_idx += 1
            if task_idx >= 4:
                break
    if task_idx >= 4:
        break

print(f"Selected {task_idx} candidates for labeling")
for c in candidates:
    print(f"  {c}")
PYEOF

# ============================================================
# Step 4: Labeling
# ============================================================
echo ""
echo "--- Step 4: Labeling ---"

for task_dir in "${LABEL_DIR}"/task.*/; do
    echo "  Running label in ${task_dir}..."
    (cd "${task_dir}" && bash run.sh) &
done
wait

LABEL_COUNT=$(ls -d "${LABEL_DIR}"/task.*/ 2>/dev/null | wc -l)
echo "Labeling complete: ${LABEL_COUNT} tasks"

# ============================================================
# Done
# ============================================================
touch "${ITER_DIR}/iter.done"
echo ""
echo "=== Iteration ${ITER_NAME} complete ==="
