# Getting Started with oh-my-batch

This guide walks through setting up and running a TESLA active learning workflow using CatFlow's oh-my-batch + template backend.

## Prerequisites

```bash
# Install CatFlow with dependencies
pip install oh-my-batch catflow

# Verify installation
python -c "import oh_my_batch; print(f'oh-my-batch v{oh_my_batch.__version__}')"
omb --help
```

## 1. Simple PMF Calculation

The PMF (Potential of Mean Force) workflow computes free energy profiles along a reaction coordinate at multiple temperatures.

### Configuration

```yaml
# pmf_config.yaml
job_config:
  work_path: "./pmf_work"
  command: "cp2k.ssmp -i input.inp"
  machine_name: "local"
  resources:
    partition: "gpu"
    node_count: 1
    cpu_per_node: 4
    gpu_per_node: 1
  reaction_pair: [0, 1]
  steps: 1000000
  timestep: 0.5

flow_config:
  coordinates: [1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8]
  t_min: 300
  t_max: 1000
  t_step: 100
  melting_test: true
  is_coordinate: 1.4
  fs_coordinate: 3.8
  init_artifact:
    - coordinate: 1.4
      structure_path: "./init_IS.xyz"
    - coordinate: 3.8
      structure_path: "./init_FS.xyz"
```

### Run

```bash
catflow tasker pmf pmf_config.yaml --checkpoint .pmf_ckpt
```

The `--checkpoint` flag enables job-level checkpointing. If the workflow is interrupted, re-run the same command to resume from the last completed task.

### What Happens

1. Python parses config → determines temperature range (via melting test)
2. For each (coordinate, temperature) pair, generates a bash script with `PmfBatchScript`
3. `WorkflowExecutor` executes the bash script on the HPC login node
4. The script runs `omb combo` → `omb batch` → `omb job submit --wait` to submit and monitor jobs
5. Results are collected and convergence is checked
6. PMF profile is plotted at the end

## 2. TESLA Active Learning Workflow

The TESLA workflow iteratively trains MLPs through Train → Explore → Screen → Label cycles.

### Using the Python API

```bash
# Enable full omb + template mode
export CATFLOW_USE_OMB=1
export CATFLOW_USE_TEMPLATE=1

# Run TESLA workflow
catflow tasker tesla param.json machine.json
```

### Using the Bash-Driven Example

For maximum transparency and control, use the bash-driven example:

```bash
cd example/tesla_omb

# 1. Inspect configuration templates
ls 00-config/deepmd/     # DeePMD training config + slurm header
ls 00-config/lammps/     # LAMMPS exploration config + slurm header

# 2. Setup initial data (generates test structures)
bash 01-workflow/setup.sh

# 3. Run 3 active learning iterations
bash run.sh
```

The bash-driven workflow generates all files in `10-workdir/`:

```
10-workdir/
  setup.done
  init.xyz                     # Initial structures
  init_data/                   # DeePMD training dataset
  iter-000/
    iter.done
    00.train/                  # DeePMD models (4 per iteration)
    01.explore/                # LAMMPS trajectories
    02.label/                  # DFT labeling tasks
  iter-001/
  iter-002/
```

### Understanding the Iteration Script

The core iteration script (`01-workflow/iter.sh`) demonstrates the `omb combo → batch → job` pipeline:

```bash
# === TRAINING ===
# Generate 4 DeePMD models with different random seeds
omb combo add_randint SEED -n 4 -a 0 -b 999999 \
  make_files ./10-workdir/iter-000/00.train/{i}/input.json \
    --template ./00-config/deepmd/input.json \
  done

# Submit all 4 models (concurrency=4 → all run simultaneously)
omb batch add_work_dirs ... add_cmd 'bash run.sh' \
  make ./train-{i}.slurm --concurrency 4
omb job slurm submit ./train-*.slurm --max_tries 2 --wait

# === EXPLORATION ===
# Run MD at multiple temperatures
omb combo add_var TEMP 300 500 700 1000 \
  make_files ./explore/job-{TEMP}K-{i}/in.lammps --template ... \
  done

# Submit with recovery
omb job slurm submit ./explore-*.slurm --max_tries 2 --wait \
  --recovery ./explore_recovery.json
```

## 3. Recovery and Resume

### Script-Level Recovery (oh-my-batch)

oh-my-batch's `--recovery` flag saves job state to a JSON file:

```bash
# First run — jobs submitted, tracked in recovery.json
omb job slurm submit ./batch-*.slurm --wait --recovery recovery.json

# If interrupted: re-run the SAME command
# omb reads recovery.json, skips completed jobs, only submits unfinished ones
omb job slurm submit ./batch-*.slurm --wait --recovery recovery.json
```

### Function-Level Checkpointing (ai2-kit)

The `@apply_checkpoint` decorator from `catflow.core` caches function return values:

```python
from catflow.core import apply_checkpoint, set_checkpoint_dir

set_checkpoint_dir("./.ckpt")

@apply_checkpoint(lambda coordinate, temperature: f"pmf_{coordinate}_{temperature}")
async def expensive_calculation(coordinate, temperature):
    ...
    return result
```

On resume, the cached result is returned without re-execution.

### Workflow-Level Recovery

```bash
# PMF: checkpoint directory stores all completed task results
catflow tasker pmf config.yaml --checkpoint .pmf_ckpt

# Interrupt with Ctrl+C, then resume:
catflow tasker pmf config.yaml --checkpoint .pmf_ckpt
# Only unfinished tasks will be submitted
```

## 4. Template Customization

To customize input generation, edit the templates in `templates/` or point to custom templates:

```bash
export CATFLOW_TEMPLATE_DIR="./my_templates"
```

Available templates:

| Template | Required Variables |
|----------|-------------------|
| `deepmd/input.json.template` | `@SEED@`, `@STEPS@`, `@TYPE_MAP@`, `@SEL@`, `@DP_DATASET@` |
| `lammps/explore.in.template` | `@TEMP@`, `@SEED@`, `@DP_MODELS@`, `@MASS_MAP@`, `@DT@` |
| `cp2k/fp.inp.template` | `@PROJECT@`, `@BASIS_FILE@`, `@POTENTIAL_FILE@`, `@CUTOFF@` |

## 5. Troubleshooting

**"dpgen is not installed" error**: Set both env vars to use the template + omb path:
```bash
export CATFLOW_USE_OMB=1
export CATFLOW_USE_TEMPLATE=1
```

**Template not found**: Ensure the `templates/` directory exists relative to the working directory, or set `CATFLOW_TEMPLATE_DIR`:
```bash
export CATFLOW_TEMPLATE_DIR="/path/to/catflow/templates"
```

**SSH executor fails**: Check SSH key authentication and host accessibility:
```python
from catflow.tasker.resources.workflow_executor import WorkflowExecutor
executor = WorkflowExecutor.ssh(host="hpc.edu", user="user")
result = executor.run_command("hostname")
print(result.stdout)  # Should print the remote hostname
```
