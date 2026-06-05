# oh-my-batch Workflow

CatFlow v0.6+ uses [oh-my-batch](https://github.com/link89/oh-my-batch) (`omb`) as its job submission engine, replacing the previous `dpdispatcher` backend. This page documents the new workflow architecture.

## Architecture Overview

The refactored CatFlow uses a **three-layer architecture**:

```
┌─────────────────────────────────────────────────┐
│                   Python Layer                    │
│  Config (Pydantic) → Flow Logic → Result Collect  │
│                                                   │
│  PMF: asyncio orchestration + convergence checks  │
│  TESLA: iteration loop + parameter updates        │
└──────────────────────┬──────────────────────────┘
                       │ generates
┌──────────────────────▼──────────────────────────┐
│              Script Generation Layer              │
│  PmfBatchScript / TeslaBatchScript / ScriptBuilder│
│                                                   │
│  Generates bash scripts with omb combo/batch/job  │
└──────────────────────┬──────────────────────────┘
                       │ executes on
┌──────────────────────▼──────────────────────────┐
│              Execution Layer                      │
│  WorkflowExecutor (local via subprocess)          │
│  WorkflowExecutor (remote via SSH/scp)            │
│                                                   │
│  Scripts run directly on HPC login nodes           │
└─────────────────────────────────────────────────┘
```

## The `omb combo → batch → job` Pipeline

Every computational stage follows this pattern:

```bash
# 1. COMBO: Generate parameter combinations and task directories
omb combo \
  add_var COORDINATE 1.4 1.8 2.2 \
  add_var TEMPERATURE 300 500 700 \
  add_randint SEED -n 3 -a 0 -b 999999 \
  set_broadcast SEED \
  make_files ./task.{i}/input.inp --template ./templates/input.inp \
  done

# 2. BATCH: Pack task directories into SLURM batch scripts
omb batch \
  add_work_dirs ./task.* \
  add_header ./templates/slurm-header.sh \
  add_cmd 'cp2k.ssmp -i input.inp' \
  make ./batch-{i}.slurm --concurrency 4

# 3. JOB: Submit, monitor, and recover on failure
omb job slurm submit ./batch-*.slurm \
  --max_tries 3 --wait --recovery recovery.json
```

## Template-Based Input Generation

When `CATFLOW_USE_TEMPLATE=1`, CatFlow uses `@VAR@` template files instead of `dpgen` to generate computation inputs:

| Template | Purpose | Key Variables |
|----------|---------|---------------|
| `templates/deepmd/input.json.template` | DeePMD-kit training input | `@SEED@`, `@STEPS@`, `@TYPE_MAP@`, `@SEL@`, `@DP_DATASET@` |
| `templates/deepmd/run.sh.template` | DeePMD training execution | `@SEED@`, `@STEPS@` |
| `templates/lammps/explore.in.template` | LAMMPS MD input | `@TEMP@`, `@SEED@`, `@DP_MODELS@`, `@MASS_MAP@` |
| `templates/lammps/run.sh.template` | LAMMPS execution | (static, no variables needed) |
| `templates/cp2k/fp.inp.template` | CP2K DFT input | `@PROJECT@`, `@CUTOFF@`, `@XC_FUNCTIONAL@` |

## Operation Mode Matrix

CatFlow supports progressive adoption via environment variables:

| `CATFLOW_USE_OMB` | `CATFLOW_USE_TEMPLATE` | Behavior | dpgen required? |
|:---:|:---:|---|---|
| `0` | `0` | Full dpgen backward compatibility (default) | ✅ Required |
| `1` | `0` | dpgen generates inputs, omb submits jobs | ✅ Required |
| `0` | `1` | Templates generate inputs, dpgen submits | ✅ Required |
| `1` | `1` | **Full omb + template mode** | ❌ Not needed |

The recommended `CATFLOW_USE_OMB=1` + `CATFLOW_USE_TEMPLATE=1` mode provides:
- **Recovery**: `omb job submit --recovery` tracks job state; interrupted workflows resume from where they left off
- **No dpgen dependency**: Templates and collectors replace all dpgen functionality
- **Transparent scripts**: Generated bash scripts are human-readable and debuggable

## Workflow Executor

The `WorkflowExecutor` runs generated scripts either locally or remotely:

```python
from catflow.tasker.resources.workflow_executor import WorkflowExecutor

# Local execution (on HPC login node)
executor = WorkflowExecutor.local(work_base="/path/to/workdir")
executor.run_script("generated_script.sh")

# SSH execution (from submission host to HPC)
executor = WorkflowExecutor.ssh(
    host="login.hpc.org",
    user="username",
    remote_work_base="/remote/path",
)
executor.run_script("remote_script.sh")
executor.download("/remote/path/results", "./local_results")
```

## Checkpointing

Function-level checkpointing using `@apply_checkpoint` from ai2-kit:

```python
from catflow.core import apply_checkpoint, set_checkpoint_dir

set_checkpoint_dir("./.ckpt")

@apply_checkpoint(lambda coordinate, temperature: f"pmf_{coordinate}_{temperature}")
async def task_pmf_calculation(coordinate, temperature, ...):
    # ... computation ...
    return result
```

Completed tasks are cached and skipped on resume. Re-run the same command to continue from the last checkpoint.

## PMF Workflow

The PMF (Potential of Mean Force) workflow generates bash scripts for each (coordinate, temperature) pair and submits them via oh-my-batch:

```bash
catflow tasker pmf config.yaml --checkpoint .ckpt
```

Key changes from the old dpdispatcher backend:
- `ProcessPoolExecutor` → `asyncio.to_thread(executor.run_script, ...)`
- `dpdispatcher.Submission` → `omb job slurm submit --wait --recovery`
- Scripts generated by `PmfBatchScript` → stored alongside task directories

## TESLA Workflow

The TESLA active learning workflow supports the same three operation modes:

```bash
# Full omb + template mode (recommended)
export CATFLOW_USE_OMB=1
export CATFLOW_USE_TEMPLATE=1
catflow tasker tesla param.json machine.json

# Legacy dpgen mode (default)
catflow tasker tesla param.json machine.json
```

### TESLA Directory Structure

```
iter.000000/
  00.train/                    # DeePMD training
    000/ input.json, run.sh, frozen_model.pb, graph.pb
    001/ ...
    ...
  01.model_devi/               # LAMMPS exploration
    task.0.0/ conf.lmp, input.lammps, model_devi.out, run.sh
    task.0.1/ ...
    ...
  02.fp/                       # DFT labeling
    task.0.0/ POSCAR, input.inp, output, run.sh
    ...
```

Each iteration generates an `iter.NNNNNN/` directory with the same structure.
