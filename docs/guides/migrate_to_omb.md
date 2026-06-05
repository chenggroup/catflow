# Migration Guide: dpdispatcher → oh-my-batch

CatFlow v0.6+ replaces `dpdispatcher` with [oh-my-batch](https://github.com/link89/oh-my-batch) for job submission, and optionally replaces `dpgen` with a template engine for input generation.

## Why oh-my-batch?

- **Built-in recovery**: `omb job submit --recovery` tracks job state in JSON — interrupted workflows resume where they left off
- **Declarative pipeline**: `omb combo` → `omb batch` → `omb job` replaces imperative Python submission code
- **Multi-scheduler**: SLURM / LSF / OpenPBS unified interface
- **Script-driven**: Workflow scripts run directly on HPC login nodes — no intermediate scheduler abstraction needed
- **Transparent**: Generated bash scripts are human-readable and debuggable

## What Changed

| Aspect | Before (v0.5) | After (v0.6+) |
|--------|--------------|---------------|
| Job submission | `dpdispatcher.Submission` | `omb job slurm submit` |
| Task generation | `dpgen.generator.run.make_*` | Template engine (`@VAR@` substitution) |
| Result collection | `dpgen.generator.run.post_*` | `catflow.tasker.collectors.*` |
| Machine config | `dynaconf` (YAML env-based) | Pydantic models (`MachineConfig`) |
| Script generation | N/A (dpgen built-in) | `PmfBatchScript` / `TeslaBatchScript` |
| Parallel execution | `ProcessPoolExecutor` | `WorkflowExecutor.run_script()` |
| Checkpointing | Manual state dump | `@apply_checkpoint` decorator |
| Execution model | Python submits jobs via dpdispatcher | Bash scripts run on HPC login nodes |

## New Config Format

```yaml
# machine.yml (new format)
machine:
  name: my-cluster
  scheduler: slurm
  resources:
    partition: gpu
    node_count: 1
    cpu_per_node: 4
    gpu_per_node: 1
```

Legacy `machine.yml` files with `POOL` key are still supported via automatic detection.

## Enabling the New Backend

### PMF Workflow

```bash
# No env vars needed — PMF always uses oh-my-batch now
catflow tasker pmf config.yaml

# With checkpointing for resumability
catflow tasker pmf config.yaml --checkpoint .ckpt
```

### TESLA Workflow

```bash
# Full omb + template mode (no dpgen needed)
export CATFLOW_USE_OMB=1
export CATFLOW_USE_TEMPLATE=1
catflow tasker tesla param.json machine.json

# OMB submission only, dpgen input generation
export CATFLOW_USE_OMB=1
export CATFLOW_USE_TEMPLATE=0
catflow tasker tesla param.json machine.json

# Legacy mode (dpgen for everything)
export CATFLOW_USE_OMB=0
catflow tasker tesla param.json machine.json
```

## Using Checkpoints

```bash
# PMF with checkpointing — resume after interruption
catflow tasker pmf config.yaml --checkpoint .pmf_ckpt
```

All completed tasks are cached. Re-run the same command to resume.

## Generated Scripts Example

The Python layer generates bash scripts like this:

```bash
#!/bin/bash
set -e
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=1

# Generate task directories
omb combo \
  add_var COORDINATE 1.4 1.8 2.2 \
  add_var TEMPERATURE 300 500 700 \
  add_seq SEED --start 1 --stop 5 --step 1 \
  set_broadcast SEED \
  make_files ./task.{i}/input.inp --template ./templates/input.inp \
  done

# Pack into batch scripts
omb batch \
  add_work_dirs ./task.* \
  add_headers '#SBATCH --partition=gpu' \
  add_cmd 'cp2k.ssmp -i input.inp' \
  make ./batch-{i}.slurm --concurrency 4

# Submit with recovery
omb job slurm submit ./batch-*.slurm \
  --max_tries 3 --wait --recovery pmf_recovery.json
```

## Quick Start

```bash
# 1. Install oh-my-batch
pip install oh-my-batch

# 2. Run PMF workflow with new backend
catflow tasker pmf config.yaml --checkpoint .ckpt

# 3. Run TESLA workflow without dpgen
export CATFLOW_USE_OMB=1
export CATFLOW_USE_TEMPLATE=1
catflow tasker tesla param.json machine.json

# 4. Or use the bash-driven TESLA example
cd example/tesla_omb
bash run.sh
```

## Legacy API

The old `JobFactory` (dpdispatcher) and `from dpgen.generator.run import ...` paths are still available for backward compatibility. They are deprecated and will produce deprecation warnings in future releases.
