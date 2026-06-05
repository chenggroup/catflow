# TESLA + oh-my-batch Example

Demonstrates the TESLA (Train-Explore-Screen-Label Active learning) workflow
using CatFlow with oh-my-batch and template-based input generation.

## Prerequisites

```bash
# Install CatFlow with oh-my-batch support
pip install oh-my-batch

# Set environment variables to enable the new code paths
export CATFLOW_USE_OMB=1
export CATFLOW_USE_TEMPLATE=1
```

## Directory Structure

```
tesla_omb/
  run.sh                     # Outer iteration loop
  00-config/                 # Configuration templates
    deepmd/                  #   DeePMD training configs
      slurm-header.sh        #     SBATCH headers
      input.json             #     Training input template
    lammps/                  #   LAMMPS exploration configs  
      slurm-header.sh        #     SBATCH headers
      plumed.in              #     PLUMED input (optional)
    cp2k/                    #   CP2K labeling configs
      slurm-header.sh        #     SBATCH headers
  01-workflow/               # Workflow scripts
    setup.sh                 #   Initial data preparation
    iter.sh                  #   Single iteration (train → explore → label)
  10-workdir/                # Working directory (generated at runtime)
```

## Usage

```bash
# 1. Prepare initial data (POSCAR/AIMD trajectories → dpdata)
bash 01-workflow/setup.sh

# 2. Run the active learning loop for 3 iterations
bash run.sh

# 3. Or run iterations individually
ITER_NAME=000 bash 01-workflow/iter.sh
ITER_NAME=001 bash 01-workflow/iter.sh
ITER_NAME=002 bash 01-workflow/iter.sh
```
