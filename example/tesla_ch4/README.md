# CatFlow CH4 Dehydrogenation Validation Case

Minimal end-to-end TESLA workflow validation using Pt4 + CH4.

## System

- **Catalyst**: Pt4 tetrahedral cluster
- **Reactant**: CH4 (methane)
- **Reaction**: CH4 → CH3 + H (single C-H bond breaking)
- **CV**: C-H distance between the dissociating H and nearest C

## Workflow

```
setup.sh → iter.sh
              ├── Train:  2 DeePMD models, 200 steps each
              ├── Explore: LAMMPS MD at 2 temperatures (300K, 500K)
              ├── Screen:  model deviation filtering
              └── Label:   dummy CP2K single-point
```

## Quick Start

```bash
bash run.sh
```

This runs a single iteration: Train → Explore → Screen → Label. All compute commands are dummies (no real DeePMD/LAMMPS/CP2K needed) — they generate expected output files to validate the workflow pipeline.

## Expected Output

```
10-workdir/
  init_data/                  # DeePMD training data (npy format)
  init_lmp/conf.lmp           # LAMMPS initial structure
  iter-000/
    00.train/
      000/ frozen_model.pb, graph.pb, lcurve.out
      001/ frozen_model.pb, graph.pb, lcurve.out
    01.model_devi/
      task.0.0/ conf.lmp, explore.in, model_devi.out
      task.0.1/ conf.lmp, explore.in, model_devi.out
    02.fp/
      task.0.0/ POSCAR, input.inp, output
      task.0.1/ POSCAR, input.inp, output
      ...
```

## To Run with Real Computation

Replace the dummy run.sh files in `00-config/`:

| Stage | File | Replace dummy with |
|-------|------|-------------------|
| Train | `00-config/deepmd/run.sh` | `dp train input.json && dp freeze -o frozen_model.pb` |
| Explore | `00-config/lammps/run.sh` | `lmp -i explore.in` |
| Label | `00-config/cp2k/run.sh` | `cp2k.ssmp -i input.inp` |

Then run on an HPC with SLURM (no other changes needed).
