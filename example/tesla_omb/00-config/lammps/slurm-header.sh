#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=lammps_explore
#SBATCH --output=explore_%j.log
#SBATCH --error=explore_%j.err
