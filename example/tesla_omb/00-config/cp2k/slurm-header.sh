#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=cp2k_label
#SBATCH --output=label_%j.log
#SBATCH --error=label_%j.err
