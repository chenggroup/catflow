#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=deepmd_train
#SBATCH --output=train_%j.log
#SBATCH --error=train_%j.err
