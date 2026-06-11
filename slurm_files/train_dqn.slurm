#!/bin/bash

# Job Flags
#SBATCH -p pg_tata
#SBATCH -G l40s:1
#SBATCH -t 00:15:00
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=keanunakamura@berkeley.edu
#SBATCH --mail-type=END,FAIL

# Set up environment
module load miniforge
source activate rlsim-env

# Run your application
rlsim rl-train -c 'configs/train_dqn.toml'
