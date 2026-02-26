#!/bin/bash
#SBATCH --job-name=training_seso
#SBATCH --output=jobs/job%j_%x.out
#SBATCH --error=jobs/job%j_%x.err
#SBATCH --nodes=1
#SBATCH --partition=GPUs
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=augustopmello@usp.br
#SBATCH --cpus-per-task=8
# train.sh

srun -N1 singularity exec --nv container.sif python3 bert_example-checkpoint.py