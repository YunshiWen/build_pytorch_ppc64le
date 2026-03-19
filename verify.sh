#!/bin/bash
#SBATCH --job-name=torch         # create a short name for your job
#SBATCH -n 1
#SBATCH --partition=dcs-2024     # appropriate partition; if not specified, slurm will automatically do it for you
#SBATCH --gres=gpu:1             # number of allocated gpus per node
#SBATCH --output=./dcs/torch_build.log  # Standard output and error log
#SBATCH --time=06:00:00          # total run time limit (HH:MM:SS)


eval "$(/gpfs/u/home/AFMD/AFMDhnns/scratch/miniconda3/bin/conda shell.bash hook)" 
conda activate torch260

python dcs/verify_pytorch.py