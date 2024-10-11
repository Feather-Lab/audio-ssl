#!/bin/bash -l
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook_%j.out
#SBATCH --error=outLogs/notebook_%j.err
#SBATCH --mem=2Gb
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --gpus=1

module load slurm gcc python3

conda activate ~/ceph/conda_envs/cochdnn_ssl_pl

export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter lab --no-browser --ip='0.0.0.0' --port=1338

