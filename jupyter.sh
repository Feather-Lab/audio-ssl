#!/bin/bash -l
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook_%j.out
#SBATCH --error=outLogs/notebook_%j.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --partition=genx


mamba activate cochdnn_ssl_pl

export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter lab --no-browser --ip='0.0.0.0' --port=1337

