#!/bin/bash -l
#SBATCH --job-name=audiomae_esc50
#SBATCH --output=outLogs/audiomae_esc50_%A_%a.out
#SBATCH --error=outLogs/audiomae_esc50_%A_%a.err
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=200Gb
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --array=12

mamba activate cochdnn_ssl_pl

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

echo "Job array ID: $SLURM_ARRAY_TASK_ID  (layers 0-11 = ViT blocks, 12 = norm)"

python3 lightning_scripts/eval_audiomae_esc50.py \
    -D "${COCHDNN_SCRATCH_DIR:-/tmp/cochdnn}" \
    -L $SLURM_ARRAY_TASK_ID \
    -A 4096 -R 5 -P -O \
    -C 0.01 0.1 1 10 100
