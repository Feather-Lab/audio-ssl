#!/bin/bash -l
#SBATCH --job-name=audiomae_zeroshot
#SBATCH --output=outLogs/audiomae_zeroshot_%A_%a.out
#SBATCH --error=outLogs/audiomae_zeroshot_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32Gb
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --array=1-2

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

echo "Job array ID: $SLURM_ARRAY_TASK_ID"

python3 lightning_scripts/eval_audiomae_layerwise_zero_shot.py --out_dir results_dfs
