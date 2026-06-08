#!/bin/bash -l
#SBATCH --job-name=byola_zeroshot
#SBATCH --output=outLogs/byola_zeroshot_%A_%a.out
#SBATCH --error=outLogs/byola_zeroshot_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32Gb
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --array=0-2

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/mnt/home/igriffith/ceph/projects/cochdnn}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

echo "Job array ID: $SLURM_ARRAY_TASK_ID"

python3 lightning_scripts/eval_layerwise_zero_shot.py \
    --model_type byola \
    --out_dir results_dfs
