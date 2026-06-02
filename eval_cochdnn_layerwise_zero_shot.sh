#!/bin/bash -l
#SBATCH --job-name=cochdnn_zeroshot
#SBATCH --output=outLogs/cochdnn_zeroshot_%A_%a.out
#SBATCH --error=outLogs/cochdnn_zeroshot_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32Gb
#SBATCH --time=02:00:00 # should take about 20 minutes, extra for overhead
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --array=0-23

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn

echo "Job array ID: $SLURM_ARRAY_TASK_ID"

python3 lightning_scripts/eval_layerwise_zero_shot.py \
    --model_type cochdnn \
    --out_dir results_dfs
