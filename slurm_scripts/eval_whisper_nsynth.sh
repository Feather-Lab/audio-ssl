#!/bin/bash -l
#SBATCH --job-name=whisper_nsynth
#SBATCH --output=outLogs/whisper_nsynth_%A_%a.out
#SBATCH --error=outLogs/whisper_nsynth_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64Gb
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=a100-80gb

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

echo "Evaluating whisper large-v3 ln_post on NSynth family"

python3 lightning_scripts/eval_nsynth_linear.py \
    --encoder_type whisper \
    --whisper_model large-v3 \
    --layer_str 31 \
    --gpus 1 --num_workers $SLURM_JOB_CPUS_PER_NODE \
    --batch_size 256 \
    --optimizer "AdamW" --lr 0.001 \
    --task "family" \
    --train_epochs 20 \
    --lr_scheduler
