#!/bin/bash -l
#SBATCH --job-name=whisper_sc
#SBATCH --output=outLogs/whisper_speech_commands_%A_%a.out
#SBATCH --error=outLogs/whisper_speech_commands_%A_%a.err
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=200Gb
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=a100-80gb

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE

srun python3 lightning_scripts/eval_speech_commands_transfer.py \
    --model_type whisper \
    --whisper_model large-v3 \
    --layer_str 'ln_post' \
    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
    --model_ckpt_dir model_checkpoints \
    --batch_size 512 \
    --optimizer "AdamW" --lr 0.1 \
    --lr_scheduler \
    --train_epochs 5
