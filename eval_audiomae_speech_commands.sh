#!/bin/bash -l
#SBATCH --job-name=audiomae_sc
#SBATCH --output=outLogs/audiomae_speech_commands_%A_%a.out
#SBATCH --error=outLogs/audiomae_speech_commands_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64Gb
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --array=12

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn

echo "Job array ID: $SLURM_ARRAY_TASK_ID  (layers 0-11 = ViT blocks, 12 = norm)"

python3 lightning_scripts/eval_audiomae_speech_commands.py \
    --layer_idx $SLURM_ARRAY_TASK_ID \
    --gpus 1 --num_workers $SLURM_JOB_CPUS_PER_NODE \
    --batch_size 256 \
    --optimizer "AdamW" --lr 0.001 \
    --train_epochs 5 \
    --no-eval_only --lr_scheduler
