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

LAYERS=(block_0 block_1 block_2 block_3 block_4 block_5 block_6 block_7 block_8 block_9 block_10 block_11 norm)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

echo "Job array ID: $SLURM_ARRAY_TASK_ID  Layer: $LAYER"

python3 lightning_scripts/eval_speech_commands_transfer.py \
    --model_type audiomae \
    --layer_str $LAYER \
    --gpus 1 --num_workers $SLURM_JOB_CPUS_PER_NODE \
    --batch_size 256 \
    --optimizer "AdamW" --lr 0.1 \
    --train_epochs 50 \
    --no-eval_only --lr_scheduler
