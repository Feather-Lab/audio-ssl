#!/bin/bash -l
#SBATCH --job-name=eval_jsin_byola
#SBATCH --output=outLogs/eval_jsin_transfer_both_tasks_byola_%j.out
#SBATCH --error=outLogs/eval_jsin_transfer_both_tasks_byola_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64Gb
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=a100-80gb

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/mnt/home/igriffith/ceph/projects/cochdnn}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE

srun python3 lightning_scripts/eval_jsin_transfer_matched.py \
    --config_path byol-a/config.yaml \
    --gpus "$num_gpus" --num_workers "$SLURM_JOB_CPUS_PER_NODE" \
    --model_ckpt_dir model_checkpoints \
    --batch_size 4096 \
    --layer_str features.10 \
    --optimizer AdamW --lr 0.0005 \
    --task both \
    --train_epochs 3 \
    --no-with_noise --eval_only --lr_scheduler --use_classifier_ckpt --no-time_avg_rep --with_dropout
