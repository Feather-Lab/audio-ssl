#!/bin/bash -l
#SBATCH --job-name=eval_jsin
#SBATCH --output=outLogs/eval_jsin_transfer_both_tasks_%A_%a.out
#SBATCH --error=outLogs/eval_jsin_transfer_both_tasks_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64Gb
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=a100-80gb
#SBATCH --array=0-11

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/mnt/home/igriffith/ceph/projects/cochdnn}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE

config_list_path="train_config_manifests/cochdnn9_sup_and_ssl_eval_configs.pkl"
model_ckpt_dir="model_checkpoints"

lr=0.01
# Keep LR deterministic: scaled models (manifest ix 6-8) use 0.0005, all others use 0.01.
if [ "$SLURM_ARRAY_TASK_ID" -ge 6 ] && [ "$SLURM_ARRAY_TASK_ID" -le 8 ]; then
    lr=0.0005
fi

supervised_flag=()
if [ "$SLURM_ARRAY_TASK_ID" -ge 8 ]; then
    supervised_flag=(--supervised_backbone)
fi

srun python3 lightning_scripts/eval_jsin_transfer_matched.py \
    --config_list_path "$config_list_path" \
    --array_ix "$SLURM_ARRAY_TASK_ID" \
    --gpus "$num_gpus" --num_workers "$SLURM_JOB_CPUS_PER_NODE" \
    --model_ckpt_dir "$model_ckpt_dir" \
    --batch_size 192 \
    --layer_str relu4 \
    --optimizer AdamW --lr "$lr" \
    --task both \
    --train_epochs 6 \
    "${supervised_flag[@]}" \
    --no-with_noise --eval_only --lr_scheduler --use_classifier_ckpt --no-time_avg_rep --with_dropout