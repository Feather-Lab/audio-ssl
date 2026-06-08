#!/bin/bash -l
#SBATCH --job-name=eval_nsynth_kell2018
#SBATCH --output=outLogs/eval_nsynth_kell2018_%A_%a.out
#SBATCH --error=outLogs/eval_nsynth_kell2018_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64Gb
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=a100-80gb
#SBATCH --array=0-11 # 0-11 in manifest

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/mnt/ceph/users/igriffith/projects/cochdnn}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export COCHDNN_NSYNTH_DIR="${COCHDNN_NSYNTH_DIR:-/mnt/ceph/users/igriffith/datasets/nsynth}"
cd "${PROJECT_ROOT}"
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE

supervised_flag=()
if [ "$SLURM_ARRAY_TASK_ID" -ge 8 ]; then
    supervised_flag=(--supervised_backbone)
fi

srun python3 lightning_scripts/eval_nsynth_linear.py \
    --config_list_path train_config_manifests/cochdnn9_sup_and_ssl_eval_configs.pkl \
    --array_ix $SLURM_ARRAY_TASK_ID \
    --gpus $num_gpus \
    --num_workers $SLURM_JOB_CPUS_PER_NODE \
    --model_ckpt_dir model_checkpoints \
    --batch_size 256 \
    --layer_str 'relu4' \
    --optimizer "AdamW" \
    --lr 0.01 \
    --task 'family' \
    --train_epochs 10 \
    --duration 2.0 \
    "${supervised_flag[@]}" \
    --no-time_avg_rep \
    --no-lr_scheduler \
    --eval_only \
    --use_classifier_ckpt  \
    --with_dropout 

