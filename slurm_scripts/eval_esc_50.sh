#!/bin/bash -l
#SBATCH --job-name=eval_esc50
#SBATCH --output=outLogs/eval_esc50_%A_%a.out
#SBATCH --error=outLogs/eval_esc50_%A_%a.err
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1

#SBATCH --mem=200Gb
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --array=0-11

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/mnt/home/igriffith/ceph/projects/cochdnn}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export COCHDNN_ESC50_DIR="${COCHDNN_ESC50_DIR:-/mnt/ceph/users/igriffith/datasets/ESC-50-master}"
cd "${PROJECT_ROOT}"
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE



CONFIG_LIST_PATH="train_config_manifests/cochdnn9_sup_and_ssl_eval_configs.pkl"
# Optional: COCHDNN_ESC50_MODEL_SET=scaled maps array 0..2 to manifest indices
# [8,7,6] -> [scaled audioset supervised, scaled ssl lambda=0.0, scaled ssl lambda=0.5].
MODEL_SET="${COCHDNN_ESC50_MODEL_SET:-all}"
ARRAY_IX="$SLURM_ARRAY_TASK_ID"
if [ "$MODEL_SET" = "scaled" ]; then
    scaled_ix=(8 7 6)
    if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -gt 2 ]; then
        echo "For COCHDNN_ESC50_MODEL_SET=scaled, submit with --array=0-2."
        exit 1
    fi
    ARRAY_IX="${scaled_ix[$SLURM_ARRAY_TASK_ID]}"
fi

echo "ESC-50 model set: $MODEL_SET, slurm_array_ix: $SLURM_ARRAY_TASK_ID, manifest_ix: $ARRAY_IX"

python3 lightning_scripts/make_esc_pl_model_plots.py \
                                   --config_list_path "$CONFIG_LIST_PATH" \
                                   --array_ix "$ARRAY_IX" \
                                   -D "${COCHDNN_SCRATCH_DIR:-/tmp/cochdnn}" -L 15 -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
                                   --model_ckpt_dir model_checkpoints
