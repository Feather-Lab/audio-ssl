#!/bin/bash -l
#SBATCH --job-name=eval_jsin
#SBATCH --output=outLogs/eval_speech_commands_transfer_%j.out
#SBATCH --error=outLogs/eval_speech_commands_transfer_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8

#SBATCH --mem=40Gb
#SBATCH --time=0:30:00 # approx 6 if training classifier from scratch. 10 min if just evaling
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=a100-80gb  # if you want a particular type of GPU
#SBATCH --array=0-12

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


if [ "$SLURM_ARRAY_TASK_ID" -eq 12 ]; then
    srun python3 lightning_scripts/eval_speech_commands_transfer.py \
        --config_path byol-a/config.yaml \
        --gpus "$num_gpus" --num_workers "$SLURM_JOB_CPUS_PER_NODE" \
        --model_ckpt_dir "$model_ckpt_dir" \
        --batch_size 4096 \
        --layer_str final \
        --optimizer AdamW --lr 0.01 \
        --classifier_ckpt_path "$model_ckpt_dir/config/speech_commands_linear_classifier_checkpoints/AdamW_0.01_cosine_lr_scheduler_/epoch=3-step=1068.ckpt" \
        --eval_only --lr_scheduler --no-time_avg_rep
elif [ "$SLURM_ARRAY_TASK_ID" -eq 5 ]; then
    srun python3 lightning_scripts/eval_speech_commands_transfer.py \
        --config_path model_configs/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment.yaml \
        --ckpt_path "$model_ckpt_dir/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment/checkpoints/epoch=124-step=22500-best_val.ckpt" \
        --gpus "$num_gpus" --num_workers "$SLURM_JOB_CPUS_PER_NODE" \
        --model_ckpt_dir "$model_ckpt_dir" \
        --batch_size 192 \
        --layer_str relu4 \
        --optimizer AdamW --lr 0.001 \
        --eval_only --no-lr_scheduler --no-time_avg_rep
else
    lr=0.01
    # Match backup behavior: only use scaled LR if matching classifier ckpts exist.
    if [ "$SLURM_ARRAY_TASK_ID" -ge 6 ] && [ "$SLURM_ARRAY_TASK_ID" -le 8 ]; then
        classifier_dir=""
        if [ "$SLURM_ARRAY_TASK_ID" -eq 6 ]; then
            classifier_dir="$model_ckpt_dir/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_5e-01_audioset_only/speech_commands_linear_classifier_checkpoints/AdamW_relu4_full_rep_0.0005_cosine_lr_scheduler_"
        elif [ "$SLURM_ARRAY_TASK_ID" -eq 7 ]; then
            classifier_dir="$model_ckpt_dir/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_0e-01_audioset_only/speech_commands_linear_classifier_checkpoints/AdamW_relu4_full_rep_0.0005_cosine_lr_scheduler_"
        elif [ "$SLURM_ARRAY_TASK_ID" -eq 8 ]; then
            classifier_dir="$model_ckpt_dir/kell2018_audioset_unbalanced_supervised/speech_commands_linear_classifier_checkpoints/AdamW_relu4_full_rep_0.0005_cosine_lr_scheduler_"
        fi

        if [ -n "$classifier_dir" ] && ls "$classifier_dir"/*.ckpt >/dev/null 2>&1; then
            lr=0.0005
            echo "Using lr=0.0005 for task ${SLURM_ARRAY_TASK_ID}; found $classifier_dir"
        else
            echo "Using lr=0.01 for task ${SLURM_ARRAY_TASK_ID}; missing 0.0005 classifier ckpt dir"
        fi
    fi

    supervised_flag=()
    if [ "$SLURM_ARRAY_TASK_ID" -ge 8 ]; then
        supervised_flag=(--supervised_backbone)
    fi

    srun python3 lightning_scripts/eval_speech_commands_transfer.py \
        --config_list_path "$config_list_path" \
        --array_ix "$SLURM_ARRAY_TASK_ID" \
        --gpus "$num_gpus" --num_workers "$SLURM_JOB_CPUS_PER_NODE" \
        --model_ckpt_dir "$model_ckpt_dir" \
        --batch_size 192 \
        --layer_str relu4 \
        --optimizer AdamW --lr "$lr" \
        "${supervised_flag[@]}" \
        --eval_only --lr_scheduler --no-time_avg_rep
fi
