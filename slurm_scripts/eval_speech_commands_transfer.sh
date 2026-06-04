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
##SBATCH --array=0 #-17 #0-7 for current 

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE



# srun python3 lightning_scripts/eval_speech_commands_transfer.py --config_path byol-a/config.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --layer_str 'final' \
#                                    --optimizer "AdamW" --lr 0.001 \
#                                   --no-eval_only --no-lr_scheduler --no-time_avg_rep
                                #    --ckpt_path model_checkpoints/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment/checkpoints/epoch=160-step=28980-best_word_task.ckpt \

# srun python3 lightning_scripts/eval_speech_commands_transfer.py \
                                #    --config_path model_configs/supervised_models/word_kell2018_MatchedDataset_LARS.yaml \
                                #    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                #    --model_ckpt_dir model_checkpoints \
                                #    --batch_size 192 \
                                #    --layer_str 'relu4' \
                                #    --optimizer "AdamW" --lr 0.001 \
                                #     --no-eval_only --no-lr_scheduler --no-time_avg_rep  --supervised_backbone


srun python3 lightning_scripts/eval_speech_commands_transfer.py \
                                   --config_path model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_0e-01.yaml \
                                   --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --model_ckpt_dir model_checkpoints \
                                   --batch_size 192 \
                                   --layer_str 'relu4' \
                                   --optimizer "AdamW" --lr 0.01 \
                                  --no-eval_only --lr_scheduler --no-time_avg_rep --supervised_backbone






