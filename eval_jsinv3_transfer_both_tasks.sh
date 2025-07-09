#!/bin/bash -l
#SBATCH --job-name=eval_jsin
#SBATCH --output=outLogs/eval_jsin_transfer_both_tasks_%j.out
#SBATCH --error=outLogs/eval_jsin_transfer_both_tasks_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8

#SBATCH --mem=32Gb ## Just 32 if evaling 
#SBATCH --time=6:00:00 # approx 6 if training classifier from scratch. 10 min if just evaling
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=a100-80gb  # if you want a particular type of GPU
##SBATCH --array=0 #-17 #0-7 for current 

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE

srun python3 lightning_scripts/eval_jsin_transfer_matched.py --config_path model_configs/resnet18_barlow_invariant_only_lmbda_1e-2_lr_2e-1_w_invar_augment_no_avgpool.yaml \
                                   --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --model_ckpt_dir model_checkpoints \
                                   --batch_size 192 \
                                   --layer_str 'layer2' \
                                   --optimizer "AdamW" --lr 0.01 \
                                   --task 'word' \
                                   --train_epochs 6 \
                                   --no-with_noise --no-eval_only --lr_scheduler --no-use_classifier_ckpt --no-time_avg_rep --with_dropout

# srun python3 lightning_scripts/eval_jsin_transfer_matched.py --config_path model_configs/resnet18_barlow_equivariant_lmbda_1e-2_lr_2e-1_w_invar_augment_no_avgpool_eq_lmbda_3e-01.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --layer_str 'layer2' \
#                                    --optimizer "AdamW" --lr 0.01 \
#                                    --task 'word' \
#                                    --train_epochs 6 \
#                                    --no-with_noise --no-eval_only --lr_scheduler --no-use_classifier_ckpt --no-time_avg_rep --with_dropout 

# srun python3 lightning_scripts/eval_jsin_transfer_matched.py --config_path model_configs/barlow_dualtask_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment_eq_lmbda_1e-1_fixed_loss.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --layer_str 'relu2' \
#                                    --optimizer "AdamW" --lr 0.0001 \
#                                    --task 'word' \
#                                    --train_epochs 6 \
#                                    --no-with_noise --no-eval_only --lr_scheduler --no-use_classifier_ckpt --no-time_avg_rep

