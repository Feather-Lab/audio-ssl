#!/bin/bash -l
#SBATCH --job-name=eval_jsin
#SBATCH --output=outLogs/eval_jsin_transfer_both_tasks_%A_%a.out
#SBATCH --error=outLogs/eval_jsin_transfer_both_tasks_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8

#SBATCH --mem=32Gb ## Just 32 if evaling 
#SBATCH --time=1-01:00:00 # approx 20 (~3hr/epoch) if training classifier from scratch. 10 min if just evaling
##SBATCH --time=00:12:00 # approx 20 (~3hr/epoch) if training classifier from scratch. 10 min if just evaling
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=a100-80gb  # if you want a particular type of GPU
##SBATCH --array=0-4 #0-4 # 0-4 for kell2018 and resnet18 equivariant training manifests  

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE


# srun python3 lightning_scripts/eval_jsin_transfer_matched.py --config_list_path train_config_manifests/kell2018_barlow_equivariant_lmbda_search_w_augments.pkl \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --layer_str 'relu0' \
#                                    --optimizer "AdamW" --lr 0.01 \
#                                    --task 'both' \
#                                    --train_epochs 6 \
#                                    --array_ix $SLURM_ARRAY_TASK_ID \
#                                    --no-with_noise --eval_only --lr_scheduler --use_classifier_ckpt --no-time_avg_rep --with_dropout 

srun python3 lightning_scripts/eval_jsin_transfer_matched.py --config_path model_configs/whisper_tiny_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_0e-01.yaml \
                                   --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --model_ckpt_dir model_checkpoints \
                                   --batch_size 192 \
                                   --layer_str 'encoder_block_3' \
                                   --optimizer "AdamW" --lr 0.005 \
                                   --task 'both' \
                                   --train_epochs 6 \
                                   --no-with_noise --eval_only --lr_scheduler --use_classifier_ckpt --no-time_avg_rep --with_dropout 
                                   
# srun python3 lightning_scripts/eval_jsin_transfer_matched.py --config_path model_configs/supervised_models/kell2018_audioset_unbalanced_supervised.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --layer_str 'relu4' \
#                                    --optimizer "AdamW" --lr 0.0005 \
#                                    --task 'both' \
#                                    --train_epochs 6 \
#                                    --supervised_backbone \
#                                    --no-with_noise --no-eval_only --lr_scheduler --no-use_classifier_ckpt --no-time_avg_rep --with_dropout 
                                   