#!/bin/bash -l
#SBATCH --job-name=eval_jsin
#SBATCH --output=outLogs/eval_jsin_transfer_both_tasks_%j.out
#SBATCH --error=outLogs/eval_jsin_transfer_both_tasks_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8

#SBATCH --mem=80Gb
#SBATCH --time=4:00:00 # approx 6 if training classifier from scratch. 10 min if just evaling
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



# srun python3 lightning_scripts/eval_jsin_transfer_matched.py --config_path model_configs/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-1_no_schedule_lr_6e-1.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --layer_str 'relufc' \
#                                    --optimizer "AdamW" --lr 0.001 \
#                                    --ckpt_path model_checkpoints/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-1_no_schedule_lr_6e-1/checkpoints/epoch=61-step=11160-best_speaker_task.ckpt \
#                                    --overwrite

# srun python3 lightning_scripts/eval_jsin_transfer_matched.py --config_path model_configs/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-1_no_schedule.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --layer_str 'relufc' \
#                                    --optimizer "AdamW" --lr 0.001 \
#                                    --ckpt_path model_checkpoints/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-1_no_schedule/checkpoints/epoch=143-step=25920-best_train.ckpt \
#                                    --overwrite

# srun python3 lightning_scripts/eval_jsin_transfer_matched.py --config_path model_configs/barlow_word_resnet18_base_Matched_blocked_batches_lmbda_1e-1_3-layer_proj.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --layer_str 'avgpool' \
#                                    --optimizer "AdamW" --lr 0.001 \
#                                    --ckpt_path model_checkpoints/barlow_word_resnet18_base_Matched_blocked_batches_lmbda_1e-1_3-layer_proj/checkpoints/epoch=54-step=9900-best_speaker_task.ckpt \
#                                 #    --overwrite

# srun python3 lightning_scripts/eval_jsin_transfer_matched.py --config_path model_configs/barlow_word_resnet18_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment_proj_1024_avg_pool.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --layer_str 'layer3' \
#                                    --optimizer "AdamW" --lr 0.01 \
#                                    --no-with_noise --no-eval_only --no-lr_scheduler --no-use_classifier_ckpt --no-time_avg_rep

                                #    --ckpt_path  \
                                #    --overwrite

# srun python3 lightning_scripts/eval_jsin_transfer_matched.py --config_path model_configs/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 96 \
#                                    --layer_str 'relufc' \
#                                    --optimizer "LARS" --lr 0.2 \
#                                    --task 'word' \
#                                    --ckpt_path model_checkpoints/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1/checkpoints/epoch=218-step=39420-best_word_task.ckpt \
#                                    --with_noise --no-eval_only
#                                 #    --overwrite

# srun python3 lightning_scripts/eval_jsin_transfer_matched.py --config_path model_configs/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --layer_str 'relu2' \
#                                    --optimizer "AdamW" --lr 0.01 \
#                                    --task 'word' \
#                                    --no-with_noise --no-eval_only --lr_scheduler --no-use_classifier_ckpt --no-time_avg_rep --crop_audio
                                   
# srun python3 lightning_scripts/eval_jsin_transfer_matched.py --config_path model_configs/barlow_word_resnet18_Matched_blocked_batches_lmbda_5e-3_lr_2e-1_w_augment_per_frame.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --layer_str 'layer2' \
#                                    --optimizer "AdamW" --lr 0.01 \
#                                    --task 'word' \
#                                    --no-with_noise --no-eval_only --lr_scheduler --no-use_classifier_ckpt --no-time_avg_rep #--crop_audio
                                   
# srun python3 lightning_scripts/eval_jsin_transfer_matched.py --config_path model_configs/supervised_models/word_resnet18_MatchedDataset_LARS.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --layer_str 'avgpool' \
#                                    --optimizer "AdamW" --lr 0.01 \
#                                    --task 'word' \
#                                    --with_noise --no-eval_only --no-lr_scheduler \
#                                    --no-use_classifier_ckpt --no-time_avg_rep --supervised_backbone
                                   
                                 #   --ckpt_path model_checkpoints/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment/checkpoints/epoch=160-step=28980-best_word_task.ckpt \
                                #    --overwrite

# srun python3 lightning_scripts/eval_jsin_transfer_matched.py --config_path model_configs/mmcr_word_kell2018_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --layer_str 'relu4' \
#                                    --optimizer "AdamW" --lr 0.01 \
#                                    --task 'word' \
#                                    --with_noise --no-eval_only --lr_scheduler --no-use_classifier_ckpt --no-time_avg_rep

                                #    --ckpt_path model_checkpoints/mmcr_word_kell2018_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment/checkpoints/epoch=110-step=19980-best_val.ckpt \
                                #    --with_noise --no-eval_only --lr_scheduler
                                #    --overwrite

srun python3 lightning_scripts/eval_jsin_transfer_matched.py --config_path model_configs/barlow_dualtask_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment_eq_lmbda_1e-1.yaml \
                                   --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --model_ckpt_dir model_checkpoints \
                                   --batch_size 192 \
                                   --layer_str 'relu2' \
                                   --optimizer "AdamW" --lr 0.01 \
                                   --task 'word' \
                                   --no-with_noise --no-eval_only --no-lr_scheduler --no-use_classifier_ckpt --no-time_avg_rep


# srun python3 lightning_scripts/eval_jsin_transfer_matched.py --config_path byol-a/config.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --layer_str 'avgpool' \
#                                    --optimizer "AdamW" --lr 0.01 \
#                                    --task 'word' \
#                                    --no-with_noise --no-eval_only --lr_scheduler --no-use_classifier_ckpt --no-time_avg_rep --crop_audio
                                   