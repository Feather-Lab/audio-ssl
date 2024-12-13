#!/bin/bash -l
#SBATCH --job-name=eval_jsin
#SBATCH --output=outLogs/eval_jsin_speaker_transfer_%A_%a.out
#SBATCH --error=outLogs/eval_jsin_speaker_transfer_%A_%a.err
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8

#SBATCH --mem=48Gb
#SBATCH --time=9:00:00 # approx 6 if training classifier from scratch. 5 if just evaling
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=h100  # if you want a particular type of GPU
#SBATCH --array=0 #-17 #0-7 for current 

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE




# srun python3 lightning_scripts/eval_jsin_transfer.py --config_path model_configs/ssl_barlow_word_resnet50_hparam_set_0.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --array_ix $SLURM_ARRAY_TASK_ID \
#                                    --layer_str 'avgpool' \
#                                    --optimizer "LARS" --lr 0.2 \
#                                    --task "speaker" --eval_only \

# srun python3 lightning_scripts/eval_jsin_transfer.py --config_path model_configs/pilot_ssl_barlow_dualtask_resnet50_hparam_set_13_lr_02_LARS_MatchedSpeechInNoiseDatasetBatched.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --array_ix $SLURM_ARRAY_TASK_ID \
#                                    --layer_str 'avgpool' \
#                                    --optimizer "LARS" --lr 0.2 \
#                                    --task "speaker" --eval_only \

# srun python3 lightning_scripts/eval_jsin_transfer.py --config_path model_configs/pilot_ssl_barlow_dualtask_resnet50_hparam_set_13_lr_06_LARS.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --array_ix $SLURM_ARRAY_TASK_ID \
#                                    --layer_str 'avgpool' \
#                                    --optimizer "LARS" --lr 0.6 \
#                                    --task "speaker" --eval_only \


# srun python3 lightning_scripts/eval_jsin_transfer.py --config_path model_configs/pilot_ssl_barlow_dualtask_resnet50_hparam_set_1_lr_02_LARS.yaml \
                                #    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                #    --model_ckpt_dir model_checkpoints \
                                #    --batch_size 192 \
                                #    --array_ix $SLURM_ARRAY_TASK_ID \
                                #    --layer_str 'avgpool' \
                                #    --optimizer "LARS" --lr 0.4 \
                                #    --task "word" #  --eval_only \


# srun python3 lightning_scripts/eval_jsin_transfer.py --config_path model_configs/pilot_ssl_barlow_dualtask_resnet50_hparam_set_1_lr_02_LARS_MatchedSpeechInNoiseDatasetBatched.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --array_ix $SLURM_ARRAY_TASK_ID \
#                                    --layer_str 'avgpool' \
#                                    --optimizer "LARS" --lr 0.2 \
#                                    --task "speaker" \
#                                    --eval_only \
                                #    --ckpt_path model_checkpoints/pilot_ssl_barlow_dualtask_resnet50_hparam_set_1_lr_02_LARS_MatchedSpeechInNoiseDatasetBatched/checkpoints/epoch=24-step=7500-v1.ckpt



srun python3 lightning_scripts/eval_jsin_transfer.py --config_path model_configs/pilot_ssl_mmcr_dualtask_resnet50_hparam_set_1_lr_02_LARS_MatchedSpeechInNoiseDatasetBatched.yaml \
                                   --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --model_ckpt_dir model_checkpoints \
                                   --batch_size 192 \
                                   --array_ix $SLURM_ARRAY_TASK_ID \
                                   --layer_str 'avgpool' \
                                   --optimizer "LARS" --lr 0.2 \
                                   --task "speaker" \
                                   --ckpt_path model_checkpoints/pilot_ssl_mmcr_dualtask_resnet50_hparam_set_1_lr_02_LARS_MatchedSpeechInNoiseDatasetBatched/checkpoints/epoch=8-step=2700.ckpt \
                                   --overwrite



