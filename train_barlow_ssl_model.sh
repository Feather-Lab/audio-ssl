#!/bin/bash -l
#SBATCH --job-name=word_ssl
#SBATCH --output=outLogs/barlow_dualtask_resnet18_MatchedSpeechInNoiseDatasetBatched_%j.out
#SBATCH --error=outLogs/barlow_dualtask_resnet18_MatchedSpeechInNoiseDatasetBatched_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16

#SBATCH --mem=48Gb
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=h100  # if you want a particular type of GPU

mamba activate cochdnn_ssl_pl

#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1

export PYTHONPATH=$PYTHONPATH:~/ceph/projects/cochdnn
export PYTHONFAULTHANDLER=1

master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs visible on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE



# srun python3 lightning_scripts/train.py --config_path model_configs/pilot_ssl_barlow_dualtask_resnet50_hparam_set_1_lr_06.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --exp_dir model_checkpoints \
#                                    --resume_training 

# srun python3 lightning_scripts/train.py --config_path model_configs/pilot_ssl_barlow_dualtask_resnet50_hparam_set_1_lr_06_LARS_sep_classifier_opt.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --exp_dir model_checkpoints \
#                                    --resume_training 

# srun python3 lightning_scripts/train.py --config_path model_configs/pilot_ssl_barlow_dualtask_resnet50_hparam_set_13_lr_02_LARS_MatchedSpeechInNoiseDatasetBatched.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --exp_dir model_checkpoints \
#                                   # --resume_training 

# srun python3 lightning_scripts/train.py --config_path model_configs/pilot_ssl_barlow_dualtask_resnet50_hparam_set_1_lr_02_LARS_MatchedSpeechInNoiseDatasetBatched.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --exp_dir model_checkpoints \
#                                   # --resume_training 


# srun python3 lightning_scripts/train.py --config_path model_configs/pilot_ssl_word_resnet18_barlow.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --exp_dir model_checkpoints \
#                                   # --resume_training 

srun --cpu-bind=cores python3 lightning_scripts/train.py --config_path model_configs/barlow_dualtask_resnet18_base_Matched_blocked_batches.yaml \
                                   --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --exp_dir model_checkpoints \
                                #   --resume_training 


