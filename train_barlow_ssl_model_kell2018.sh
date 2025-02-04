#!/bin/bash -l
#SBATCH --job-name=word_ssl
#SBATCH --output=outLogs/barlow_word_kell2018_MatchedSpeechInNoiseDatasetBatched_%j.out
#SBATCH --error=outLogs/barlow_word_kell2018_MatchedSpeechInNoiseDatasetBatched_%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16

#SBATCH --mem=1000Gb
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=a100-80gb  # if you want a particular type of GPU
#SBATCH -x workergpu156

##SBATCH --constraint=h100  # if you want a particular type of GPU
mamba activate cochdnn_ssl_pl


export PYTHONPATH=$PYTHONPATH:~/ceph/projects/cochdnn
export PYTHONFAULTHANDLER=1
# export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG_SUBSYS=ALL

master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs visible on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE

# srun -K --cpu-bind=cores python3 lightning_scripts/train.py --config_path model_configs/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-1_no_schedule.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --exp_dir model_checkpoints \
#                                    --resume_training 

# srun -K --cpu-bind=cores python3 lightning_scripts/train.py --config_path model_configs/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --exp_dir model_checkpoints \
#                                    --resume_training 
                                
# srun -K --cpu-bind=cores python3 lightning_scripts/train.py --config_path model_configs/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_skip_pairing.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --exp_dir model_checkpoints \
#                                    --resume_training 

srun -K --cpu-bind=cores python3 lightning_scripts/train.py --config_path model_configs/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment.yaml \
                                   --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --exp_dir model_checkpoints \
                                #    --resume_training 
                                 

# srun -K --cpu-bind=cores python3 lightning_scripts/train.py --config_path model_configs/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_pos_SNR.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --exp_dir model_checkpoints \
#                                    --resume_training 

# srun -K --cpu-bind=cores python3 lightning_scripts/train.py --config_path model_configs/barlow_word_kell2018_Matched_AdamW_lmbda_1e-2_lr_1e-3.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --exp_dir model_checkpoints \
#                                    --resume_training 
                                 
# srun -K --cpu-bind=cores python3 lightning_scripts/train.py --config_path model_configs/barlow_word_kell2018_base_Matched_blocked_batches_lmbda_1e-1.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --exp_dir model_checkpoints \
#                                    --resume_training 
