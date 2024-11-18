#!/bin/bash -l
#SBATCH --job-name=train_ssl_config
#SBATCH --output=outLogs/train_ssl_dual_task_lr_and_opt_%A_%a.out
#SBATCH --error=outLogs/train_ssl_dual_task_lr_and_opt_%A_%a.err
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=12

#SBATCH --mem=64Gb
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=h100  # if you want a particular type of GPU
#SBATCH --array=0-3 # 0-3 in manifest

# module purge
# module load python
# module load cuda cudnn nccl

conda activate cochdnn_ssl_pl

# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE


# srun python3 lightning_scripts/train.py --config_list train_config_manifests/single_task_ssl_hpara_search_bs_768.pkl \
#                                    --array_id $SLURM_ARRAY_TASK_ID \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --exp_dir model_checkpoints \
#                                  #  --resume_training 

srun python3 lightning_scripts/train.py --config_list train_config_manifests/pilot_dualtask_learning_rate_barlow_768-batch-size.pkl \
                                   --array_id $SLURM_ARRAY_TASK_ID \
                                   --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --exp_dir model_checkpoints \
                                 #  --resume_training 

