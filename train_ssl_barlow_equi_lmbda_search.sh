#!/bin/bash -l
#SBATCH --job-name=equi_lmbda_search
#SBATCH --output=outLogs/barlow_equivariant_lmda_search_resnet18_%A_%a.out
#SBATCH --error=outLogs/barlow_equivariant_lmda_search_resnet18_%A_%a.err
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16

#SBATCH --mem=1000Gb
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100  # if you want a particular type of GPU
#SBATCH -N 1
#SBATCH --array=1 #0-3 # 0-5 in manifest; 0-6 if invar_to_augments; 0-4 for resnet18s

# module purge
# module load python
# module load cuda cudnn nccl

conda activate cochdnn_ssl_pl

# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

export PYTHONPATH=$PYTHONPATH:~/ceph/projects/cochdnn
export PYTHONFAULTHANDLER=1
# export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE

srun -K --cpu-bind=cores python3 lightning_scripts/train.py --config_list train_config_manifests/resnet18_barlow_equivariant_lmbda_search_w_augments.pkl \
                                   --array_id $SLURM_ARRAY_TASK_ID \
                                   --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --exp_dir model_checkpoints \
                                   --ckpt_path model_checkpoints/resnet18_barlow_equivariant_lmbda_1e-2_lr_2e-1_no_avgpool_eq_lmbda_2e-01/checkpoints/epoch=49-step=11250-best_val.ckpt \
                                   --resume_training 

# srun -K --cpu-bind=cores python3 lightning_scripts/train.py --config_list train_config_manifests/barlow_equivariant_lmbda_search_kell2018_invar_to_augments.pkl \
#                                    --array_id $SLURM_ARRAY_TASK_ID \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --exp_dir model_checkpoints \
#                                   --resume_training 

