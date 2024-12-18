#!/bin/bash -l
#SBATCH --job-name=word_ssl
#SBATCH --output=outLogs/train_audioset_barlow__pilot_%j.out
#SBATCH --error=outLogs/train_audioset_barlow__pilot_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8

#SBATCH --mem=32Gb
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=h100  # if you want a particular type of GPU

mamba activate cochdnn_ssl_pl

#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1

export PYTHONPATH=$PYTHONPATH:~/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs visible on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE



srun python3 lightning_scripts/train.py --config_path model_configs/barlow_audioset_resnet18_rand_crop.yaml \
                                   --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --exp_dir model_checkpoints \
                                   --resume_training 



