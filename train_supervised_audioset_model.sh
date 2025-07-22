#!/bin/bash -l
#SBATCH --job-name=train_supervised
#SBATCH --output=outLogs/train_audioset_supervised_kell2018_%j.out
#SBATCH --error=outLogs/train_audioset_supervised_kell2018_%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=1000Gb
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100  # if you want a particular type of GPU
#SBATCH -N 1

conda activate cochdnn_ssl_pl


export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1

export PYTHONPATH=$PYTHONPATH:~/ceph/projects/cochdnn

master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE


srun python3 lightning_scripts/train.py --config_path model_configs/supervised_models/audioset_kell2018_MatchedDataset_LARS.yaml \
                                   --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --exp_dir model_checkpoints \
                                   --resume_training 

