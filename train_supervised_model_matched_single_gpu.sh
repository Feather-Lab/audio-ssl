#!/bin/bash -l
#SBATCH --job-name=train_supervised
#SBATCH --output=outLogs/train_word_aud_supervised_matched_%j.out
#SBATCH --error=outLogs/train_word_aud_supervised_matched_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16

#SBATCH --mem=32Gb
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=h100  # if you want a particular type of GPU

module purge
module load cuda cudnn nccl

conda activate cochdnn_ssl_pl

export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE



srun --cpu-bind=cores python3 lightning_scripts/train.py --config_path model_configs/word_speaker_audioset_resnet18_MatchedDataset_shuffle_one_gpu.yaml \
                                   --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --exp_dir model_checkpoints \
                                   --resume_training 
# 
