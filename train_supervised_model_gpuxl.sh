#!/bin/bash -l
#SBATCH --job-name=train_supervised
#SBATCH --output=outLogs/train_word_aud_supervised_%j.out
#SBATCH --error=outLogs/train_word_aud_supervised_%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16

#SBATCH --mem=1000Gb
#SBATCH --time=12:00:00
#SBATCH --partition=gpuxl
#SBATCH -N 1

module purge
module load python
module load cuda cudnn nccl

conda activate ~/ceph/conda_envs/cochdnn_ssl_pl

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE


# srun python3 lightning_scripts/train.py --config lightning_scripts/configs/word_audioset_resnet50.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --exp_dir model_checkpoints \
#                                    --resume_training 

srun python3 lightning_scripts/train.py --config lightning_scripts/configs/word_audioset_resnet50_for_gpuxl.yaml \
                                   --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --exp_dir model_checkpoints \
                                   --resume_training 

