#!/bin/bash -l
#SBATCH --job-name=check_py
#SBATCH --output=outLogs/check_py_%j.out
#SBATCH --error=outLogs/check_py_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1

#SBATCH --mem=1Gb
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=h100  # if you want a particular type of GPU

module load cuda cudnn nccl
conda activate ~/ceph/conda_envs/cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME



# python3 -c "import torch; print(torch.cuda.is_available())" 
python3 -c "print('hello')" 
 