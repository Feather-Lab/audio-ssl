#!/bin/bash -l
#SBATCH --job-name=param_decoding
#SBATCH --output=outLogs/param_decoding_%j.out
#SBATCH --error=outLogs/param_decoding_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8

#SBATCH --mem=20Gb
#SBATCH --time=0:30:00 # approx 6 if training classifier from scratch. 10 min if just evaling
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=a100-80gb  # if you want a particular type of GPU
##SBATCH --array=0 #-17 #0-7 for current 

module load cuda cudnn nccl
mamba activate cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE

# srun python3 lightning_scripts/run_param_decoding.py  \
#                                    --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --batch_size 192 \
#                                    --num_eval 5 \
#                                    --num_train 25 \
#                                    --layer 'invar_head' \


srun -K python3 -u lightning_scripts/run_param_decoding.py  \
                                   --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --batch_size 192 \
                                   --num_eval 5 \
                                   --num_train 50 \
                                   --layer 'relu4' \

