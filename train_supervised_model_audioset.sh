#!/bin/bash -l
#SBATCH --job-name=train_sup_audioset
#SBATCH --output=outLogs/train_kell2018_audioset_unbalanced_%j.out
#SBATCH --error=outLogs/train_kell2018_audioset_unbalanced_%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16

#SBATCH --mem=1000Gb
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH -C h100

conda activate cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: $master_node Local node: $HOSTNAME GPUs used: $CUDA_VISIBLE_DEVICES Total GPUs on that node: $num_gpus CPUs per node: $SLURM_JOB_CPUS_PER_NODE"

srun -K --cpu-bind=cores python3 lightning_scripts/train.py \
    --config_path model_configs/supervised_models/kell2018_audioset_unbalanced_supervised.yaml \
    --gpus $num_gpus \
    --num_workers $SLURM_JOB_CPUS_PER_NODE \
    --exp_dir model_checkpoints \
    --resume_training

