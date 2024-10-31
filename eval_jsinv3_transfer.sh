#!/bin/bash -l
#SBATCH --job-name=eval_jsin
#SBATCH --output=outLogs/eval_jsin_transfer_%A_%a.out
#SBATCH --error=outLogs/eval_jsin_transfer_%A_%a.err
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=1

#SBATCH --mem=16Gb
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=a100  # if you want a particular type of GPU
#SBATCH --array=0 #-17 #0-7 for current 

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE


python3 lightning_scripts/eval_jsin_transfer.py --config_list_path train_config_manifests/single_task_barlow_hpara_search_bs_256.pkl \
                                   --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --model_ckpt_dir model_checkpoints \
                                   --batch_size 192 \
                                   --array_ix $SLURM_ARRAY_TASK_ID \
                                   --layer_str 'avgpool'


