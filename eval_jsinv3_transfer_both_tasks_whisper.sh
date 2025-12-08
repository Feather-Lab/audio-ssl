#!/bin/bash -l
#SBATCH --job-name=eval_jsin
#SBATCH --output=outLogs/eval_jsin_transfer_both_tasks_%A_%a.out
#SBATCH --error=outLogs/eval_jsin_transfer_both_tasks_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8

#SBATCH --mem=64Gb ## Just 32 if evaling 
#SBATCH --time=2-01:00:00 # approx 20 (~3hr/epoch) if training classifier from scratch. 10 min if just evaling
##SBATCH --time=00:12:00 # approx 20 (~3hr/epoch) if training classifier from scratch. 10 min if just evaling
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=a100-80gb  # if you want a particular type of GPU
##SBATCH --array=0-4 #0-4 # 0-4 for kell2018 and resnet18 equivariant training manifests  

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE


srun python3 lightning_scripts/eval_jsin_transfer_matched.py --config_path whisper_large-v3-turbo \
                                   --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --model_ckpt_dir model_checkpoints \
                                   --batch_size 192 \
                                   --layer_str '31' \
                                   --optimizer "AdamW" --lr 0.0005 \
                                   --task 'both' \
                                   --train_epochs 6 \
                                   --checkpoint_every_n_steps 2000 \
                                   --no-with_noise --no-eval_only --lr_scheduler --use_classifier_ckpt --no-time_avg_rep --with_dropout 
                                   