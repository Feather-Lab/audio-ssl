#!/bin/bash -l
#SBATCH --job-name=eval_nsynth_byola
#SBATCH --output=outLogs/eval_nsynth_byola_%A_%a.out
#SBATCH --error=outLogs/eval_nsynth_byola_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64Gb
#SBATCH --time=1-01:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=a100-80gb

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE

# NSynth instrument family classification with pre-trained BYOL-A
srun python3 lightning_scripts/eval_nsynth_linear.py \
    --config_path byol-a/config.yaml \
    --gpus $num_gpus \
    --num_workers $SLURM_JOB_CPUS_PER_NODE \
    --model_ckpt_dir model_checkpoints \
    --batch_size 256 \
    --layer_str 'final' \
    --optimizer "AdamW" \
    --lr 0.005 \
    --task 'family' \
    --train_epochs 10 \
    --duration 1.0 \
    --time_avg_rep \
    --no-lr_scheduler \
    --no-eval_only \
    --no-use_classifier_ckpt
