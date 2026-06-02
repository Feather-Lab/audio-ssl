#!/bin/bash -l
#SBATCH --job-name=whisper_esc50
#SBATCH --output=outLogs/whisper_esc50_%A_%a.out
#SBATCH --error=outLogs/whisper_esc50_%A_%a.err
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=200Gb
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --array=32

mamba activate cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn

echo "Job array ID: $SLURM_ARRAY_TASK_ID  (layers 0-31 = encoder blocks, 32 = ln_post)"

python3 lightning_scripts/eval_audiomae_esc50.py \
    --model_type whisper --whisper_model large-v3 \
    -D /tmp/igriffith \
    -L $SLURM_ARRAY_TASK_ID \
    -A 4096 -R 5 -P -O \
    -C 0.01 0.1 1 10 100
