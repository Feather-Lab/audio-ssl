#!/bin/bash -l
#SBATCH --job-name=audiomae_natsounds
#SBATCH --output=outLogs/audiomae_natsounds_%A.out
#SBATCH --error=outLogs/audiomae_natsounds_%A.err
#SBATCH --cpus-per-gpu=2
#SBATCH --gpus=1
#SBATCH --mem=64Gb
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH -N 1

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

python3 fmri_analysis/measure_layer_activations_165_natural_sounds_lightning.py \
    --config_path model_configs/audiomae_pretrained_natsounds.yaml
