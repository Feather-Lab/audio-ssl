#!/bin/bash -l
# Example CE-SSL CochCNN9 training job (one representative config).
#
# Required environment variables — HDF5 paths are substituted from the config:
#   COCHDNN_JSIN_TRAIN_H5, COCHDNN_JSIN_VALID_H5
#   COCHDNN_AUDIONOISE_TRAIN_H5, COCHDNN_AUDIONOISE_VALID_H5
#
# Optional overrides (edit below or export before sbatch):
#   COCHDNN_EXP_DIR — checkpoint/log root (default: ./exp)
#
# Submit from the repository root:
#   sbatch slurm_scripts/train_cochdnn_ce_ssl.sh

#SBATCH --job-name=train_ce_ssl
#SBATCH --output=outLogs/train_ce_ssl_%A.out
#SBATCH --error=outLogs/train_ce_ssl_%A.err
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=4
#SBATCH --mem=128Gb
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH -N 1

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

CONFIG_PATH="model_configs/kell2018_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_5e-01.yaml"
EXP_DIR="${COCHDNN_EXP_DIR:-./exp}"

python3 lightning_scripts/train.py \
    --config_path "${CONFIG_PATH}" \
    --exp_dir "${EXP_DIR}" \
    --gpus 4 \
    --num_workers 32 \
    --num_nodes 1
