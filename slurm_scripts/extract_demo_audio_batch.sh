#!/bin/bash -l
# Extract demo audio batch for audio_transforms_demo.ipynb (I/O-bound HDF5 read).
#
# Required environment variables (set before sbatch or in your shell profile):
#   COCHDNN_JSIN_TRAIN_H5       — path to JSIN train speech HDF5
#   COCHDNN_AUDIONOISE_TRAIN_H5 — path to Audionoise train noise HDF5
#
# Writes float32 waveforms to notebooks/demo_notebooks/demo_audio_batch/.
# Submit from the repository root:
#   sbatch slurm_scripts/extract_demo_audio_batch.sh

#SBATCH --job-name=extract_demo_audio
#SBATCH --output=outLogs/extract_demo_audio_%A.out
#SBATCH --error=outLogs/extract_demo_audio_%A.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16Gb
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH -N 1

mamba activate cochdnn_ssl_pl

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

python3 data_scripts/extract_demo_audio_batch.py \
    --out_dir notebooks/demo_notebooks/demo_audio_batch
