#!/bin/bash -l
#SBATCH --job-name=param_decoding_whisper
#SBATCH --output=outLogs/param_decoding_whisper_%A_%a.out
#SBATCH --error=outLogs/param_decoding_whisper_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=512Gb
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=a100-80gb
#SBATCH --array=32

module load cuda cudnn nccl
mamba activate cochdnn_ssl_pl

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

WHISPER_MODEL=large-v3
echo "Job array ID: $SLURM_ARRAY_TASK_ID  Whisper model: $WHISPER_MODEL"

srun -K python3 -u lightning_scripts/run_param_decoding_single_model.py \
    --model_type whisper \
    --whisper_model "$WHISPER_MODEL" \
    --input_sample_rate 20000 \
    --num_workers "${SLURM_JOB_CPUS_PER_NODE}" \
    --output_dir parameter_decoding_v2 \
    --batch_size 128 \
    --num_eval 10 \
    --num_train 50 \
    --job_id "$SLURM_ARRAY_TASK_ID" \
    --ridge_alpha 0.5 \
