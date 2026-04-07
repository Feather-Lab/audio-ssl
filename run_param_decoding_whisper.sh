#!/bin/bash -l
#SBATCH --job-name=param_decoding_whisper
#SBATCH --output=outLogs/param_decoding_whisper_%A_%a.out
#SBATCH --error=outLogs/param_decoding_whisper_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=256Gb
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=a100-80gb
#SBATCH --array=0-32:2

module load cuda cudnn nccl
mamba activate cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn

WHISPER_MODEL=${WHISPER_MODEL:-large-v3}

LAYERS=(
encoder_block_0 encoder_block_1 encoder_block_2 encoder_block_3 encoder_block_4 encoder_block_5
encoder_block_6 encoder_block_7 encoder_block_8 encoder_block_9 encoder_block_10 encoder_block_11
encoder_block_12 encoder_block_13 encoder_block_14 encoder_block_15 encoder_block_16 encoder_block_17
encoder_block_18 encoder_block_19 encoder_block_20 encoder_block_21 encoder_block_22 encoder_block_23
encoder_block_24 encoder_block_25 encoder_block_26 encoder_block_27 encoder_block_28 encoder_block_29
encoder_block_30 encoder_block_31 ln_post
)

if [ -n "${LAYER}" ]; then
    echo "Manual layer mode: LAYER=${LAYER}"
    LAYER_ARGS=(--layer "${LAYER}")
else
    JOB_ID=${SLURM_ARRAY_TASK_ID:-0}
    echo "Array mode: job_id=${JOB_ID}"
    LAYER_ARGS=(--job_id "${JOB_ID}")
fi

srun -K python3 -u lightning_scripts/run_param_decoding_single_model.py \
    --model_type whisper \
    --whisper_model "${WHISPER_MODEL}" \
    --input_sample_rate 20000 \
    --num_workers "${SLURM_JOB_CPUS_PER_NODE}" \
    --output_dir parameter_decoding_whisper \
    --batch_size 128 \
    --num_eval 10 \
    --num_train 50 \
    "${LAYER_ARGS[@]}" \
    --ridge_alpha 0.5
