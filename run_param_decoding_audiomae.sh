#!/bin/bash -l
#SBATCH --job-name=param_decoding_am
#SBATCH --output=outLogs/param_decoding_audiomae_%A_%a.out
#SBATCH --error=outLogs/param_decoding_audiomae_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=256Gb
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=a100-80gb
#SBATCH --array=12

module load cuda cudnn nccl
mamba activate cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn

LAYERS=(block_0 block_1 block_2 block_3 block_4 block_5 block_6 block_7 block_8 block_9 block_10 block_11 norm)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

echo "Job array ID: $SLURM_ARRAY_TASK_ID  Layer: $LAYER"

srun -K python3 -u lightning_scripts/run_param_decoding_single_model.py \
    --model_type audiomae \
    --input_sample_rate 20000 \
    --num_workers "$SLURM_JOB_CPUS_PER_NODE" \
    --output_dir parameter_decoding_audiomae \
    --batch_size 192 \
    --num_eval 10 \
    --num_train 50 \
    --job_id "$SLURM_ARRAY_TASK_ID" \
    --ridge_alpha 0.5
