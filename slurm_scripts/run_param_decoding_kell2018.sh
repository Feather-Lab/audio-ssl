#!/bin/bash -l
#SBATCH --job-name=param_decoding
#SBATCH --output=outLogs/param_decoding_%A_%a.out
#SBATCH --error=outLogs/param_decoding_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8

#SBATCH --mem=256Gb
#SBATCH --time=3:00:00 
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=a100-80gb  # if you want a particular type of GPU
#SBATCH --array=3,7,11,13,15,18 # 3,7,11,13,15,18 in manifest

module load cuda cudnn nccl
mamba activate cochdnn_ssl_pl

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE

srun -K python3 -u lightning_scripts/run_param_decoding_single_model.py  \
                                   --model_config model_configs/supervised_models/word_kell2018_MatchedDataset_LARS.yaml \
                                   --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --output_dir parameter_decoding_v2 \
                                   --batch_size 192 \
                                   --num_eval 10 \
                                   --num_train 50 \
                                   --job_id $SLURM_ARRAY_TASK_ID \
                                   --ridge_alpha 0.5
