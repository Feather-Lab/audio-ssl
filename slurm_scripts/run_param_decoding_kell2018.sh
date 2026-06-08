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
#SBATCH --array=0-11

module load cuda cudnn nccl
mamba activate cochdnn_ssl_pl

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/mnt/home/igriffith/ceph/projects/cochdnn}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE

MODEL_CONFIG=$(python3 - "$SLURM_ARRAY_TASK_ID" <<'PY'
import pickle
import sys

with open("train_config_manifests/cochdnn9_sup_and_ssl_eval_configs.pkl", "rb") as handle:
    manifest = pickle.load(handle)
print(manifest[int(sys.argv[1])])
PY
)

supervised_flag=()
if [[ "$MODEL_CONFIG" == *"supervised_models"* ]]; then
    supervised_flag=(--supervised)
fi

srun -K python3 -u lightning_scripts/run_param_decoding_single_model.py  \
                                   --model_config "$MODEL_CONFIG" \
                                   --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --output_dir parameter_decoding_v2 \
                                   --batch_size 192 \
                                   --num_eval 10 \
                                   --num_train 50 \
                                   --layer relu4 \
                                   --ridge_alpha 0.5 \
                                   "${supervised_flag[@]}"
