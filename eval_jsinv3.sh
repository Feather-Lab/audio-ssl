#!/bin/bash -l
#SBATCH --job-name=eval_jsin
#SBATCH --output=outLogs/eval_jsin_%A_%a.out
#SBATCH --error=outLogs/eval_jsin_%A_%a.err
#SBATCH --cpus-per-gpu=10
#SBATCH --gpus=1

#SBATCH --mem=16Gb
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=h100  # if you want a particular type of GPU
##SBATCH --array=1-7 #0-7 for current 

module load cuda cudnn nccl

conda activate ~/ceph/conda_envs/cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE

# python3 lightning_scripts/eval_jsin.py --config lightning_scripts/configs/word_audioset_resnet50.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192


# python3 lightning_scripts/eval_jsin.py --config lightning_scripts/configs/word_audioset_resnet50_lower_lr.yaml \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192

python3 lightning_scripts/eval_jsin.py --config_path model_configs/word_audioset_resnet50_lower_lr_slower_schedule_5.yaml \
                                   --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --model_ckpt_dir model_checkpoints \
                                   --batch_size 192 \
                                   --ckpt_path model_checkpoints/word_audioset_resnet50_lower_lr_slower_schedule_5/checkpoints/epoch=5-step=45300-best_word_task.ckpt

# python3 lightning_scripts/eval_jsin.py --config_list_path eval_config_manifests/all_config_eval_list_09_2024.pkl \
#                                    --gpus $num_gpus --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --batch_size 192 \
#                                    --array_ix $SLURM_ARRAY_TASK_ID

