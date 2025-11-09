#!/bin/bash -l
#SBATCH --job-name=measure_165_natsound_acts
#SBATCH --output=outLogs/measure_165_natsound_acts_%A_%a.out
#SBATCH --error=outLogs/measure_165_natsound_acts_%A_%a.err
#SBATCH --cpus-per-gpu=2
#SBATCH --gpus=1

#SBATCH --mem=8Gb
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH -N 1
##SBATCH --array=0# 1-5 # 0-5 in equivariant manifest

module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE


python3 fmri_analysis/measure_layer_activations_165_natural_sounds_lightning.py --config_path model_configs/resnet50_barlow_equivariant_lmbda_1e-2_lr_2e-1_eq_lmbda_5e-01.yaml \
                                #    --dir_name_modifier "latest_ckpt"





