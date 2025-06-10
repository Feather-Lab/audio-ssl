#!/bin/bash -l
#SBATCH --job-name=eval_esc50
#SBATCH --output=outLogs/eval_esc50_%A_%a.out
#SBATCH --error=outLogs/eval_esc50_%A_%a.err
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1

#SBATCH --mem=200Gb
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --array=0-20 #0-19 for kell

# module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE

# python3 lightning_scripts/make_esc_pl_model_plots.py --config_path model_configs/audioset_resnet18_MatchedDataset_LARS.yaml \
#                                    -D /tmp/igriffith -L -1 -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
#                                    --model_ckpt_dir model_checkpoints \

python3 lightning_scripts/make_esc_pl_model_plots.py --config_path model_configs/barlow_dualtask_kell2018_base_Matched_blocked_batches_lmbda_1e-2_lr_2e-1_w_augment_eq_lmbda_5e-1_fixed_loss.yaml \
                                   -D /tmp/igriffith -L $SLURM_ARRAY_TASK_ID -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
                                   --model_ckpt_dir model_checkpoints \

# python3 lightning_scripts/make_esc_pl_model_plots.py --config_path model_configs/resnet18_barlow_invariant_only_lmbda_1e-2_lr_2e-1_w_invar_augment_no_avgpool.yaml \
#                                    -D /tmp/igriffith -L $SLURM_ARRAY_TASK_ID -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
#                                    --model_ckpt_dir model_checkpoints \

# python3 lightning_scripts/make_esc_pl_model_plots.py --config_path model_configs/resnet18_barlow_equivariant_lmbda_1e-2_lr_2e-1_w_invar_augment_no_avgpool_eq_lmbda_1e-02.yaml \
#                                    -D /tmp/igriffith -L $SLURM_ARRAY_TASK_ID -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
#                                    --model_ckpt_dir model_checkpoints \
