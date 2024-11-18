#!/bin/bash -l
#SBATCH --job-name=eval_esc50
#SBATCH --output=outLogs/eval_esc50_%A_%a.out
#SBATCH --error=outLogs/eval_esc50_%A_%a.err
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1

#SBATCH --mem=12Gb
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --array=0-3 #0-53 for current 

# module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE

# python3 lightning_scripts/make_esc_pl_model_plots.py --config_path lightning_scripts/configs/word_audioset_resnet50.yaml \
#                                    -D /tmp/igriffith -L -4 -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
#                                    --model_ckpt_dir model_checkpoints \

# python3 lightning_scripts/make_esc_pl_model_plots.py --config_path lightning_scripts/configs/word_audioset_resnet50.yaml \
#                                    -D /tmp/igriffith -L -4 -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
#                                    --model_ckpt_dir model_checkpoints \


# python3 lightning_scripts/make_esc_pl_model_plots.py --config_path lightning_scripts/configs/word_audioset_resnet50_lower_lr.yaml \
#                                    -D /tmp/igriffith -L -4 -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
#                                    --model_ckpt_dir model_checkpoints \

# python3 lightning_scripts/make_esc_pl_model_plots.py --config_path lightning_scripts/configs/word_audioset_resnet50_lower_lr_slower_schedule_lower_task_weight.yaml \
#                                    -D /tmp/igriffith -L -4 -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
#                                    --model_ckpt_dir model_checkpoints \

# python3 lightning_scripts/make_esc_pl_model_plots.py --config_path model_configs/word_audioset_resnet50_lower_lr_slower_schedule_5.yaml \
#                                    -D /tmp/igriffith -L -4 -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
#                                    --model_ckpt_dir model_checkpoints \
#                                 --ckpt_path model_checkpoints/word_audioset_resnet50_lower_lr_slower_schedule_5/checkpoints/epoch=5-step=45300-best_word_task.ckpt

# python3 lightning_scripts/make_esc_pl_model_plots.py --config_path model_configs/audioset_resnet50.yaml \
#                                    -D /tmp/igriffith -L -4 -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
#                                    --model_ckpt_dir model_checkpoints \
#                                 --ckpt_path model_checkpoints/audioset_resnet50/checkpoints/epoch=4-step=37750-best_audioset_task.ckpt


# python3 lightning_scripts/make_esc_pl_model_plots.py --config_path model_configs/pilot_ssl_mmcr_dualtask_resnet50_hparam_set_1_lr_06.yaml \
#                                    -D /tmp/igriffith -L -1 -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --ckpt_path model_checkpoints/pilot_ssl_mmcr_dualtask_resnet50_hparam_set_1_lr_06/checkpoints/epoch=8-step=33750.ckpt


python3 lightning_scripts/make_esc_pl_model_plots.py --config_list_path train_config_manifests/pilot_dualtask_learning_rate_barlow_768-batch-size.pkl \
                                   -D /tmp/igriffith -L -1 -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
                                   --model_ckpt_dir model_checkpoints \
                                   --array_ix $SLURM_ARRAY_TASK_ID

