#!/bin/bash -l
#SBATCH --job-name=eval_esc50
#SBATCH --output=outLogs/eval_esc50_%A_%a.out
#SBATCH --error=outLogs/eval_esc50_%A_%a.err
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1

#SBATCH --mem=32Gb
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --array=0#-3 #0-53 for current 

# module load cuda cudnn nccl

mamba activate cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE

python3 lightning_scripts/make_esc_pl_model_plots.py --config_path model_configs/barlow_audioset_resnet18_rand_crop.yaml \
                                   -D /tmp/igriffith -L -3 -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
                                   --model_ckpt_dir model_checkpoints \
                                #    --ckpt_path model_checkpoints/pilot_ssl_mmcr_dualtask_resnet50_hparam_set_1_lr_06/checkpoints/epoch=8-step=33750.ckpt

# python3 lightning_scripts/make_esc_pl_model_plots.py --config_path model_configs/pilot_ssl_barlow_dualtask_resnet50_hparam_set_13_lr_02_LARS_MatchedSpeechInNoiseDatasetBatched.yaml \
#                                    -D /tmp/igriffith -L -1 -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
#                                    --model_ckpt_dir model_checkpoints \
#                                 #    --ckpt_path model_checkpoints/pilot_ssl_mmcr_dualtask_resnet50_hparam_set_1_lr_06/checkpoints/epoch=8-step=33750.ckpt

# python3 lightning_scripts/make_esc_pl_model_plots.py --config_path model_configs/pilot_ssl_barlow_dualtask_resnet50_hparam_set_13_lr_06_LARS_MatchedSpeechInNoiseDatasetBatched.yaml \
#                                    -D /tmp/igriffith -L -1 -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --ckpt_path model_checkpoints/pilot_ssl_barlow_dualtask_resnet50_hparam_set_13_lr_06_LARS_MatchedSpeechInNoiseDatasetBatched/checkpoints/epoch=31-step=9600-v1.ckpt


# python3 lightning_scripts/make_esc_pl_model_plots.py --config_path model_configs/pilot_ssl_mmcr_dualtask_resnet50_hparam_set_1_lr_02_LARS_MatchedSpeechInNoiseDatasetBatched.yaml \
#                                    -D /tmp/igriffith -L -1 -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --ckpt_path model_checkpoints/pilot_ssl_mmcr_dualtask_resnet50_hparam_set_1_lr_02_LARS_MatchedSpeechInNoiseDatasetBatched/checkpoints/epoch=8-step=2700.ckpt


# python3 lightning_scripts/make_esc_pl_model_plots.py --config_path model_configs/pilot_ssl_barlow_dualtask_resnet50_hparam_set_1_lr_02_LARS_MatchedSpeechInNoiseDatasetBatched.yaml \
#                                    -D /tmp/igriffith -L -1 -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
#                                    --model_ckpt_dir model_checkpoints \
#                                 #    --ckpt_path model_checkpoints/pilot_ssl_mmcr_dualtask_resnet50_hparam_set_1_lr_06/checkpoints/epoch=8-step=33750.ckpt


# python3 lightning_scripts/make_esc_pl_model_plots.py --config_path model_configs/pilot_ssl_barlow_dualtask_resnet50_hparam_set_1_lr_02_LARS_MatchedSpeechInNoiseDatasetBatched.yaml \
#                                    -D /tmp/igriffith -L -4 -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100 \
#                                    --model_ckpt_dir model_checkpoints \
#                                    --ckpt_path model_checkpoints/pilot_ssl_barlow_dualtask_resnet50_hparam_set_1_lr_02_LARS_MatchedSpeechInNoiseDatasetBatched/checkpoints/epoch=24-step=7500-v1.ckpt


