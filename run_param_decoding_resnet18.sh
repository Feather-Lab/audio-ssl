#!/bin/bash -l
#SBATCH --job-name=param_decoding
#SBATCH --output=outLogs/param_decoding_%A_%a.out
#SBATCH --error=outLogs/param_decoding_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8

#SBATCH --mem=400Gb
#SBATCH --time=1:00:00 
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --constraint=a100-80gb  # if you want a particular type of GPU
#SBATCH --array=5-8 # -9 #  0-9 in manifest

module load cuda cudnn nccl
mamba activate cochdnn_ssl_pl

export PYTHONPATH=$PYTHONPATH:/mnt/home/igriffith/ceph/projects/cochdnn
master_node=$SLURMD_NODENAME

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
echo "Master: "$master_node" Local node: "$HOSTNAME" GPUs used: "$CUDA_VISIBLE_DEVICES" Total GPUs on that node: "$num_gpus" CPUs per node: "$SLURM_JOB_CPUS_PER_NODE

# srun python3 lightning_scripts/run_param_decoding.py  \
#                                    --num_workers $SLURM_JOB_CPUS_PER_NODE \
#                                    --batch_size 192 \
#                                    --num_eval 5 \
#                                    --num_train 25 \
#                                    --layer 'invar_head' \

srun -K python3 -u lightning_scripts/run_param_decoding_sep_db_and_filter.py  \
                                   --num_workers $SLURM_JOB_CPUS_PER_NODE \
                                   --batch_size 192 \
                                   --num_eval 10 \
                                   --num_train 50 \
                                   --job_id $SLURM_ARRAY_TASK_ID \
                                   --invar_model_config model_configs/resnet18_barlow_invariant_only_lmbda_1e-2_lr_2e-1_w_invar_augment_no_avgpool.yaml \
                                   --equi_model_config model_configs/resnet18_barlow_equivariant_lmbda_1e-2_lr_2e-1_w_invar_augment_no_avgpool_eq_lmbda_2e-01.yaml \
                                   --ridge_alpha 0.5
                                  #  --equi_model_ckpt model_checkpoints/resnet18_barlow_equivariant_lmbda_1e-2_lr_2e-1_no_avgpool_eq_lmbda_2e-01/checkpoints/epoch=49-step=11250-best_val.ckpt \
                                  #  --equi_model_config model_configs/resnet18_barlow_equivariant_lmbda_1e-2_lr_2e-1_w_invar_augment_no_avgpool_eq_lmbda_1e-01.yaml \
                                #    --layer 'avgpool' \
