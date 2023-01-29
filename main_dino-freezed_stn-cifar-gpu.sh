#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080 # partition (queue)
#SBATCH -t 23:59:59 # time (D-HH:MM:SS)
#SBATCH --gres=gpu:1
#SBATCH -D /work/dlclarge1/rapanti-stn_cifar/thetacropspenalty
#SBATCH --mail-type=ALL # (receive mails about end and timeouts/crashes of your job)
#SBATCH --array 0-30%1

# Print some information about the job to STDOUT
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

source /home/rapanti/.profile
source activate dino

EXP_D=/work/dlclarge1/rapanti-stn_cifar/experiments/
# Job to perform
torchrun \
  --standalone \
  --nproc_per_node=1 \
  --nnodes=1 \
  main_dino.py \
    --arch vit_small \
    --data_path /work/dlclarge1/rapanti-stn_cifar/data/datasets/CIFAR10 \
    --dataset CIFAR10 \
    --output_dir $EXP_D$EXPERIMENT_NAME \
    --epochs $EPOCHS \
    --warmup_epochs $WARMUP_EPOCHS \
    --batch_size_per_gpu $BATCH_SIZE \
    --stn_pretrained_weights $PRETRAINED_STN \
    --stn_mode $STN_MODE \
    --invert_stn_gradients True \
    --stn_theta_norm $TNORM \
    --local_crops_number 2 \
    --use_fp16 False \
    --use_bn True \
    --num_workers 8 \
    --saveckp_freq $SAVECKP_FREQ \
    --summary_writer_freq 750

# Print some Information about the end-time to STDOUT
echo "DONE"
echo "Finished at $(date)"
scancel $SLURM_ARRAY_JOB_ID
