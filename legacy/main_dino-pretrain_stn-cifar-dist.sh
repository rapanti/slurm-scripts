#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080 # partition (queue)
#SBATCH -t 23:59:59 # time (D-HH:MM:SS)
#SBATCH --gres=gpu:8
#SBATCH -D /work/dlclarge1/rapanti-stn_cifar/dino
#SBATCH -o /work/dlclarge1/rapanti-stn_cifar/experiments/dino_cifar10_baseline/log/%x.%A.%a.%N.out  # STDOUT
#SBATCH -e /work/dlclarge1/rapanti-stn_cifar/experiments/dino_cifar10_baseline/log/%x.%A.%a.%N.out  # STDERR
#SBATCH --mail-type=END,FAIL # (receive mails about end and timeouts/crashes of your job)
#SBATCH --array 0-30%1

# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source /home/rapanti/.profile
source activate dino

# Job to perform
torchrun --nproc_per_node=8 --nnodes=1 \
    main_dino.py \
        --arch vit_small \
        --batch_size_per_gpu 48 \
        --data_path /work/dlclarge1/rapanti-stn_cifar/data/datasets/CIFAR10/train \
        --output_dir /work/dlclarge1/rapanti-stn_cifar/experiments/dino_cifar10_baseline \
        --saveckp_freq 20 \
        --lr 5e-5 \
        --min_lr 1e-6 \
        --weight_decay 0.05 \
        --warmup_epochs 5 \
        --use_fp16 False \
        --epochs 1000

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
scancel $SLURM_ARRAY_JOB_ID
