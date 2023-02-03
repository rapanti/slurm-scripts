#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced # partition (queue)
#SBATCH -t 23:59:59 # time (D-HH:MM:SS)
#SBATCH --gres=gpu:4
#SBATCH -D /work/dlclarge1/rapanti-stn_cifar/dino_cifar10
#SBATCH -J eval_linear-horizontalflip-experiment-ep300-pretrain7
#SBATCH -o /work/dlclarge1/rapanti-stn_cifar/experiments/eval_linear-horizontalflip-experiment-ep300-pretrain7/log/%A.%a.%N.out
#SBATCH -e /work/dlclarge1/rapanti-stn_cifar/experiments/eval_linear-horizontalflip-experiment-ep300-pretrain7/log/%A.%a.%N.out

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source /home/rapanti/.profile
source activate dino

WEIGHTS=dino-horizontalflip-experiment-ep300-pretrain7/checkpoint.pth
EXP_D=/work/dlclarge1/rapanti-stn_cifar/experiments/eval_linear-horizontalflip-experiment-ep300-pretrain7
# Job to perform
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --standalone \
  eval_linear.py \
    --arch vit_96  \
    --dataset CIFAR10 \
    --data_path /work/dlclarge1/rapanti-stn_cifar/data/datasets/CIFAR10 \
    --pretrained_weights /work/dlclarge1/rapanti-stn_cifar/experiments/$WEIGHTS \
    --img_size 32 \
    --patch_size 4 \
    --output_dir $EXP_D \
    --epochs 300 \
    --batch_size_per_gpu 192 \
    --lr 0.01 \
    --weight_decay 0 \
    --num_workers 8 \
    --val_freq 10

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
