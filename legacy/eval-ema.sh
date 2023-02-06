#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced # partition (queue)
#SBATCH -t 23:59:59 # time (D-HH:MM:SS)
#SBATCH --gres=gpu:4
#SBATCH -D /work/dlclarge2/rapanti-metassl-dino-stn/ema
#SBATCH -J eval_linear-pretrain-stn_with_ema-vit_nano-testrun
#SBATCH -o /work/dlclarge2/rapanti-metassl-dino-stn/experiments/eval_linear-pretrain-stn_with_ema-vit_nano-testrun/log/%A.%a.%N.out
#SBATCH -e /work/dlclarge2/rapanti-metassl-dino-stn/experiments/eval_linear-pretrain-stn_with_ema-vit_nano-testrun/log/%A.%a.%N.out
/work/dlclarge2/rapanti-metassl-dino-stn/experiments/pretrain-stn_with_ema-vit_nano-testrun/log
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source /home/rapanti/.profile
source activate dino

WEIGHTS=/work/dlclarge2/rapanti-metassl-dino-stn/experiments/pretrain-stn_with_ema-vit_nano-testrun/checkpoint.pth
EXP_D=/work/dlclarge2/rapanti-metassl-dino-stn/experiments/eval_linear-pretrain-stn_with_ema-vit_nano-testrun
# Job to perform
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --standalone \
  eval_linear.py \
    --arch vit_nano  \
    --dataset CIFAR10 \
    --data_path /work/dlclarge2/rapanti-metassl-dino-stn/datasets/CIFAR10 \
    --pretrained_weights $WEIGHTS \
    --img_size 32 \
    --patch_size 4 \
    --output_dir $EXP_D \
    --epochs 300 \
    --batch_size 768 \
    --lr 0.01 \
    --weight_decay 0 \
    --num_workers 8 \
    --val_freq 10

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
