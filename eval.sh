#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced # partition (queue)
#SBATCH -t 23:59:59 # time (D-HH:MM:SS)
#SBATCH --gres=gpu:4
#SBATCH -D /work/dlclarge1/rapanti-stn_cifar/dino_cifar10
#SBATCH -J eval_linear-stn-translation_scale_symmetric-thetacropspenalty-pretrain-exp3
#SBATCH -o /work/dlclarge1/rapanti-stn_cifar/experiments/eval_linear-stn-translation_scale_symmetric-thetacropspenalty-pretrain-exp3/log/%A.%a.%N.out
#SBATCH -e /work/dlclarge1/rapanti-stn_cifar/experiments/eval_linear-stn-translation_scale_symmetric-thetacropspenalty-pretrain-exp3/log/%A.%a.%N.out

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source /home/rapanti/.profile
source activate dino

WEIGHTS=dino-stn-translation_scale_symmetric-thetacropspenalty-pretrain-exp3/checkpoint.pth
EXP_D=/work/dlclarge1/rapanti-stn_cifar/experiments/eval_linear-stn-translation_scale_symmetric-thetacropspenalty-pretrain-exp3
# Job to perform
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --standalone \
  eval_linear.py \
    --arch vit_nano \
    --dataset CIFAR10 \
    --data_path /work/dlclarge1/rapanti-stn_cifar/data/datasets/CIFAR10 \
    --pretrained_weights /work/dlclarge1/rapanti-stn_cifar/experiments/$WEIGHTS \
    --img_size 224 \
    --patch_size 16 \
    --output_dir $EXP_D \
    --epochs 100 \
    --batch_size_per_gpu 192 \
    --lr 0.01 \
    --weight_decay 0 \
    --num_workers 8 \
    --val_freq 10

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
