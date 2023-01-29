#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced # partition (queue)
#SBATCH -t 23:59:59 # time (D-HH:MM:SS)
#SBATCH --gres=gpu:8
#SBATCH -J dino-vit_nano-stn_tss-32_16-lcrops8-lcmod-penalty-eps10-coloraug-ep100-pretrain # sets the job name. If not specified, the file name will be used as job name
#SBATCH -D /work/dlclarge1/rapanti-stn_cifar/thetacropspenalty
#SBATCH -o /work/dlclarge1/rapanti-stn_cifar/experiments/dino-vit_nano-stn_tss-32_16-lcrops8-lcmod-penalty-eps10-coloraug-ep100-pretrain/log/%x.%A.%a.%N.out  # STDOUT
#SBATCH -e /work/dlclarge1/rapanti-stn_cifar/experiments/dino-vit_nano-stn_tss-32_16-lcrops8-lcmod-penalty-eps10-coloraug-ep100-pretrain/log/%x.%A.%a.%N.out  # STDERR

# Print some information about the job to STDOUT
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

source /home/rapanti/.profile
source activate dino

EXP_D=/work/dlclarge1/rapanti-stn_cifar/experiments/dino-vit_nano-stn_tss-32_16-lcrops8-lcmod-penalty-eps10-coloraug-ep100-pretrain
# Job to perform
torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --standalone \
    main_dino.py \
      --arch vit_nano \
      --img_size 32 \
      --patch_size 8 \
      --stn_res 32 16 \
      --out_dim 32768 \
      --data_path /work/dlclarge1/rapanti-stn_cifar/data/datasets/CIFAR10 \
      --dataset CIFAR10 \
      --output_dir $EXP_D \
      --epochs 100 \
      --batch_size_per_gpu 256 \
      --invert_stn_gradients true \
      --stn_theta_norm true \
      --use_unbounded_stn true \
      --stn_mode translation_scale_symmetric \
      --use_stn_penalty true \
      --invert_penalty true \
      --penalty_loss thetacropspenalty \
      --epsilon 20 \
      --local_crops_number 8 \
      --local_crops_scale 0.2 0.6 \
      --global_crops_scale 0.6 1 \
      --stn_color_augment true \
      --use_fp16 true \
      --saveckp_freq 25 \
      --summary_writer_freq 50

torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --standalone \
    /work/dlclarge1/rapanti-stn_cifar/dino_cifar10/eval_linear.py \
      --arch vit_nano \
      --dataset CIFAR10 \
      --data_path /work/dlclarge1/rapanti-stn_cifar/data/datasets/CIFAR10 \
      --pretrained_weights /work/dlclarge1/rapanti-stn_cifar/experiments/eval_linear-vit_nano-stn_tss-32_16-lcrops8-lcmod-penalty-eps10-coloraug-ep100-pretrain/checkpoint.pth \
      --img_size 32 \
      --patch_size 4 \
      --output_dir $EXP_D \
      --epochs 300 \
      --batch_size_per_gpu 768 \
      --lr 0.01 \
      --weight_decay 0 \
      --num_workers 8 \
      --val_freq 10

# Print some Information about the end-time to STDOUT
echo "DONE"
echo "Finished at $(date)"
