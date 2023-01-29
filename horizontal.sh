#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced # partition (queue)
#SBATCH -t 23:59:59 # time (D-HH:MM:SS)
#SBATCH --gres=gpu:8
#SBATCH -J dino-horizontalflip-experiment-ep300-pretrain3 # sets the job name. If not specified, the file name will be used as job name
#SBATCH -D /work/dlclarge1/rapanti-stn_cifar/horizontalflip
#SBATCH -o /work/dlclarge1/rapanti-stn_cifar/experiments/dino-horizontalflip-experiment-ep300-pretrain3/log/%A.%a.%N.out  # STDOUT
#SBATCH -e /work/dlclarge1/rapanti-stn_cifar/experiments/dino-horizontalflip-experiment-ep300-pretrain3/log/%A.%a.%N.out  # STDERR

# Print some information about the job to STDOUT
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

source /home/rapanti/.profile
source activate dino

EXP_D=/work/dlclarge1/rapanti-stn_cifar/experiments/dino-horizontalflip-experiment-ep300-pretrain3
# Job to perform
torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --standalone \
    main_dino.py \
      --arch vit_nano \
      --img_size 32 \
      --patch_size 4 \
      --stn_res 32 16 \
      --out_dim 32768 \
      --data_path /work/dlclarge1/rapanti-stn_cifar/data/datasets/CIFAR10 \
      --dataset CIFAR10 \
      --output_dir $EXP_D \
      --epochs 300 \
      --warmup_epoch 30 \
      --batch_size_per_gpu 32 \
      --invert_stn_gradients true \
      --stn_theta_norm true \
      --stn_mode rotation \
      --local_crops_number 8 \
      --stn_color_augment true \
      --use_fp16 true \
      --saveckp_freq 100 \
      --global_crops_scale 0.7 1 \
      --local_crops_scale 0.4 0.7 \
      --summary_writer_freq 200

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
