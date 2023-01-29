#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced # partition (queue)
#SBATCH -t 23:59:59 # time (D-HH:MM:SS)
#SBATCH --gres=gpu:4
#SBATCH -J dino-stn_tss-vit_nano-60_30_6-tcp_eps100_rand-caugm-pretrain-ep300 # sets the job name. If not specified, the file name will be used as job name
#SBATCH -D /work/dlclarge1/rapanti-stn_cifar/metassl-dino-rpn
#SBATCH -o /work/dlclarge1/rapanti-stn_cifar/experiments/dino-stn_tss-vit_nano-60_30_6-tcp_eps100_rand-caugm-pretrain-ep300/log/%A.%a.%N.out  # STDOUT
#SBATCH -e /work/dlclarge1/rapanti-stn_cifar/experiments/dino-stn_tss-vit_nano-60_30_6-tcp_eps100_rand-caugm-pretrain-ep300/log/%A.%a.%N.out  # STDERR
#SBATCH --array 0-32%1

# Print some information about the job to STDOUT
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

source /home/rapanti/.profile
source activate dino

EXP_D=/work/dlclarge1/rapanti-stn_cifar/experiments/dino-stn_tss-vit_nano-60_30_6-tcp_eps100_rand-caugm-pretrain-ep300
# Job to perform
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --standalone \
    main_dino.py \
      --arch vit_nano \
      --img_size 60 \
      --patch_size 6 \
      --stn_res 60 30 \
      --out_dim 32768 \
      --data_path /work/dlclarge1/rapanti-stn_cifar/data/datasets/CIFAR10 \
      --dataset CIFAR10 \
      --output_dir $EXP_D \
      --epochs 300 \
      --warmup_epoch 30 \
      --batch_size_per_gpu 64 \
      --invert_stn_gradients true \
      --stn_theta_norm true \
      --use_unbounded_stn true \
      --stn_mode translation_scale_symmetric \
      --use_stn_penalty true \
      --invert_penalty true \
      --penalty_loss thetacropspenalty \
      --epsilon 100 \
      --local_crops_number 8 \
      --local_crops_scale 0.05 0.4 \
      --global_crops_scale 0.4 1 \
      --stn_color_augment true \
      --use_fp16 false \
      --saveckp_freq 100 \
      --summary_writer_freq 400

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
