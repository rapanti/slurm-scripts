#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced # partition (queue)
#SBATCH -t 23:59:59 # time (D-HH:MM:SS)
#SBATCH --gres=gpu:4
#SBATCH -D /work/dlclarge2/rapanti-metassl-dino-stn/wo-hflip
#SBATCH -J vanilla-dino-wo-hflip # sets the job name. If not specified, the file name will be used as job name
#SBATCH -o /work/dlclarge2/rapanti-metassl-dino-stn/experiments/vanilla-dino-wo-hflip/log/%A.%a.%N.out  # STDOUT
#SBATCH -e /work/dlclarge2/rapanti-metassl-dino-stn/experiments/vanilla-dino-wo-hflip/log/%A.%a.%N.out  # STDERR
#SBATCH --array 0-31%1

# Print some information about the job to STDOUT
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

source /home/rapanti/.profile
source activate dino

EXP_D=/work/dlclarge2/rapanti-metassl-dino-stn/experiments/vanilla-dino-wo-hflip

# Job to perform
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --standalone \
    run_train_eval.py \
      --arch vit_nano \
      --img_size 32 \
      --patch_size 4 \
      --out_dim 32768 \
      --data_path /work/dlclarge2/rapanti-metassl-dino-stn/datasets/CIFAR10 \
      --dataset CIFAR10 \
      --output_dir $EXP_D \
      --epochs 300 \
      --warmup_epoch 30 \
      --batch_size_per_gpu 64 \
      --use_fp16 false \
      --saveckp_freq 100

x=$?
if [ $x == 0 ]
then
  scancel "$SLURM_JOB_ID"
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
