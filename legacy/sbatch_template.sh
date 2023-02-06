#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080 # partition (queue)
#SBATCH -t 23:59:59 # time (D-HH:MM:SS)
#SBATCH --gres=gpu:8
#SBATCH -J dino_stn_cifar10 # sets the job name. If not specified, the file name will be used as job name
#SBATCH -D /work/dlclarge1/rapanti-stn_cifar/metassl-dino-rpn
#SBATCH -o /work/dlclarge1/rapanti-stn_cifar/experiments/dino_cifar10_baseline/log/%x.%A.%a.%N.out  # STDOUT
#SBATCH -e /work/dlclarge1/rapanti-stn_cifar/experiments/dino_cifar10_baseline/log/%x.%A.%a.%N.out  # STDERR
#SBATCH --mail-type=END,FAIL # ALL (receive mails about end and timeouts/crashes of your job)
#SBATCH --array 0-30%1

# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source /home/rapanti/.profile
source activate dino

EXP_D=/work/dlclarge1/rapanti-stn_cifar/experiments/
# Job to perform
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --standalone \
    main_dino.py \
        --arch vit_small \
        --data_path /work/dlclarge1/rapanti-stn_cifar/data/datasets/CIFAR10/train \
        --dataset CIFAR10 \
        --output_dir $EXP_D$EXPERIMENT_NAME \
        --epochs $EPOCHS \
        --batch_size_per_gpu $BATCH_SIZE \
        --invert_rpn_gradients $INVERT_GRADIENTS \
        --stn_mode $STN_MODE \
        --similarity_penalty $SIMILARITY_PENALTY \
        --local_crops_number 2 \
        --use_fp16 True \
        --use_bn True \
        --saveckp_freq 20 \
        --summary_writer_freq 500

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
scancel $SLURM_ARRAY_JOB_ID
