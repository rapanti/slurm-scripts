#!/bin/bash
#SBATCH -p testdlc_gpu-rtx2080 # partition (queue)
#SBATCH -t 00:09:59 # time (D-HH:MM:SS)
#SBATCH --gres=gpu:4
#SBATCH -J TesT # sets the job name. If not specified, the file name will be used as job name
#SBATCH -D /work/dlclarge1/rapanti-stn_cifar/thetacropspenalty
#SBATCH -o /work/dlclarge1/rapanti-stn_cifar/experiments/test/log/%x.%A.%a.%N.out  # STDOUT
#SBATCH -e /work/dlclarge1/rapanti-stn_cifar/experiments/test/log/%x.%A.%a.%N.out  # STDERR
# SBATCH --mail-type=END,FAIL # ALL (receive mails about end and timeouts/crashes of your job)
# SBATCH --array 0-30%1

# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source /home/rapanti/.profile
source activate dino

EXP_D=/work/dlclarge1/rapanti-stn_cifar/experiments/test
# Job to perform
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --standalone \
    main_dino.py \
        --arch vit_small \
        --data_path /data/datasets/ImageNet/imagenet-pytorch \
        --dataset ImageNet \
        --output_dir $EXP_D \
        --batch_size_per_gpu 16 \
        --invert_stn_gradients true \
        --local_crops_number 2 \
        --resize_input true \
        --resize_size 512 \
        --use_fp16 true

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
scancel $SLURM_ARRAY_JOB_ID
