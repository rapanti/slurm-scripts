#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080 # partition (queue)
#SBATCH -t 23:59:59 # time (D-HH:MM:SS)
#SBATCH --gres=gpu:1
#SBATCH -D /work/dlclarge1/rapanti-stn_cifar/thetacropspenalty
#SBATCH --mail-type=END,FAIL # (receive mails about end and timeouts/crashes of your job)
#SBATCH --array 0-30%1
# SBATCH -J metassl_dino_stn_cifar_linear-eval # sets the job name. If not specified, the file name will be used as job name

# Print some information about the job to STDOUT
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

source /home/rapanti/.profile
source activate dino

EXP_D=/work/dlclarge1/rapanti-stn_cifar/experiments/
# Job to perform
python simple_eval_linear.py \
    --dataset ImageNet \
    --arch vit_small \
    --batch_size_per_gpu $BATCH_SIZE \
    --lr $LR \
    --weight_decay $WD \
    --data_path /work/dlclarge1/rapanti-stn_cifar/data/datasets/CIFAR10 \
    --output_dir $EXP_D$EXPERIMENT_NAME \
    --pretrained_weights $PRETRAINED_WEIGHTS \
    --num_labels 10 \
    --num_workers 8 \
    --val_freq 10 \
    --epochs $EPOCHS

# Print some Information about the end-time to STDOUT
echo "DONE"
echo "Finished at $(date)"
scancel $SLURM_ARRAY_JOB_ID
