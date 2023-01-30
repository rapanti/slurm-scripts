set export

# List available just commands
@list:
  just --list

# ---------------------------------------------------------------------------------------
# BASELINES - SIMSIAM ON CIFAR10
# ---------------------------------------------------------------------------------------

@vanilla_dino-gpu EXPERIMENT_NAME BATCH_SIZE EPOCHS WARMUP_EPOCHS SAVECKP_FREQ:
    #!/usr/bin/env bash
    EXP_DIR=/work/dlclarge1/rapanti-stn_cifar/experiments
    mkdir -p $EXP_DIR/{{EXPERIMENT_NAME}}/log/
    sbatch --output=$EXP_DIR/{{EXPERIMENT_NAME}}/log/%x.%A.%a.%N.out \
        --error=$EXP_DIR/{{EXPERIMENT_NAME}}/log/%x.%A.%a.%N.out \
        --job-name={{EXPERIMENT_NAME}} \
        --export=ALL \
        vanilla_dino-cifar-gpu.sh

@main_dino-pretrain_stn-cifar-gpu EXPERIMENT_NAME BATCH_SIZE EPOCHS WARMUP_EPOCHS SAVECKP_FREQ STN_MODE TNORM USE_STN_PENALTY PENALTY_LOSS INVERT_PENALTY EPSILON:
    #!/usr/bin/env bash
    EXP_DIR=/work/dlclarge1/rapanti-stn_cifar/experiments
    mkdir -p $EXP_DIR/{{EXPERIMENT_NAME}}/log/
    sbatch --output=$EXP_DIR/{{EXPERIMENT_NAME}}/log/%x.%A.%a.%N.out \
        --error=$EXP_DIR/{{EXPERIMENT_NAME}}/log/%x.%A.%a.%N.out \
        --job-name={{EXPERIMENT_NAME}} \
        --export=ALL \
        main_dino-pretrain_stn-cifar-gpu.sh

@main_dino-freezed_stn-cifar-gpu EXPERIMENT_NAME BATCH_SIZE EPOCHS WARMUP_EPOCHS SAVECKP_FREQ PRETRAINED_STN STN_MODE TNORM:
    #!/usr/bin/env bash
    EXP_DIR=/work/dlclarge1/rapanti-stn_cifar/experiments
    mkdir -p $EXP_DIR/{{EXPERIMENT_NAME}}/log/
    sbatch --output=$EXP_DIR/{{EXPERIMENT_NAME}}/log/%x.%A.%a.%N.out \
        --error=$EXP_DIR/{{EXPERIMENT_NAME}}/log/%x.%A.%a.%N.out \
        --job-name={{EXPERIMENT_NAME}} \
        --export=ALL \
        main_dino-freezed_stn-cifar-gpu.sh

@eval_linear-gpu EXPERIMENT_NAME BATCH_SIZE LR WD EPOCHS PRETRAINED_WEIGHTS:
    #!/usr/bin/env bash
    EXP_DIR=/work/dlclarge1/rapanti-stn_cifar/experiments
    mkdir -p $EXP_DIR/{{EXPERIMENT_NAME}}/log/
    sbatch --output=$EXP_DIR/{{EXPERIMENT_NAME}}/log/%x.%A.%a.%N.out \
        --error=$EXP_DIR/{{EXPERIMENT_NAME}}/log/%x.%A.%a.%N.out \
        --job-name={{EXPERIMENT_NAME}} \
        --export=ALL \
        save-eval_linear-gpu.sh
