#!/bin/bash
# Run the training via CLI, adjust command as needed.
# Define experiment parameters
EXPERIMENT_NAME="my_experiment"
LOG_DIR=logs/
VERSION=v1

# Define checkpoint path
CKPT_PATH="$LOG_DIR/$EXPERIMENT_NAME/$VERSION/last.ckpt"

# Define monitoring and saving parameters
MONITOR_METRIC="val_acc"
SAVE_TOP_K=5

# Define WandB parameters
export WANDB_API_KEY=""
export WANDB_NOTES=""
export WANDB_MODE="offline"

# Check if checkpoint file exists
if [ ! -f "$CKPT_PATH" ]; then
    CKPT_PATH=null
fi

python main.py fit \
    "$@" \
    --ckpt_path=$CKPT_PATH \
    --model models.UNet \
    --data data.CIFAR10 \
    --trainer.log_every_n_steps=5 \
    --trainer.callbacks+=BatchSizeFinder \
    --trainer.callbacks.max_trials=8 \
    --trainer.callbacks+=ModelCheckpoint \
    --trainer.callbacks.filename="best_{epoch:02d}_{$MONITOR_METRIC:.2f}" \
    --trainer.callbacks.dirpath=$LOG_DIR/$EXPERIMENT_NAME/$VERSION \
    --trainer.callbacks.monitor=$MONITOR_METRIC \
    --trainer.callbacks.save_top_k=$SAVE_TOP_K \
    --trainer.callbacks.save_last=True \
    --trainer.callbacks.mode=max \
    --trainer.logger+=WandbLogger \
    --trainer.logger.save_dir=$LOG_DIR \
    --trainer.logger.name=$EXPERIMENT_NAME \
    --trainer.logger.version=$VERSION \
    --trainer.logger+=CSVLogger \
    --trainer.logger.save_dir=$LOG_DIR \
    --trainer.logger.name=$EXPERIMENT_NAME \
    --trainer.logger.version=$VERSION \