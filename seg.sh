#!/bin/bash
# Run the training via CLI, adjust command as needed.
# Define experiment parameters
EXPERIMENT_NAME="my_experiment"
LOG_DIR=logs/
VERSION=seg

# Define checkpoint path
CKPT_PATH="$LOG_DIR/$EXPERIMENT_NAME/$VERSION/last.ckpt"

# Define monitoring and saving parameters
MONITOR_METRIC="avg_f1"
SAVE_TOP_K=5

# Define WandB parameters
export WANDB_API_KEY=""
export WANDB_NOTES=""
export WANDB_MODE="offline"

# Define class labels as a string
CLASS_LABELS="[background, farmland, water, forest, structure, meadow]"

# Check if checkpoint file exists
if [ ! -f "$CKPT_PATH" ]; then
    CKPT_PATH=null
fi

python3 main.py fit \
    "$@" \
    --ckpt_path=$CKPT_PATH \
    --model models.DeepLabV3 \
    --data data.BingRGB \
    --data.data_dir .data/data0 \
    --data.batch_size 4 \
    --data.val_batch_size 24 \
    --data.num_workers 4 \
    --model.class_labels="$CLASS_LABELS" \
    --optimizer AdamW \
    --optimizer.lr 1e-4 \
    --optimizer.weight_decay 1e-4 \
    --lr_scheduler ReduceLROnPlateau \
    --lr_scheduler.factor 0.95 \
    --lr_scheduler.patience 5 \
    --lr_scheduler.monitor $MONITOR_METRIC \
    --lr_scheduler.mode max \
    --trainer.max_epochs 100 \
    --trainer.callbacks+=BatchSizeFinder \
    --trainer.callbacks.max_trials=8 \
    --trainer.log_every_n_steps=5 \
    --trainer.callbacks+=ModelCheckpoint \
    --trainer.callbacks.filename="best_{epoch:02d}_{$MONITOR_METRIC:.2f}" \
    --trainer.callbacks.dirpath=$LOG_DIR/$EXPERIMENT_NAME/$VERSION \
    --trainer.callbacks.monitor=$MONITOR_METRIC \
    --trainer.callbacks.save_top_k=$SAVE_TOP_K \
    --trainer.callbacks.save_last=True \
    --trainer.callbacks.mode=max \
    --trainer.logger+=WandbLogger \
    --trainer.logger.save_dir=$LOG_DIR/$EXPERIMENT_NAME/$VERSION \
    --trainer.logger.name=$EXPERIMENT_NAME \
    --trainer.logger.version=$VERSION \
    --trainer.logger+=CSVLogger \
    --trainer.logger.save_dir=$LOG_DIR \
    --trainer.logger.name=$EXPERIMENT_NAME \
    --trainer.logger.version=$VERSION \
    # --trainer.limit_train_batches=100 \