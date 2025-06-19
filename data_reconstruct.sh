#!/bin/bash

# Common settings for the super-resolution model (if used)
MODEL_ARCH="reconstruction_base_patch250_250"
LAYER_SCALE=0.1
CHECKPOINT_BASE_PATH="./checkpoints" # Base path to user_X folders
SR_EPOCH=1999 # Epoch of the SR model to load

# Path to your python interpreter
PYTHON_INTERP="/home/daibo/.conda/envs/xinyu/bin/python"
SCRIPT_NAME="train_ssvep_tdca_test_channel_attack_multi_time_n_fold.py"

# Common device
DEVICE="cuda:0"


# Experiment 1: 15-channel direct classification with 8 noisy channels
echo "Starting: 15-channel direct classification with 8 noisy channels (background)"
$PYTHON_INTERP -u $SCRIPT_NAME \
    --model $MODEL_ARCH \
    --layer_scale_init_value $LAYER_SCALE \
    --checkpoint_base_path $CHECKPOINT_BASE_PATH \
    --checkpoint_epoch $SR_EPOCH \
    --device $DEVICE \
    --experiment_mode "7channel_direct" \
    --target_channels_for_tdca "15" > 15ch_8noisy_direct_output.log 2>&1 &

echo "Started: 15-channel direct classification with 8 noisy channels"
echo "------------------------------------------"

# # Experiment 2: 15-channel direct classification (Clean)
# echo "Starting: 15-channel direct classification (Clean) (background)"
# $PYTHON_INTERP -u $SCRIPT_NAME \
#     --model $MODEL_ARCH \
#     --layer_scale_init_value $LAYER_SCALE \
#     --checkpoint_base_path $CHECKPOINT_BASE_PATH \
#     --checkpoint_epoch $SR_EPOCH \
#     --device $DEVICE \
#     --experiment_mode "15channel_direct" \
#     --target_channels_for_tdca "15" > 15ch_direct_output.log 2>&1 &

# echo "Started: 15-channel direct classification (Clean)"
# echo "------------------------------------------"

# # Experiment 3: 7-channel to 15-channel reconstruction then classification
# echo "Starting: 7-to-15 channel reconstruction and classification (background)"
# $PYTHON_INTERP -u $SCRIPT_NAME \
#     --model $MODEL_ARCH \
#     --layer_scale_init_value $LAYER_SCALE \
#     --checkpoint_base_path $CHECKPOINT_BASE_PATH \
#     --checkpoint_epoch $SR_EPOCH \
#     --device $DEVICE \
#     --experiment_mode "7to15_reconstruct" \
#     --target_channels_for_tdca "15" > 7to15_recon_output.log 2>&1 &

# echo "Started: 7-to-15 channel reconstruction and classification"
# echo "------------------------------------------"

# echo "All experiments launched in background. Waiting for completion..."
# wait

# echo "All experiments completed."