#!/bin/bash

# Debugging script wrapper for get_focus_object_from_pixel_coords
# This script activates the molmoact environment and runs the debug script

set -e

# Change to script directory
cd /scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/experiments/libero


# Set environment variables
export MOLMOACT_DATA_DIR=/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data
export HF_HOME=/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

# Enable offline mode for GPU nodes without internet access
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FORCE_OFFLINE=1

# Optional: Override checkpoint or image path via environment variables
# export DEBUG_CHECKPOINT="allenai/MolmoAct-7B-D-LIBERO-Object-0812"
export DEBUG_IMAGE_PATH="/scratch/gpfs/TSILVER/el5267/molmoact/output_frame.png"
export DEBUG_PIXEL_COORDS="(159, 137)"

# Run the debug script
echo "Running debug script..."
python debug_focus_object.py "$@"

