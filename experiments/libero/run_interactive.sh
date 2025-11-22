#!/bin/bash
# Quick script to run on interactive node with all environment variables set
# Set environment variables BEFORE Python imports anything

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DISPLAY=""
export MUJOCO_EGL_DEVICE_ID=0  # Critical! Prevents MIG UUID parsing errors
export HF_HOME=/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

# Set offline mode BEFORE any Python imports
# These MUST be set before transformers/huggingface_hub are imported
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Disable tokenizer parallelism to avoid forking issues that can hang on compute nodes
export TOKENIZERS_PARALLELISM=false

# Set network timeouts to fail fast instead of hanging (in case something tries to access internet)
export PYTHONUNBUFFERED=1

# Verify environment variables are set
echo "Environment variables set:"
echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "  TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "  HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
echo "  TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
echo "  MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
echo ""

cd /scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/experiments/libero
python tersoo_run_libero_eval_scratch.py --task spatial --task_id 1 --checkpoint allenai/MolmoAct-7B-D-LIBERO-Spatial-0812
