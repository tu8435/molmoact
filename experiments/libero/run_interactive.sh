#!/bin/bash
# Quick script to run on interactive node with all environment variables set

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DISPLAY=""
export MUJOCO_EGL_DEVICE_ID=0
export HF_HOME=/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/experiments/libero
python tersoo_run_libero_eval.py --task spatial --task_id 1 --checkpoint allenai/MolmoAct-7B-D-LIBERO-Spatial-0812
