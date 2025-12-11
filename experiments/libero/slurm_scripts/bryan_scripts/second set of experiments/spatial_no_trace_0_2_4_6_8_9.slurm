#!/bin/bash

#SBATCH --job-name=spatial_no_trace_0_2_4_6_8_9       # create a short name for your job
#SBATCH --nodes=1                    # node count
#SBATCH --ntasks=1                   # total number of tasks across all nodes
#SBATCH --cpus-per-task=2            # cpu-cores per task (increased for data loading)
#SBATCH --gres=gpu:1                 # request 1 GPU
#SBATCH --mem=32G                    # total memory per node
#SBATCH --time=16:00:00              # total run time limit (HH:MM:SS) -    16 hours and 0 minutes and 0 seconds
#SBATCH --mail-type=begin,end,fail   # receive email notifications
#SBATCH --mail-user=bb8404@princeton.edu  # replace with your email
#SBATCH --output=spatial_no_trace_0_2_4_6_8_9_%j.out        # output file name (%j is the job ID)
#SBATCH --error=spatial_no_trace_0_2_4_6_8_9_%j.err         # error file name (%j is the job ID)

# Change to the script directory
cd /scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/experiments/libero

# Load modules (we still need some system modules, but will use user's miniconda3)
module purge

# Initialize and use the user's miniconda3 installation directly
# The environment exists at: /scratch/gpfs/TSILVER/tu8435/miniconda3/envs/molmoact
export PATH="/scratch/gpfs/TSILVER/tu8435/miniconda3/bin:$PATH"

# Initialize conda for bash
eval "$(/scratch/gpfs/TSILVER/tu8435/miniconda3/bin/conda shell.bash hook)"

# Activate the molmoact environment
conda activate molmoact

# Set environment variables for better GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable display for offscreen rendering
export DISPLAY=""

# Set CUDA_VISIBLE_DEVICES to match MUJOCO_EGL_DEVICE_ID to avoid robosuite assertion error
# This must be set before robosuite is imported
# SLURM may set CUDA_VISIBLE_DEVICES, but we override it to ensure it matches MUJOCO_EGL_DEVICE_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set MUJOCO_EGL_DEVICE_ID to 0 to avoid MIG UUID parsing issues in robosuite
# This is checked BEFORE CUDA_VISIBLE_DEVICES in robosuite's EGL context code
# Setting this to 0 ensures robosuite uses device 0 instead of trying to parse CUDA_VISIBLE_DEVICES
# Note: robosuite asserts that MUJOCO_EGL_DEVICE_ID must be in CUDA_VISIBLE_DEVICES
export MUJOCO_EGL_DEVICE_ID=0

# Set HuggingFace cache directory (compute nodes don't have internet)
# HF_HOME should point to the directory containing 'hub/', not the hub directory itself
export HF_HOME=/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

# Set offline mode to prevent attempting downloads (must be set before Python imports)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Disable tokenizer parallelism to avoid forking issues that can hang on compute nodes
# The forking warning in oct30.txt suggests this might be causing hangs on compute nodes
export TOKENIZERS_PARALLELISM=false

# Enable unbuffered output for better logging
export PYTHONUNBUFFERED=1

# Run the evaluation script for task IDs 0, 2, 4, 6, 8, 9
# --include_trace False: Disable visual trace generation (ablation study)
# Using task-specific checkpoint for spatial tasks
python tersoo_run_libero_eval_long.py --task spatial --task_id 0 --checkpoint allenai/MolmoAct-7B-D-LIBERO-Spatial-0812 --include_trace False --base_dir /scratch/gpfs/TSILVER/bb8404/molmoact/experiments/libero/rollouts
python tersoo_run_libero_eval_long.py --task spatial --task_id 2 --checkpoint allenai/MolmoAct-7B-D-LIBERO-Spatial-0812 --include_trace False --base_dir /scratch/gpfs/TSILVER/bb8404/molmoact/experiments/libero/rollouts
python tersoo_run_libero_eval_long.py --task spatial --task_id 4 --checkpoint allenai/MolmoAct-7B-D-LIBERO-Spatial-0812 --include_trace False --base_dir /scratch/gpfs/TSILVER/bb8404/molmoact/experiments/libero/rollouts
python tersoo_run_libero_eval_long.py --task spatial --task_id 6 --checkpoint allenai/MolmoAct-7B-D-LIBERO-Spatial-0812 --include_trace False --base_dir /scratch/gpfs/TSILVER/bb8404/molmoact/experiments/libero/rollouts
python tersoo_run_libero_eval_long.py --task spatial --task_id 8 --checkpoint allenai/MolmoAct-7B-D-LIBERO-Spatial-0812 --include_trace False --base_dir /scratch/gpfs/TSILVER/bb8404/molmoact/experiments/libero/rollouts
python tersoo_run_libero_eval_long.py --task spatial --task_id 9 --checkpoint allenai/MolmoAct-7B-D-LIBERO-Spatial-0812 --include_trace False --base_dir /scratch/gpfs/TSILVER/bb8404/molmoact/experiments/libero/rollouts

