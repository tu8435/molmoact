#!/bin/bash

#SBATCH --job-name=libero_eval       # create a short name for your job
#SBATCH --nodes=1                    # node count
#SBATCH --ntasks=1                   # total number of tasks across all nodes
#SBATCH --cpus-per-task=8            # cpu-cores per task (increased for data loading)
#SBATCH --gres=gpu:1                 # request 1 GPU
#SBATCH --mem=64G                    # total memory per node (7B model needs significant memory)
#SBATCH --time=06:00:00              # total run time limit (HH:MM:SS) - 6 hours should be enough
#SBATCH --mail-type=begin,end,fail   # receive email notifications
#SBATCH --mail-user=tu8435@princeton.edu  # replace with your email
#SBATCH --output=nov_%j.out        # output file name (%j is the job ID)
#SBATCH --error=nov_%j.err         # error file name (%j is the job ID)

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

# Set MUJOCO_EGL_DEVICE_ID to 0 to avoid MIG UUID parsing issues in robosuite
# This is checked BEFORE CUDA_VISIBLE_DEVICES in robosuite's EGL context code
# Setting this to 0 ensures robosuite uses device 0 instead of trying to parse CUDA_VISIBLE_DEVICES
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

# Run the evaluation script
# Output will be redirected by the script itself to out.txt if running interactively,
# or captured by SLURM's --output and --error flags
python tersoo_run_libero_eval.py --task spatial --task_id 1 --checkpoint allenai/MolmoAct-7B-D-LIBERO-Spatial-0812

