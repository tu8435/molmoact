#!/bin/bash
# Shell script wrapper for coerce_gemini_visual_trace.py
# Generates visual traces using Google Gemini 2.5 Pro

set -e

# Change to script directory
SCRIPT_DIR="/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/experiments/libero"
cd "$SCRIPT_DIR"

# Set environment variables
export MOLMOACT_DATA_DIR="/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data"
export GEMINI_PROJECT="${GEMINI_PROJECT:{insert project name here}}"
export GEMINI_LOCATION="${GEMINI_LOCATION:-us-central1}"

# Optional: Set default image path and pixel coordinates
export IMAGE_PATH="${IMAGE_PATH:-/scratch/gpfs/TSILVER/el5267/molmoact/output_frame.png}"
export PIXEL_COORDS="${PIXEL_COORDS:-159,137}"

# Run the script (default is 1 trial for fast execution, use --trials N for more)
echo "Running coerce_gemini_visual_trace.py..."
echo "Use --trials N to run multiple trials (default is 1)"
python coerce_gemini_visual_trace.py --task-suite libero_object --task-id 0 --output /scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/experiments/libero/image_perturbations/image_perturbation_trace.png "$@"

