#!/bin/bash
# Shell script wrapper for draw_trace_manual.py
# Manually draws a trace on an image with specified coordinates

set -e

# Change to script directory
SCRIPT_DIR="/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/experiments/libero"
cd "$SCRIPT_DIR"

# Set environment variables
export MOLMOACT_DATA_DIR="/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data"

# Optional: Set default image path and trace coordinates
export IMAGE_PATH="${IMAGE_PATH:-/scratch/gpfs/TSILVER/el5267/molmoact/output_frame.png}"
export TRACE_COORDS="${TRACE_COORDS:-[[128,40],[165,110],[205,165],[150,160],[60,150]]}"

# Run the script
echo "Drawing trace on image..."
python draw_trace_manual.py "$@"
