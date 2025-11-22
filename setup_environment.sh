#!/bin/bash

# MolmoAct Environment Setup Script
# This script sets up the conda environment for MolmoAct experiments

set -e  # Exit on error

echo "========================================="
echo "MolmoAct Environment Setup"
echo "========================================="

# Initialize conda in this shell
source ~/. bashrc

# Create conda environment with Python 3.11
echo ""
echo "Creating conda environment 'molmoact' with Python 3.11..."
conda create -n molmoact python=3.11 -y

# Activate the environment
echo ""
echo "Activating molmoact environment..."
conda activate molmoact

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA 12.1 support..."
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Navigate to molmoact directory
cd /scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact

# Install MolmoAct package
echo ""
echo "Installing MolmoAct package..."
pip install -e .[all]

# Set up environment variables
echo ""
echo "Setting up environment variables..."
export MOLMOACT_DATA_DIR=/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data
export HF_HOME=/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data/huggingface

# Create data directories
mkdir -p $MOLMOACT_DATA_DIR
mkdir -p $HF_HOME

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate molmoact"
echo ""
echo "Environment variables set for this session:"
echo "  MOLMOACT_DATA_DIR=$MOLMOACT_DATA_DIR"
echo "  HF_HOME=$HF_HOME"
echo ""
echo "To make these permanent, add them to your ~/.bashrc:"
echo "  export MOLMOACT_DATA_DIR=/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data"
echo "  export HF_HOME=/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data/huggingface"
echo ""

