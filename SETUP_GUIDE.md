# MolmoAct Environment Setup Guide

## Quick Setup (Recommended)

Run the automated setup script:

```bash
cd /scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact
bash setup_environment.sh
```

This script will:
1. Create a conda environment named `molmoact` with Python 3.11
2. Install PyTorch with CUDA 12.1 support
3. Install all MolmoAct dependencies
4. Set up data directories

## Manual Setup

If you prefer to set up manually:

### 1. Create Conda Environment

```bash
conda create -n molmoact python=3.11 -y
conda activate molmoact
```

### 2. Install PyTorch with CUDA Support

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 3. Install MolmoAct

```bash
cd /scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact
pip install -e .[all]
```

### 4. Set Up Environment Variables and Data Directories

These directories store downloaded datasets and model caches so you don't have to re-download them every time.

Add these to your `~/.bashrc`:

```bash
export MOLMOACT_DATA_DIR=/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data
export HF_HOME=/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data/huggingface
```

Then create the directories:

```bash
mkdir -p /scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data
mkdir -p /scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data/huggingface
```

**What these directories are for:**
- `MOLMOACT_DATA_DIR`: Stores robot datasets and extra assets downloaded by MolmoAct
- `HF_HOME`: HuggingFace cache for downloaded models and datasets (can be several GB)

**Note:** These directories are already in `.gitignore` so large datasets won't be committed to git.

## Using the Environment

### Activate the environment:

```bash
conda activate molmoact
```

### Verify installation:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Important Locations

- **Miniconda installation**: `/scratch/gpfs/TSILVER/tu8435/miniconda3`
- **MolmoAct project**: `/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact`
- **Data directory**: `/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data`
- **HuggingFace cache**: `/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data/huggingface`

## Next Steps

After setup, you can:

1. **Download datasets** (see README.md section on downloading robot datasets)
2. **Run training** (see README.md section 4 on Training)
3. **Run evaluation** (see README.md section 5 on Evaluation)

## Troubleshooting

### If conda command is not found:
```bash
source ~/.bashrc
```

### If you need to reinstall the environment:
```bash
conda deactivate
conda env remove -n molmoact
# Then run the setup script again
```

### Check your GPU:
```bash
nvidia-smi
```

