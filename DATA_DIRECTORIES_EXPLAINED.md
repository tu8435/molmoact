# Data Directories Explained

## Why Do We Need These Directories?

When working with MolmoAct, you'll be downloading large datasets and pre-trained models. These directories keep everything organized and prevent you from having to re-download gigabytes of data every time you run experiments.

## Directory Structure

```
molmoact/
├── data/                           # Main data directory (MOLMOACT_DATA_DIR)
│   ├── huggingface/                # HuggingFace cache (HF_HOME)
│   │   ├── datasets/               # Downloaded datasets from HuggingFace
│   │   └── models/                 # Downloaded model checkpoints
│   ├── robot_datasets/             # Robot demonstration datasets
│   └── checkpoints/                # Model checkpoints during training
```

## Environment Variables

### `MOLMOACT_DATA_DIR`
- **Purpose**: Root directory for MolmoAct-specific data
- **Contains**: Robot datasets, auxiliary assets, dataset statistics
- **Size**: Can grow to several GB depending on which datasets you download
- **Path**: `/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data`

### `HF_HOME`
- **Purpose**: HuggingFace cache directory for models and datasets
- **Contains**: Pre-trained models, tokenizers, datasets from HuggingFace Hub
- **Size**: Can be 10+ GB for large models
- **Path**: `/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data/huggingface`
- **Why separate**: HuggingFace has its own caching mechanism that's standardized across projects

## What Gets Stored Here?

### Robot Datasets
According to the README, MolmoAct uses several datasets:
- **MolmoAct Dataset** (in-house collected data)
- **OXE subset** (for pre-training)
- **LIBERO** (for post-training)
- **RT-1, BC-Z, BridgeData V2** (various robot datasets)

### Model Checkpoints
- Pre-trained MolmoAct models (7B parameters → ~14GB per model)
- Fine-tuned checkpoints during your training
- VQVAE models for depth estimation

### Auxiliary Files
- Depth-Anything-V2 checkpoints (depth_anything_v2_vitb.pth)
- Dataset statistics (dataset_statistics.json)
- Normalization parameters

## Why Are They in .gitignore?

These directories are excluded from git because:
1. **Size**: Model files and datasets are too large for git (many GB)
2. **Reproducibility**: Others should download fresh copies from official sources
3. **Storage**: Avoids bloating your git repository
4. **Privacy**: Some datasets may have usage restrictions

## Managing Disk Space

If you need to free up space:

```bash
# Remove HuggingFace cache (will be re-downloaded when needed)
rm -rf /scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data/huggingface

# Remove specific datasets
rm -rf /scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data/robot_datasets/<dataset_name>

# Clean old checkpoints
rm -rf /scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/checkpoints/<old_experiment>
```

## Best Practices

1. **Set environment variables in ~/.bashrc** so they persist across sessions
2. **Monitor disk usage** with `du -sh data/` periodically
3. **Only download datasets you need** for your specific experiments
4. **Back up important checkpoints** before cleaning up old ones
5. **Use `HF_DATASETS_OFFLINE=1`** during training to avoid rate limiting (as mentioned in README)



