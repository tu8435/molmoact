#!/usr/bin/env python3
"""
Script to download and cache the Molmo-7B-D-0924 model.
"""

import os
from transformers import AutoModelForCausalLM, AutoProcessor

# Set cache directory
cache_dir = "/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data/huggingface"
os.environ["HF_HOME"] = cache_dir
os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_dir, "hub")

model_id = "allenai/Molmo-7B-D-0924"

print("=" * 80)
print(f"Downloading and caching {model_id}")
print("=" * 80)
print(f"Cache directory: {cache_dir}")
print()

# Download processor
print("Downloading processor...")
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype='auto',
    cache_dir=os.path.join(cache_dir, "hub"),
)

print("Processor downloaded successfully!")
print()

# Download model
print("Downloading model (this may take a while, ~14GB)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto',
    cache_dir=os.path.join(cache_dir, "hub"),
)

print("Model downloaded successfully!")
print()
print("=" * 80)
print("Download complete!")
print("=" * 80)
print(f"Model cached at: {cache_dir}/hub/models--allenai--Molmo-7B-D-0924")
print()

