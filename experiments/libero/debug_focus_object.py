#!/usr/bin/env python3
"""
Lite debugging script for get_focus_object_from_pixel_coords function.
"""

import sys
import os
import torch
from PIL import Image
import numpy as np
from collections import Counter

# Add necessary paths
sys.path.insert(0, "/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/experiments/LIBERO")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
os.environ["MOLMOACT_DATA_DIR"] = "/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data"

from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM

# Import the function from the original file
sys.path.insert(0, os.path.dirname(__file__))
from run_libero_pixel import get_focus_object_from_pixel_coords


def main():
    print("=" * 80)
    print("Debugging get_focus_object_from_pixel_coords")
    print("=" * 80)
    
    # Set up environment variables (similar to main script)
    # Enable offline mode by default for GPU nodes without internet
    use_offline = (os.environ.get("HF_HUB_OFFLINE") == "1" or 
                   os.environ.get("TRANSFORMERS_OFFLINE") == "1" or
                   os.environ.get("FORCE_OFFLINE", "0") == "1")
    
    if use_offline:
        print("OFFLINE MODE ENABLED - Using cached models only")
    
    # Default cached model path - use Molmo-7B-D-0924 (base multimodal model, not fine-tuned for actuation)
    default_cache_path = "/scratch/gpfs/TSILVER/tu8435/ECE531_final_project/molmoact/data/huggingface/hub/models--allenai--Molmo-7B-D-0924"
    
    # Model checkpoint - use cached model by default
    checkpoint = os.environ.get("DEBUG_CHECKPOINT", None)
    
    # If no checkpoint specified, try to find the cached model
    if checkpoint is None:
        # Check if the default cache path exists
        if os.path.exists(default_cache_path):
            snapshots_dir = os.path.join(default_cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshots:
                    snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                    if os.path.exists(os.path.join(snapshot_path, "config.json")):
                        checkpoint = snapshot_path
                        print(f"Found cached model at: {checkpoint}")
                    else:
                        print(f"Warning: No config.json found in {snapshot_path}")
                else:
                    print(f"Warning: No snapshots found in {snapshots_dir}")
            else:
                print(f"Warning: No snapshots directory found in {default_cache_path}")
        else:
            print(f"ERROR: Default cache path not found: {default_cache_path}")
            if use_offline:
                print("ERROR: Cannot proceed in offline mode without cached model!")
                return 1
            # Fallback to model name (will fail in offline mode)
            checkpoint = "allenai/Molmo-7B-D-0924"
            print(f"Falling back to model name: {checkpoint}")
    else:
        # If checkpoint is provided, check if it's a path to a cached model
        if os.path.exists(checkpoint):
            # It's already a path, use it directly
            print(f"Using provided checkpoint path: {checkpoint}")
        else:
            # It's a model name, try to find it in cache
            hub_cache = os.environ.get("HF_HUB_CACHE", os.path.join(os.environ.get("HF_HOME", ""), "hub"))
            model_name_safe = checkpoint.replace("/", "--")
            model_cache_path = os.path.join(hub_cache, f"models--{model_name_safe}")
            
            if os.path.exists(model_cache_path):
                snapshots_dir = os.path.join(model_cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshots:
                        snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                        if os.path.exists(os.path.join(snapshot_path, "config.json")):
                            checkpoint = snapshot_path
                            print(f"Found cached model at: {checkpoint}")
    
    # Final check: if offline mode and checkpoint is not a valid path, error out
    if use_offline and not os.path.exists(checkpoint):
        print(f"ERROR: Offline mode enabled but checkpoint path does not exist: {checkpoint}")
        print("Please ensure the cached model is available at the expected location.")
        return 1
    
    print(f"Using checkpoint: {checkpoint}")
    
    # Set hub_cache for offline mode
    hub_cache = os.environ.get("HF_HUB_CACHE", os.path.join(os.environ.get("HF_HOME", ""), "hub"))
    
    # Load processor
    print("\nLoading processor...")
    processor_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
    }
    if use_offline:
        processor_kwargs["local_files_only"] = True
        if hub_cache:
            processor_kwargs["cache_dir"] = hub_cache
    
    processor = AutoProcessor.from_pretrained(checkpoint, **processor_kwargs)
    print("Processor loaded successfully!")
    
    # Load model - try Molmo first (AutoModelForCausalLM), fallback to MolmoAct (AutoModelForImageTextToText)
    print("\nLoading model...")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
        "device_map": "auto",
    }
    if use_offline:
        model_kwargs["local_files_only"] = True
        if hub_cache:
            model_kwargs["cache_dir"] = hub_cache
    
    # Try to load as Molmo (AutoModelForCausalLM) first
    try:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **model_kwargs)
        print("Model loaded as Molmo (AutoModelForCausalLM)")
    except Exception as e:
        print(f"Failed to load as Molmo, trying MolmoAct: {e}")
        # Fallback to MolmoAct
        model = AutoModelForImageTextToText.from_pretrained(checkpoint, **model_kwargs)
        model._qwen_tokenizer = processor.tokenizer
        print("Model loaded as MolmoAct (AutoModelForImageTextToText)")
    
    print("Model loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Create a test image (or load from file if provided)
    test_image_path = os.environ.get("DEBUG_IMAGE_PATH", None)
    if test_image_path and os.path.exists(test_image_path):
        print(f"\nLoading test image from: {test_image_path}")
        image = Image.open(test_image_path).convert("RGB")
    else:
        print("\nCreating dummy test image (256x256 RGB)...")
        # Create a simple test image with some colored regions
        img_array = np.zeros((256, 256, 3), dtype=np.uint8)
        # Add some colored rectangles to make it more interesting
        img_array[50:150, 50:150] = [255, 0, 0]  # Red square
        img_array[100:200, 100:200] = [0, 255, 0]  # Green square
        img_array[150:250, 150:250] = [0, 0, 255]  # Blue square
        image = Image.fromarray(img_array)
        print("Using dummy image. Set DEBUG_IMAGE_PATH env var to use a real image.")
    
    print(f"Image size: {image.size}")
    print(f"Image dimensions: width={image.size[0]}, height={image.size[1]}")
    
    # Test pixel coordinates
    pixel_coords_str = os.environ.get("DEBUG_PIXEL_COORDS", "(128, 128)")
    # Parse pixel coords - handle both "(X, Y)" and "X, Y" formats
    pixel_coords_str = pixel_coords_str.strip("()")
    x, y = map(int, pixel_coords_str.split(","))
    pixel_coords = (x, y)
    
    print(f"\nTesting with pixel coordinates: {pixel_coords}")
    print(f"Pixel value at {pixel_coords}: {image.getpixel(pixel_coords)}")
    
    # Run 20 trials
    num_trials = 20
    successes = 0
    failures = 0
    results = []
    
    print("\n" + "=" * 80)
    print(f"Running {num_trials} trials...")
    print("=" * 80)
    
    for trial in range(1, num_trials + 1):
        print(f"\nTrial {trial}/{num_trials}...")
        try:
            result = get_focus_object_from_pixel_coords(pixel_coords, image, model, processor)
            if result and result.strip():  # Success if we get a non-empty result
                successes += 1
                results.append(result.strip())
                print(f"  ✓ Success: {result.strip()}")
            else:
                failures += 1
                print(f"  ✗ Failed: Empty result")
        except Exception as e:
            failures += 1
            print(f"  ✗ Failed: {e}")
    
    # Report results
    print("\n" + "=" * 80)
    print("TRIAL RESULTS")
    print("=" * 80)
    print(f"Total trials: {num_trials}")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")
    print(f"Success rate: {successes / num_trials * 100:.1f}%")
    print("=" * 80)
    
    if results:
        print("\nAll successful results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result}")
        
        # Count unique results
        result_counts = Counter(results)
        print("\nResult frequency:")
        for result, count in result_counts.most_common():
            print(f"  '{result}': {count} times ({count/len(results)*100:.1f}%)")
    
    print("=" * 80)
    
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    exit(main())

