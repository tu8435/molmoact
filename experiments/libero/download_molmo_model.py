#!/usr/bin/env python3
"""
Download and cache the Molmo-7B-D-0924 model from HuggingFace.

This script downloads the model and processor to a local cache directory for offline use.

Usage:
    python download_molmo_model.py [--cache-dir PATH] [--model-id MODEL_ID]

Environment Variables:
    HF_HOME: HuggingFace home directory (default: <project_root>/data/huggingface)
    HF_HUB_CACHE: HuggingFace hub cache directory
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from transformers import AutoModelForCausalLM, AutoProcessor
except ImportError as e:
    print(f"Error: Missing required package. Install with: pip install transformers")
    sys.exit(1)

# Set default paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "huggingface"
DEFAULT_MODEL_ID = "allenai/Molmo-7B-D-0924"


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and cache Molmo model from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Model ID to download (default: {DEFAULT_MODEL_ID})",
    )

    args = parser.parse_args()

    # Determine cache directory
    cache_dir = args.cache_dir or os.environ.get("HF_HOME", str(DEFAULT_CACHE_DIR))
    cache_dir = Path(cache_dir)
    hub_cache = cache_dir / "hub"

    # Set environment variables
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))

    model_id = args.model_id

    print("=" * 80)
    print(f"Downloading and caching {model_id}")
    print("=" * 80)
    print(f"Cache directory: {cache_dir}")
    print(f"Hub cache: {hub_cache}")
    print()

    # Download processor
    print("Downloading processor...")
    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            cache_dir=str(hub_cache),
        )
        print("Processor downloaded successfully!")
    except Exception as e:
        print(f"Error downloading processor: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print()

    # Download model
    print("Downloading model (this may take a while, ~14GB)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=str(hub_cache),
        )
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print()
    print("=" * 80)
    print("Download complete!")
    print("=" * 80)
    print(f"Model cached at: {hub_cache}/models--{model_id.replace('/', '--')}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
