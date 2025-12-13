#!/usr/bin/env python3
"""
Generate visual traces using Google Gemini 2.5 Pro.

This script uses Gemini 2.5 Pro to generate 5-point trajectory traces starting from
a specified pixel coordinate, then draws the trace on the image to create a perturbed version.

Usage:
    python coerce_gemini_visual_trace.py [--image PATH] [--pixel-coords X,Y] [--trials N] [--output PATH]

Environment Variables:
    GEMINI_PROJECT: Google Cloud project ID (default: "pokeagent-011")
    GEMINI_LOCATION: Google Cloud location (default: "us-central1")
    IMAGE_PATH: Path to input image (can be overridden by --image)
    PIXEL_COORDS: Pixel coordinates as "X,Y" (can be overridden by --pixel-coords)
"""

import argparse
import ast
import io
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError as e:
    print(f"Error: Missing required package. Install with: pip install google-genai")
    sys.exit(1)

# Add necessary paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "LIBERO"))
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MOLMOACT_DATA_DIR", str(PROJECT_ROOT / "data"))


def get_libero_task_language_instruction(
    task_suite_name: str = "libero_spatial", task_id: int = 0, task_order_index: int = 0
) -> str:
    """
    Get the language instruction for a LIBERO task by reading from BDDL files.

    Args:
        task_suite_name: Name of the task suite (e.g., "libero_spatial", "libero_object")
        task_id: Task ID (0-9)
        task_order_index: Task order index (default: 0, which uses [0,1,2,3,4,5,6,7,8,9])

    Returns:
        Language instruction string for the task
    """
    try:
        # Import task map - add the benchmark directory to path
        benchmark_dir = SCRIPT_DIR.parent / "LIBERO" / "libero" / "libero" / "benchmark"
        if str(benchmark_dir) not in sys.path:
            sys.path.insert(0, str(benchmark_dir))
        from libero_suite_task_map import libero_task_map

        # Task orders (from benchmark/__init__.py)
        task_orders = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [4, 6, 8, 7, 3, 1, 2, 0, 9, 5],
            [6, 3, 5, 0, 4, 2, 9, 1, 8, 7],
            [7, 4, 3, 0, 8, 1, 2, 5, 9, 6],
            [4, 5, 6, 3, 8, 0, 2, 7, 1, 9],
            [1, 2, 3, 0, 6, 9, 5, 7, 4, 8],
            [3, 7, 8, 1, 6, 2, 9, 4, 0, 5],
            [4, 2, 9, 7, 6, 8, 5, 1, 3, 0],
            [1, 8, 5, 4, 0, 9, 6, 7, 2, 3],
            [8, 3, 6, 4, 9, 5, 1, 2, 0, 7],
            [6, 9, 0, 5, 7, 1, 2, 8, 3, 4],
            [6, 8, 3, 1, 0, 2, 5, 9, 7, 4],
            [8, 0, 6, 9, 4, 1, 7, 3, 2, 5],
            [3, 8, 6, 4, 2, 5, 0, 7, 1, 9],
            [7, 1, 5, 6, 3, 2, 8, 9, 4, 0],
            [2, 0, 9, 5, 3, 6, 8, 7, 1, 4],
            [3, 5, 9, 6, 2, 4, 8, 7, 1, 0],
            [7, 6, 5, 9, 0, 3, 4, 2, 8, 1],
            [2, 5, 0, 9, 3, 1, 6, 4, 8, 7],
            [3, 5, 1, 2, 7, 8, 6, 0, 4, 9],
            [3, 4, 1, 9, 7, 6, 8, 2, 0, 5],
        ]

        # Get the actual task index from the task order
        if task_order_index >= len(task_orders):
            task_order_index = 0
        task_order = task_orders[task_order_index]
        if task_id >= len(task_order):
            raise ValueError(f"Task ID {task_id} is out of range for task order")
        actual_task_index = task_order[task_id]

        # Get the task name from the task map
        if task_suite_name not in libero_task_map:
            raise ValueError(f"Unknown task suite: {task_suite_name}")
        task_list = libero_task_map[task_suite_name]
        if actual_task_index >= len(task_list):
            raise ValueError(f"Task index {actual_task_index} is out of range for {task_suite_name}")
        task_name = task_list[actual_task_index]

        # Construct BDDL file path
        bddl_dir = Path(SCRIPT_DIR.parent / "LIBERO" / "libero" / "libero" / "bddl_files" / task_suite_name)
        bddl_file = bddl_dir / f"{task_name}.bddl"

        if not bddl_file.exists():
            raise FileNotFoundError(f"BDDL file not found: {bddl_file}")

        # Read and parse the BDDL file to extract language instruction
        with open(bddl_file, "r") as f:
            content = f.read()

        # Extract language instruction from (:language ...) line
        # Pattern matches (:language followed by text until closing parenthesis
        # Uses DOTALL to handle multi-line instructions
        language_match = re.search(r"\(:language\s+(.+?)\)", content, re.DOTALL)
        if language_match:
            language_instruction = language_match.group(1).strip()
            # Clean up any extra whitespace/newlines
            language_instruction = " ".join(language_instruction.split())
            return language_instruction
        else:
            raise ValueError(f"Could not find language instruction in BDDL file: {bddl_file}")

    except Exception as e:
        print(f"Warning: Could not load LIBERO task language instruction: {e}")
        import traceback

        traceback.print_exc()
        return ""


def get_trace_points_with_gemini(
    pixel_coords: Tuple[int, int],
    image: Image.Image,
    client: genai.Client,
    language_instruction: str,
    model_name: str = "gemini-2.5-pro",
) -> Optional[List[List[int]]]:
    """
    Use Gemini to generate 5 trace points starting from pixel coordinates.

    Args:
        pixel_coords: Starting pixel coordinates as (x, y)
        image: PIL Image object
        client: Initialized Gemini client
        language_instruction: LIBERO task language instruction
        model_name: Name of the Gemini model to use

    Returns:
        List of 5 [x, y] coordinates, or None if parsing fails
    """
    x, y = pixel_coords
    w, h = image.size

    # Convert full image to bytes (not cropped, so Gemini can see context)
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    image_bytes = img_bytes.getvalue()

    # Create image part
    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/png",
    )

    # Prompt for 5 trace points incorporating the language instruction
    prompt = (
        f"Task instruction: {language_instruction}\n\n"
        f"The image is {w}x{h} pixels. "
        f"Generate a trajectory path of exactly 5 points that represents a path starting from the coordinate of the end effector/robot hand in the image to the target object and then to the final goal, following the task instruction. "
        f"Return the coordinates strictly in this format: "
        "Trace: [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]]"
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, image_part],
        )
        response_text = response.text.strip()

        # Parse the trace coordinates from the response
        trace_match = re.search(r"Trace:\s*(\[\[.*?\]\])", response_text, re.DOTALL)
        if trace_match:
            try:
                trace_str = trace_match.group(1)
                trace = ast.literal_eval(trace_str)
                # Validate: should be a list of 5 points
                if isinstance(trace, list) and len(trace) == 5:
                    # Ensure all points are [x, y] pairs
                    if all(isinstance(p, list) and len(p) == 2 for p in trace):
                        return trace
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse trace: {e}")
                print(f"Response text: {response_text}")

        # If parsing failed, return None
        print(f"Warning: Failed to extract valid trace from response")
        print(f"Response text: {response_text}")
        return None

    except Exception as e:
        print(f"Error calling Gemini: {e}")
        raise


def draw_trace_on_image(
    image_path: str, trace: List[List[int]], output_path: str, color: Tuple[int, int, int] = (255, 0, 0)
) -> bool:
    """
    Draw a trace on the image using OpenCV.

    Args:
        image_path: Path to input image
        trace: List of [x, y] coordinates
        output_path: Path to save annotated image
        color: BGR color tuple for the trace (default: blue)

    Returns:
        True if successful, False otherwise
    """
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return False

    print(f"Drawing trace with {len(trace)} points: {trace}")

    line_width = 3
    point_radius = 4

    # Draw lines connecting consecutive points
    for i in range(len(trace) - 1):
        start_point = tuple(trace[i])
        end_point = tuple(trace[i + 1])
        cv2.line(img, start_point, end_point, color, line_width, cv2.LINE_AA)
        print(f"  Drawing line from {start_point} to {end_point}")

    # Draw circles at each point for visibility
    for i, point in enumerate(trace):
        x, y = tuple(point)
        cv2.circle(img, (x, y), point_radius, color, -1)  # -1 for filled circle
        print(f"  Drawing point {i+1} at ({x}, {y})")

    # Save the annotated image
    success = cv2.imwrite(output_path, img)
    if success:
        print(f"Annotated image saved to: {output_path}")
        # Verify file was actually created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"  File size: {file_size} bytes")
            return True
        else:
            print(f"  Error: File was not created at {output_path}")
            return False
    else:
        print(f"Error: cv2.imwrite failed to save image to {output_path}")
        return False


def parse_pixel_coords(coords_str: str) -> Tuple[int, int]:
    """
    Parse pixel coordinates from string.

    Args:
        coords_str: String in format "(X, Y)" or "X, Y" or "X,Y"

    Returns:
        Tuple of (x, y) coordinates

    Raises:
        ValueError: If coordinates cannot be parsed
    """
    coords_str = coords_str.strip().strip("()")
    parts = coords_str.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate format: {coords_str}. Expected 'X,Y' or '(X, Y)'")
    try:
        x, y = int(parts[0].strip()), int(parts[1].strip())
        return (x, y)
    except ValueError as e:
        raise ValueError(f"Invalid coordinate values: {coords_str}") from e


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate visual traces using Google Gemini 2.5 Pro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (default: from IMAGE_PATH env var or create dummy)",
    )
    parser.add_argument(
        "--pixel-coords",
        type=str,
        default=None,
        help='Pixel coordinates as "X,Y" (default: from PIXEL_COORDS env var or "(128, 128)")',
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials to run (default: 5, image saved after each trial)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for annotated image (default: <input>_with_trace.png)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Google Cloud project ID (default: from GEMINI_PROJECT env var or 'pokeagent-011')",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Google Cloud location (default: from GEMINI_LOCATION env var or 'us-central1')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
        help="Gemini model name (default: gemini-2.5-pro)",
    )
    parser.add_argument(
        "--task-suite",
        type=str,
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10"],
        help="LIBERO task suite name (default: libero_spatial)",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="LIBERO task ID (0-9, default: 0)",
    )
    parser.add_argument(
        "--language-instruction",
        type=str,
        default=None,
        help="Language instruction (default: load from LIBERO task suite and task ID)",
    )
    parser.add_argument(
        "--task-order-index",
        type=int,
        default=0,
        help="Task order index (default: 0, which uses [0,1,2,3,4,5,6,7,8,9])",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Generating 5-point trace using Gemini 2.5 Pro and drawing on image")
    print("=" * 80)

    # Initialize Vertex AI client
    print("\nInitializing Vertex AI client...")
    try:
        project = args.project or os.environ.get("GEMINI_PROJECT", "pokeagent-011")
        location = args.location or os.environ.get("GEMINI_LOCATION", "us-central1")
        client = genai.Client(vertexai=True, project=project, location=location)
        print(f"Gemini client initialized successfully! (project: {project}, location: {location})")
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Load or create image
    image_path = args.image or os.environ.get("IMAGE_PATH")
    print(f"DEBUG: args.image = {args.image}")
    print(f"DEBUG: os.environ.get('IMAGE_PATH') = {os.environ.get('IMAGE_PATH')}")
    print(f"DEBUG: Resolved image_path = {image_path}")
    
    if image_path and os.path.exists(image_path):
        print(f"\nLoading image from: {image_path}")
        image = Image.open(image_path).convert("RGB")
    else:
        print("\nCreating dummy test image (256x256 RGB)...")
        # Create a simple test image with some colored regions
        img_array = np.zeros((256, 256, 3), dtype=np.uint8)
        img_array[50:150, 50:150] = [255, 0, 0]  # Red square
        img_array[100:200, 100:200] = [0, 255, 0]  # Green square
        img_array[150:250, 150:250] = [0, 0, 255]  # Blue square
        image = Image.fromarray(img_array)
        # Save dummy image so we can draw on it later
        dummy_image_path = "/tmp/dummy_test_image.png"
        image.save(dummy_image_path)
        image_path = dummy_image_path
        print(f"Using dummy image. Set --image or IMAGE_PATH env var to use a real image.")
        print(f"Dummy image saved to: {dummy_image_path}")

    print(f"Image size: {image.size}")
    print(f"Image dimensions: width={image.size[0]}, height={image.size[1]}")

    # Parse pixel coordinates
    pixel_coords_str = args.pixel_coords or os.environ.get("PIXEL_COORDS", "(128, 128)")
    try:
        pixel_coords = parse_pixel_coords(pixel_coords_str)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(f"\nTesting with pixel coordinates: {pixel_coords}")
    if image_path:
        print(f"Pixel value at {pixel_coords}: {image.getpixel(pixel_coords)}")

    # Get language instruction
    if args.language_instruction:
        language_instruction = args.language_instruction
        print(f"\nUsing provided language instruction: {language_instruction}")
    else:
        print(
            f"\nLoading language instruction from LIBERO {args.task_suite} task {args.task_id} (order index {args.task_order_index})..."
        )
        language_instruction = get_libero_task_language_instruction(
            args.task_suite, args.task_id, args.task_order_index
        )
        if language_instruction:
            print(f"Language instruction: {language_instruction}")
        else:
            print("Warning: Could not load language instruction, proceeding without it")
            language_instruction = ""

    # Run trials
    num_trials = args.trials
    successes = 0
    failures = 0
    traces = []  # Store successful traces

    print("\n" + "=" * 80)
    print(f"Running {num_trials} trials...")
    print("=" * 80)

    # Determine output path once
    if args.output:
        output_path = args.output
    else:
        base_name = Path(image_path).stem if image_path else "output"
        output_dir = Path(image_path).parent if image_path else Path(".")
        output_path = str(output_dir / f"{base_name}_with_trace.png")
    
    # Create output directory if it doesn't exist
    output_path_obj = Path(output_path)
    output_dir = output_path_obj.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput will be saved to: {output_path}", flush=True)
    
    for trial in range(1, num_trials + 1):
        print(f"\nTrial {trial}/{num_trials}...", flush=True)
        try:
            trace = get_trace_points_with_gemini(pixel_coords, image, client, language_instruction, args.model)
            if trace and isinstance(trace, list) and len(trace) == 5:
                successes += 1
                traces.append(trace)
                print(f"  ✓ Success: Got trace with {len(trace)} points", flush=True)
                print(f"     Trace: {trace}", flush=True)
                
                # Save image immediately after each successful trace
                if image_path and os.path.exists(image_path):
                    print(f"  → Saving image with trace #{successes}...", flush=True)
                    try:
                        if draw_trace_on_image(image_path, trace, output_path):
                            file_size = os.path.getsize(output_path)
                            print(f"  ✓ Image saved successfully ({file_size} bytes)", flush=True)
                        else:
                            print(f"  ✗ Failed to save image", flush=True)
                    except Exception as e:
                        print(f"  ✗ Error saving image: {e}", flush=True)
            else:
                failures += 1
                print(f"  ✗ Failed: Invalid or empty trace", flush=True)
        except Exception as e:
            failures += 1
            print(f"  ✗ Failed: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # Report results
    print("\n" + "=" * 80)
    print("TRIAL RESULTS")
    print("=" * 80)
    print(f"Total trials: {num_trials}")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")
    print(f"Success rate: {successes / num_trials * 100:.1f}%")
    print("=" * 80)

    if traces:
        print(f"\nAll {len(traces)} successful traces:")
        for i, trace in enumerate(traces, 1):
            print(f"  {i}. {trace}")
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"\n✓ Final image saved at: {output_path} (size: {file_size} bytes)")
        else:
            print(f"\n✗ Warning: Final image not found at {output_path}")
    else:
        print("\n✗ No successful traces generated")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

