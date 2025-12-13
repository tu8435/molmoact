#!/usr/bin/env python3
"""
Manually draw a trace on an image with specified coordinates.

This script draws a blue line trajectory on an image using OpenCV.

Usage:
    python draw_trace_manual.py [--image PATH] [--trace COORDS] [--output PATH]

Environment Variables:
    IMAGE_PATH: Path to input image (can be overridden by --image)
    TRACE_COORDS: Trace coordinates as JSON list (can be overridden by --trace)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2

# Add paths similar to run_libero_pixel.py
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "LIBERO"))
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MOLMOACT_DATA_DIR", str(PROJECT_ROOT / "data"))


def draw_trace_on_image(
    image_path: str,
    trace: List[List[int]],
    output_path: str,
    color: Tuple[int, int, int] = (255, 0, 0),
    line_width: int = 3,
    point_radius: int = 4,
) -> bool:
    """
    Draw a trace on the image using OpenCV.

    Args:
        image_path: Path to input image
        trace: List of [x, y] coordinates
        output_path: Path to save annotated image
        color: BGR color tuple for the trace (default: blue)
        line_width: Width of the line
        point_radius: Radius of points

    Returns:
        True if successful, False otherwise
    """
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return False

    print(f"Image size: {img.shape[1]}x{img.shape[0]} (width x height)")
    print(f"Drawing blue line with {len(trace)} points: {trace}")

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
    cv2.imwrite(output_path, img)
    print(f"\nAnnotated image saved to: {output_path}")
    return True


def parse_trace(trace_str: str) -> List[List[int]]:
    """
    Parse trace coordinates from string.

    Args:
        trace_str: JSON string or Python list string

    Returns:
        List of [x, y] coordinates

    Raises:
        ValueError: If trace cannot be parsed
    """
    try:
        # Try JSON first
        trace = json.loads(trace_str)
    except json.JSONDecodeError:
        try:
            # Try Python literal eval
            import ast

            trace = ast.literal_eval(trace_str)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid trace format: {trace_str}") from e

    if not isinstance(trace, list):
        raise ValueError(f"Trace must be a list, got {type(trace)}")

    if not all(isinstance(p, list) and len(p) == 2 for p in trace):
        raise ValueError(f"Trace must be a list of [x, y] pairs")

    return trace


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Manually draw a trace on an image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (default: from IMAGE_PATH env var)",
    )
    parser.add_argument(
        "--trace",
        type=str,
        default=None,
        help='Trace coordinates as JSON list, e.g., "[[128,40],[165,110],...]" (default: from TRACE_COORDS env var)',
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for annotated image (default: <input>_with_trace_manual.png)",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="255,0,0",
        help="BGR color as R,G,B (default: 255,0,0 for blue)",
    )

    args = parser.parse_args()

    # Get image path
    image_path = args.image or os.environ.get("IMAGE_PATH")
    if not image_path:
        print("Error: Image path not provided. Use --image or set IMAGE_PATH env var.")
        return 1

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return 1

    print(f"Loading image from: {image_path}")

    # Get trace coordinates
    trace_str = args.trace or os.environ.get("TRACE_COORDS")
    if not trace_str:
        # Default trace
        trace = [[128, 40], [165, 110], [205, 165], [150, 160], [60, 150]]
        print("Using default trace (set --trace or TRACE_COORDS env var to override)")
    else:
        try:
            trace = parse_trace(trace_str)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    # Parse color
    try:
        color_parts = [int(x.strip()) for x in args.color.split(",")]
        if len(color_parts) != 3:
            raise ValueError("Color must have 3 components")
        color = tuple(color_parts)
    except (ValueError, AttributeError) as e:
        print(f"Error: Invalid color format '{args.color}'. Expected R,G,B")
        return 1

    # Generate output path
    if args.output:
        output_path = args.output
    else:
        base_name = Path(image_path).stem
        output_dir = Path(image_path).parent
        output_path = str(output_dir / f"{base_name}_with_trace_manual.png")

    # Draw trace
    if draw_trace_on_image(image_path, trace, output_path, color=color):
        print(f"\n✓ Successfully created annotated image: {output_path}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
