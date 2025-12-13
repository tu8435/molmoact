#!/usr/bin/env python3
"""
Script to use Google Gemini 3 Pro via Vertex AI to generate a trajectory trace.
Asks Gemini to draw a blue line from the end effector/robot hand to the cream cheese to the basket.
"""

import os
import sys
import re
import ast
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image, ImageDraw
import base64
import io
import numpy as np

def main():
    # Initialize Vertex AI client
    print("Initializing Vertex AI client...")
    client = genai.Client(
        vertexai=True,
        project="pokeagent-011",
        location="us-central1",
    )
    
    # Image path
    image_path = "/scratch/gpfs/TSILVER/el5267/molmoact/output_frame.png"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return 1
    
    print(f"Loading image from: {image_path}")
    
    # Load and prepare the image
    # For local files, we need to convert to bytes or upload to GCS
    # Using base64 encoding for local file
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # Create image part from bytes
    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/png",
    )
    
    # Prompt asking Gemini to generate the trace
    prompt = (
        "Our image is 256x256 pixels. Draw a line that gives the correct trajectory that the effector/robot hand should take "
        "to the cream cheese box, then to the basket."
        "Return a set of five coordinates that form this trajectory path and a few word description of what you are looking for, strictly following this format. [x1, y1] is the coordinate of the robots hand, [x3,y3] is the coordinate of the cream cheese box, [x5,y5] is the coordinate of the basket: "
        "Trace: [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]], "
        "Description: [description]"
    )
    
    print("\nSending request to Gemini 2.5 Pro...")
    print(f"Prompt: {prompt}")
    
    # Generate content using Gemini 2.5 Pro
    model = "gemini-2.5-pro"
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                prompt,
                image_part,
            ],
        )
        
        print("\n" + "=" * 80)
        print("GEMINI RESPONSE")
        print("=" * 80)
        print(response.text)
        print("=" * 80)
        
        response_text = response.text
        
        # Save the response
        output_file = "gemini_trace_response.txt"
        with open(output_file, "w") as f:
            f.write(response_text)
        print(f"\nResponse saved to: {output_file}")
        
        # Parse the trace coordinates from the response
        # Expected format: "Trace: [[130, 116], [178, 135], ...], Description: [...]"
        trace = None
        description = None
        
        # Try to extract trace using regex
        trace_match = re.search(r'Trace:\s*(\[\[.*?\]\])', response_text, re.DOTALL)
        if trace_match:
            try:
                trace_str = trace_match.group(1)
                trace = ast.literal_eval(trace_str)
                print(f"\nExtracted trace: {trace}")
            except Exception as e:
                print(f"Warning: Could not parse trace: {e}")
        
        # Try to extract description
        desc_match = re.search(r'Description:\s*(\[.*?\])', response_text)
        if desc_match:
            try:
                desc_str = desc_match.group(1)
                description = ast.literal_eval(desc_str)
                print(f"Extracted description: {description}")
            except Exception as e:
                print(f"Warning: Could not parse description: {e}")
        
        # If trace was found, draw it on the image
        if trace and len(trace) > 0:
            print(f"\nDrawing blue line with {len(trace)} points...")
            
            # Load the original image
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            
            # Draw blue line connecting the points
            blue_color = (0, 0, 255)  # RGB for blue
            line_width = 3
            
            # Convert trace to list of tuples if needed
            if isinstance(trace[0], list):
                points = [tuple(point) for point in trace]
            else:
                points = trace
            
            # Draw lines connecting consecutive points
            for i in range(len(points) - 1):
                start_point = tuple(points[i])
                end_point = tuple(points[i + 1])
                draw.line([start_point, end_point], fill=blue_color, width=line_width)
            
            # Also draw circles at each point for visibility
            point_radius = 4
            for point in points:
                x, y = tuple(point)
                draw.ellipse(
                    [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                    fill=blue_color,
                    outline=blue_color
                )
            
            # Save the annotated image
            output_image_path = "output_frame_with_trace.png"
            img.save(output_image_path)
            print(f"Annotated image saved to: {output_image_path}")
        else:
            print("\nWarning: No trace coordinates found in response. Cannot draw line.")
        
        return 0
        
    except Exception as e:
        print(f"\nError calling Gemini API: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

