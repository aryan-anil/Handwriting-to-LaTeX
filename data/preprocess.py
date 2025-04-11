import os
import json
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
import random

def parse_inkml(file_path):
    """
    Parse InkML file to extract strokes and the normalized label.

    Args:
        file_path (str): Path to the InkML file.

    Returns:
        strokes (list): List of strokes, where each stroke is a list of (x, y) tuples.
        normalized_label (str): The normalized LaTeX label.
        sample_id (str): The unique sample identifier.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    namespace = "{http://www.w3.org/2003/InkML}"

    # Extract the normalized label
    normalized_label = None
    for annotation in root.findall(f"{namespace}annotation"):
        if annotation.attrib.get("type") == "normalizedLabel":
            normalized_label = annotation.text
        elif annotation.attrib.get("type") == "label":
            normalized_label = annotation.text

    # Extract the sample ID
    sample_id = None
    for annotation in root.findall(f"{namespace}annotation"):
        if annotation.attrib.get("type") == "sampleId":
            sample_id = annotation.text

    # Extract the strokes
    strokes = []
    for trace in root.findall(f"{namespace}trace"):
        points = trace.text.strip().split(',')
        stroke = [tuple(map(float, point.split()))[:2] for point in points]
        strokes.append(stroke)

    return strokes, normalized_label, sample_id

def render_strokes_flexible(strokes, sample_id, width, base_padding=500):
    """
    Render strokes into a smooth grayscale image using PIL with random padding on random sides.

    Args:
        strokes (list): List of strokes, where each stroke is a list of (x, y) tuples.
        sample_id (str): The sample identifier.
        width (int): Line width for rendering.
        base_padding (int): Minimum padding around the content in pixels.

    Returns:
        tuple: (filename, image array)
    """
    if not strokes:
        return f"{sample_id}_{width}.png", np.array(Image.new('L', (100, 100), color=255))

    # Find the bounds of the content
    all_points = np.array([point for stroke in strokes for point in stroke])
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    
    # Calculate dimensions with base padding
    content_width = max_vals[0] - min_vals[0]
    content_height = max_vals[1] - min_vals[1]
    
    # Scale factor to make the strokes a reasonable size
    scale_factor = min(1000 / content_width, 1000 / content_height) if content_width > 0 and content_height > 0 else 1
    
    # Generate random additional padding for each side (0 to 100 pixels)
    padding_left = base_padding + random.randint(0, 100) if random.random() > 0.5 else base_padding
    padding_right = base_padding + random.randint(0, 100) if random.random() > 0.5 else base_padding
    padding_top = base_padding + random.randint(0, 100) if random.random() > 0.5 else base_padding
    padding_bottom = base_padding + random.randint(0, 100) if random.random() > 0.5 else base_padding
    
    # Calculate final image dimensions with random padding
    image_width = int(content_width * scale_factor) + padding_left + padding_right
    image_height = int(content_height * scale_factor) + padding_top + padding_bottom
    
    # Create a white canvas with PIL (using higher resolution for better anti-aliasing)
    aa_scale = 2  # Anti-aliasing scale factor
    large_width = image_width * aa_scale
    large_height = image_height * aa_scale
    image = Image.new('L', (large_width, large_height), color=255)
    draw = ImageDraw.Draw(image)

    # Calculate transformation to fit strokes in the image with padding
    scale_x = (large_width - (padding_left + padding_right) * aa_scale) / content_width
    scale_y = (large_height - (padding_top + padding_bottom) * aa_scale) / content_height
    
    # Render each stroke
    for stroke in strokes:
        # Convert stroke points to canvas coordinates with random padding offsets
        normalized_stroke = [
            (
                int((x - min_vals[0]) * scale_x + padding_left * aa_scale),
                int((y - min_vals[1]) * scale_y + padding_top * aa_scale)
            ) 
            for x, y in stroke
        ]
        
        # Draw smooth lines between consecutive points
        if len(normalized_stroke) > 1:
            draw.line(normalized_stroke, fill=0, width=width * aa_scale, joint="curve")
            
            # Draw smaller dots at points for smoother connections
            for point in normalized_stroke:
                dot_size = width * aa_scale * 0.2
                draw.ellipse([
                    point[0] - dot_size, 
                    point[1] - dot_size, 
                    point[0] + dot_size, 
                    point[1] + dot_size
                ], fill=0)

    # Resize back to original size with anti-aliasing
    image = image.resize((image_width, image_height), Image.Resampling.LANCZOS)
    return f"{sample_id}_{width}.png", np.array(image)

def process_inkml_files(input_path, output_path):
    """
    Process InkML files in a directory to generate smooth PNG images and a JSON file mapping filenames to labels.

    Args:
        input_path (str): Path to the directory containing InkML files.
        output_path (str): Path to save PNG images and JSON file.
    """
    os.makedirs(output_path, exist_ok=True)
    json_path = os.path.join(output_path, "normalized.json")
    label_data = {}

    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith(".inkml"):
                file_path = os.path.join(root, file)

                try:
                    strokes, label, sample_id = parse_inkml(file_path)

                    if not label:
                        print(f"Skipping file {file_path} due to missing label")
                        continue

                    # Generate 3 different random widths between 3 and 9
                    widths = random.sample(range(3, 10), 3)
                    
                    # Render three versions with different widths and random padding
                    for width in widths:
                        png_filename, image = render_strokes_flexible(strokes, sample_id, width)
                        
                        # Save the image
                        png_path = os.path.join(output_path, png_filename)
                        Image.fromarray(image).save(png_path, optimize=True)
                        
                        # Add entry to the JSON data
                        label_data[png_filename] = label

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Save the JSON file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(label_data, f, indent=4)

if __name__ == "__main__":
    input_path = r""     # path to dataset containing inkml files
    output_path = r""    # output path for extracted images
    process_inkml_files(input_path, output_path)
