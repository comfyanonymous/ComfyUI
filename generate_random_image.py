#!/usr/bin/env python3
"""
Generate a random image with specified dimensions.
Default: 1920x1080 pixels
"""

import random
from PIL import Image, ImageDraw
import os

def generate_random_image(width=1920, height=1080, output_path="random_image.png"):
    """
    Generate a random image with random colors and patterns.
    
    Args:
        width: Image width in pixels (default: 1920)
        height: Image height in pixels (default: 1080)
        output_path: Output file path (default: random_image.png)
    """
    # Create a new image with random background
    image = Image.new('RGB', (width, height), color=(
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    ))
    
    draw = ImageDraw.Draw(image)
    
    # Draw random rectangles and circles
    for _ in range(random.randint(50, 150)):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        
        # Ensure x1 <= x2 and y1 <= y2
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        if random.choice([True, False]):
            draw.rectangle([x_min, y_min, x_max, y_max], fill=color, outline=color)
        else:
            radius = random.randint(10, 100)
            draw.ellipse([x_min, y_min, x_min+radius, y_min+radius], fill=color, outline=color)
    
    # Save the image
    image.save(output_path)
    print("Random image generated: {}".format(output_path))
    print("  Resolution: {}x{} pixels".format(width, height))
    return output_path

if __name__ == "__main__":
    output = generate_random_image(1920, 1080, "random_image.png")
