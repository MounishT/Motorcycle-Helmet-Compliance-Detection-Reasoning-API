"""
Generate sample test images for API testing
Creates simple test images with shapes for validation
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def create_sample_image(filename: str, width: int = 640, height: int = 640,
                       scenario: str = "helmet_detected"):
    """
    Create a sample test image.
    
    Args:
        filename: Output filename
        width: Image width
        height: Image height
        scenario: Test scenario type
    """
    # Create image with background
    img = Image.new('RGB', (width, height), color=(135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(img)
    
    # Draw ground
    draw.rectangle([0, height-100, width, height], fill=(34, 139, 34))  # Green ground
    
    if scenario == "helmet_detected":
        # Draw motorcycle
        draw.rectangle([200, 350, 440, 450], fill=(50, 50, 50))  # Body
        draw.rectangle([250, 400, 390, 450], fill=(30, 30, 30))  # Wheels area
        
        # Draw rider body
        draw.rectangle([280, 250, 360, 400], fill=(0, 0, 255))  # Blue shirt
        
        # Draw head with helmet
        draw.ellipse([290, 180, 350, 250], fill=(255, 165, 0))  # Orange helmet
        draw.rectangle([300, 220, 340, 250], fill=(255, 140, 0))  # Helmet visor
        
        # Add label
        draw.text((280, 150), "Helmet Detected", fill=(0, 255, 0))
        
    elif scenario == "no_helmet":
        # Draw motorcycle
        draw.rectangle([200, 350, 440, 450], fill=(50, 50, 50))
        draw.rectangle([250, 400, 390, 450], fill=(30, 30, 30))
        
        # Draw rider body
        draw.rectangle([280, 250, 360, 400], fill=(255, 0, 0))  # Red shirt
        
        # Draw head WITHOUT helmet
        draw.ellipse([290, 180, 350, 250], fill=(255, 218, 185))  # Skin color
        draw.rectangle([300, 180, 340, 200], fill=(139, 69, 19))  # Hair
        
        # Add label
        draw.text((280, 150), "No Helmet!", fill=(255, 0, 0))
        
    elif scenario == "multiple_riders":
        # Rider 1 (with helmet)
        draw.rectangle([100, 350, 280, 450], fill=(50, 50, 50))
        draw.rectangle([140, 250, 220, 400], fill=(0, 128, 0))  # Green shirt
        draw.ellipse([150, 180, 210, 250], fill=(255, 165, 0))  # Orange helmet
        
        # Rider 2 (without helmet)
        draw.rectangle([350, 350, 530, 450], fill=(50, 50, 50))
        draw.rectangle([390, 250, 470, 400], fill=(128, 0, 128))  # Purple shirt
        draw.ellipse([400, 180, 460, 250], fill=(255, 218, 185))  # Skin (no helmet)
        
        # Add labels
        draw.text((130, 150), "Helmet", fill=(0, 255, 0))
        draw.text((380, 150), "No Helmet", fill=(255, 0, 0))
        
    elif scenario == "no_objects":
        # Just scenery
        draw.text((250, 300), "No riders here", fill=(255, 255, 255))
        
    # Add timestamp
    draw.text((10, 10), "Test Image for Helmet Detection API", fill=(255, 255, 255))
    draw.text((10, 30), f"Scenario: {scenario}", fill=(255, 255, 255))
    
    # Save image
    img.save(filename, quality=95)
    print(f"Created: {filename}")


def main():
    """Create all sample test images."""
    # Create tests directory if it doesn't exist
    tests_dir = Path("tests")
    tests_dir.mkdir(exist_ok=True)
    
    # Create sample images
    scenarios = [
        ("sample_helmet.jpg", "helmet_detected"),
        ("sample_no_helmet.jpg", "no_helmet"),
        ("sample_multiple.jpg", "multiple_riders"),
        ("sample_empty.jpg", "no_objects"),
    ]
    
    for filename, scenario in scenarios:
        filepath = tests_dir / filename
        create_sample_image(str(filepath), scenario=scenario)
    
    print("\nSample images created successfully!")
    print("Use these to test the API endpoints.")


if __name__ == '__main__':
    main()
