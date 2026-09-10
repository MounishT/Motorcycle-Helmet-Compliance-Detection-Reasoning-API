"""
Dataset Download Script for Helmet Detection
Downloads or creates sample dataset for training
"""

import os
import argparse
import yaml
import json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw


class DatasetDownloader:
    """Downloads or creates helmet detection dataset."""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Class definitions
        self.classes = {0: 'helmet', 1: 'no-helmet', 2: 'motorcycle'}
        self.nc = len(self.classes)
    
    def create_sample_dataset(self, num_images: int = 100) -> dict:
        """
        Create synthetic sample dataset for testing.
        
        Args:
            num_images: Number of sample images to create
            
        Returns:
            Dictionary with dataset statistics
        """
        print(f"Creating sample dataset with {num_images} images...")
        
        # Create directories
        images_dir = self.output_dir / "images"
        labels_dir = self.output_dir / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        stats = {"total_images": 0, "total_annotations": 0}
        
        for i in range(num_images):
            # Create random image
            width = np.random.randint(640, 1280)
            height = np.random.randint(480, 960)
            image = Image.new('RGB', (width, height), 
                            (np.random.randint(50, 200), 
                             np.random.randint(50, 200), 
                             np.random.randint(50, 200)))
            draw = ImageDraw.Draw(image)
            
            # Generate random annotations
            labels = []
            num_objects = np.random.randint(0, 5)
            
            for _ in range(num_objects):
                # Random class (0: helmet, 1: no-helmet, 2: motorcycle)
                class_id = np.random.choice([0, 1, 2])
                
                # Random bounding box
                x_center = np.random.uniform(0.1, 0.9)
                y_center = np.random.uniform(0.1, 0.9)
                bbox_width = np.random.uniform(0.05, 0.3)
                bbox_height = np.random.uniform(0.05, 0.4)
                
                # YOLO format: class_id x_center y_center width height
                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
                
                # Draw bounding box on image
                x1 = int((x_center - bbox_width/2) * width)
                y1 = int((y_center - bbox_height/2) * height)
                x2 = int((x_center + bbox_width/2) * width)
                y2 = int((y_center + bbox_height/2) * height)
                
                color = ['red', 'blue', 'green'][class_id]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Save image
            image_path = images_dir / f"sample_{i:04d}.jpg"
            image.save(image_path, quality=85)
            
            # Save label
            label_path = labels_dir / f"sample_{i:04d}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
            
            stats["total_images"] += 1
            stats["total_annotations"] += len(labels)
        
        print(f"Created {stats['total_images']} images with {stats['total_annotations']} annotations")
        return stats
    
    def generate_data_yaml(self, output_path: Optional[str] = None) -> str:
        """Generate data.yaml configuration file."""
        if output_path is None:
            output_path = str(self.output_dir / "data.yaml")
        
        data_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images',
            'val': 'images',
            'test': 'images',
            'nc': self.nc,
            'names': self.classes
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"Generated data.yaml at {output_path}")
        return output_path
    
    def create_yolo_structure(self) -> dict:
        """Create YOLO dataset directory structure."""
        structure = {}
        
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            split_dir.mkdir(exist_ok=True)
            (split_dir / 'images').mkdir(exist_ok=True)
            (split_dir / 'labels').mkdir(exist_ok=True)
            structure[split] = str(split_dir)
        
        print(f"Created YOLO dataset structure at {self.output_dir}")
        return structure


def main():
    """Main download function."""
    parser = argparse.ArgumentParser(description='Download or create helmet detection dataset')
    parser.add_argument('--source', type=str, default='sample',
                       choices=['sample', 'kaggle', 'roboflow'],
                       help='Dataset source')
    parser.add_argument('--output', type=str, default='data/raw',
                       help='Output directory')
    parser.add_argument('--num-images', type=int, default=100,
                       help='Number of sample images (for sample source)')
    parser.add_argument('--generate-config', action='store_true',
                       help='Generate data.yaml configuration')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.output)
    
    if args.source == 'sample':
        stats = downloader.create_sample_dataset(args.num_images)
    else:
        print(f"Source '{args.source}' not implemented yet. Use 'sample' for testing.")
        return
    
    if args.generate_config:
        downloader.generate_data_yaml()
    
    print("Dataset preparation complete!")


if __name__ == "__main__":
    main()
