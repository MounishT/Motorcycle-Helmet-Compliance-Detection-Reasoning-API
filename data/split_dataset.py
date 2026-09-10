"""
Dataset Split Script for Helmet Detection
Splits dataset into train/val/test with seed=42 for reproducibility
"""

import os
import shutil
import argparse
import yaml
from pathlib import Path
from typing import Tuple, Dict
import random
import numpy as np


class DatasetSplitter:
    """Splits dataset into train/val/test sets."""
    
    def __init__(self, source_dir: str = "data/raw", output_dir: str = "data/processed", 
                 seed: int = 42):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Default split ratios: 70/15/15
        self.train_ratio = 0.70
        self.val_ratio = 0.15
        self.test_ratio = 0.15
    
    def get_image_label_pairs(self) -> list:
        """Find matching image-label pairs in the dataset."""
        pairs = []
        
        images_dir = self.source_dir / "images"
        labels_dir = self.source_dir / "labels"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                # Find corresponding label file
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    pairs.append((img_path, label_path))
        
        print(f"Found {len(pairs)} image-label pairs")
        return pairs
    
    def split_dataset(self) -> Dict[str, int]:
        """
        Split dataset into train/val/test sets.
        
        Returns:
            Dictionary with counts for each split
        """
        # Get all image-label pairs
        pairs = self.get_image_label_pairs()
        
        if len(pairs) == 0:
            raise ValueError("No image-label pairs found in the dataset")
        
        # Shuffle pairs
        random.shuffle(pairs)
        
        # Calculate split indices
        total = len(pairs)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)
        
        # Split pairs
        train_pairs = pairs[:train_end]
        val_pairs = pairs[train_end:val_end]
        test_pairs = pairs[val_end:]
        
        print(f"Split sizes - Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
        
        # Create output directories
        stats = {"train": 0, "val": 0, "test": 0}
        
        for split_name, split_pairs in [("train", train_pairs), 
                                        ("val", val_pairs), 
                                        ("test", test_pairs)]:
            split_dir = self.output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"
            images_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)
            
            # Copy files
            for img_path, label_path in split_pairs:
                shutil.copy2(img_path, images_dir / img_path.name)
                shutil.copy2(label_path, labels_dir / label_path.name)
            
            stats[split_name] = len(split_pairs)
            print(f"  {split_name}: {len(split_pairs)} pairs")
        
        return stats
    
    def generate_data_yaml(self) -> str:
        """Generate data.yaml for YOLO training."""
        data_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 3,
            'names': {0: 'helmet', 1: 'no-helmet', 2: 'motorcycle'}
        }
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"Generated data.yaml at {yaml_path}")
        return str(yaml_path)
    
    def validate_split(self) -> bool:
        """Validate that the split is correct."""
        required_dirs = ['train/images', 'train/labels', 
                        'val/images', 'val/labels',
                        'test/images', 'test/labels']
        
        for dir_path in required_dirs:
            full_path = self.output_dir / dir_path
            if not full_path.exists():
                print(f"Missing directory: {full_path}")
                return False
        
        # Check that counts match
        train_imgs = len(list((self.output_dir / "train/images").iterdir()))
        val_imgs = len(list((self.output_dir / "val/images").iterdir()))
        test_imgs = len(list((self.output_dir / "test/images").iterdir()))
        
        total = train_imgs + val_imgs + test_imgs
        print(f"Validation: {train_imgs} train + {val_imgs} val + {test_imgs} test = {total} total")
        
        return total > 0


def main():
    """Main split function."""
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test')
    parser.add_argument('--source', type=str, default='data/raw',
                       help='Source dataset directory')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--generate-config', action='store_true',
                       help='Generate data.yaml configuration')
    
    args = parser.parse_args()
    
    splitter = DatasetSplitter(args.source, args.output, args.seed)
    
    try:
        stats = splitter.split_dataset()
        
        if args.generate_config:
            splitter.generate_data_yaml()
        
        if splitter.validate_split():
            print("Dataset split completed successfully!")
        else:
            print("Warning: Dataset split validation failed!")
            
    except Exception as e:
        print(f"Error during split: {e}")
        raise


if __name__ == "__main__":
    main()
