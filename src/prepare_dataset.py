"""
Dataset Preparation Script for Helmet Detection
Downloads and prepares helmet detection dataset from public sources
"""

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import yaml
import requests
from sklearn.model_selection import train_test_split
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HelmetDatasetPreparer:
    """Prepare helmet detection dataset from various sources."""
    
    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize dataset preparer.
        
        Args:
            output_dir: Output directory for processed dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create split directories
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.test_dir = self.output_dir / "test"
        
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            split_dir.mkdir(exist_ok=True)
            (split_dir / "images").mkdir(exist_ok=True)
            (split_dir / "labels").mkdir(exist_ok=True)
        
        # Class mapping
        self.class_names = {0: 'helmet', 1: 'no-helmet', 2: 'motorcycle'}
        self.class_to_id = {v: k for k, v in self.class_names.items()}
    
    def download_kaggle_dataset(self, dataset_url: str, dest_path: str) -> None:
        """
        Download dataset from Kaggle.
        
        Args:
            dataset_url: Kaggle dataset URL
            dest_path: Destination path
        """
        logger.info(f"Downloading dataset from {dataset_url}")
        
        # Note: Kaggle datasets require authentication
        # This is a placeholder - user needs to provide their own data
        # or use kaggle CLI: kaggle datasets download -d <dataset> -p <path>
        
        logger.warning(
            "Kaggle download requires authentication. "
            "Please download manually or use kaggle CLI."
        )
        
        # Alternative: Use direct download links if available
        # For this example, we'll create a placeholder structure
        pass
    
    def prepare_from_yolo(self, source_dir: str, class_mapping: Dict[str, int] = None):
        """
        Prepare dataset from YOLO format.
        
        Args:
            source_dir: Source directory with images and labels
            class_mapping: Optional custom class mapping
        """
        source_path = Path(source_dir)
        
        if not source_path.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        images = []
        for ext in image_extensions:
            images.extend(source_path.glob(f"*{ext}"))
        
        logger.info(f"Found {len(images)} images in {source_dir}")
        
        # Split dataset
        train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
        
        # Copy files to respective directories
        splits = {
            'train': (train_imgs, self.train_dir),
            'val': (val_imgs, self.val_dir),
            'test': (test_imgs, self.test_dir)
        }
        
        split_counts = defaultdict(lambda: defaultdict(int))
        
        for split_name, (img_list, split_dir) in splits.items():
            for img_path in img_list:
                # Copy image
                img_dest = split_dir / "images" / img_path.name
                shutil.copy2(img_path, img_dest)
                
                # Copy corresponding label if exists
                label_path = img_path.with_suffix('.txt')
                if label_path.exists():
                    label_dest = split_dir / "labels" / label_path.name
                    shutil.copy2(label_path, label_dest)
                    
                    # Count classes in label
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                class_name = self.class_names.get(class_id, f'class_{class_id}')
                                split_counts[split_name][class_name] += 1
            
            logger.info(f"{split_name}: {len(img_list)} images")
        
        # Log class distribution
        logger.info("Class distribution:")
        for split_name, counts in split_counts.items():
            logger.info(f"  {split_name}: {dict(counts)}")
        
        return split_counts
    
    def prepare_from_coco(self, coco_json_path: str, images_dir: str):
        """
        Prepare dataset from COCO format.
        
        Args:
            coco_json_path: Path to COCO format JSON
            images_dir: Directory containing images
        """
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Create category mapping
        category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Group annotations by image
        image_annotations = defaultdict(list)
        for ann in coco_data['annotations']:
            image_annotations[ann['image_id']].append(ann)
        
        # Get all images
        images = {img['id']: img for img in coco_data['images']}
        
        logger.info(f"Found {len(images)} images, {len(coco_data['annotations'])} annotations")
        
        # Split images
        image_ids = list(images.keys())
        train_ids, temp_ids = train_test_split(image_ids, test_size=0.3, random_state=42)
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
        
        # Convert to YOLO format and save
        splits = {
            'train': (train_ids, self.train_dir),
            'val': (val_ids, self.val_dir),
            'test': (test_ids, self.test_dir)
        }
        
        for split_name, (ids, split_dir) in splits.items():
            for img_id in ids:
                img_info = images[img_id]
                
                # Copy image
                src_img = Path(images_dir) / img_info['file_name']
                dst_img = split_dir / "images" / img_info['file_name']
                shutil.copy2(src_img, dst_img)
                
                # Convert annotations to YOLO format
                annotations = image_annotations.get(img_id, [])
                label_path = split_dir / "labels" / (Path(img_info['file_name']).stem + '.txt')
                
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        # Get category name
                        cat_name = category_map.get(ann['category_id'], 'unknown')
                        
                        # Map to our class names
                        if 'helmet' in cat_name.lower() and 'no' not in cat_name.lower():
                            class_id = 0
                        elif 'no-helmet' in cat_name.lower() or 'no_helmet' in cat_name.lower():
                            class_id = 1
                        elif 'motorcycle' in cat_name.lower() or 'bike' in cat_name.lower():
                            class_id = 2
                        else:
                            continue
                        
                        # Convert COCO bbox to YOLO format
                        x, y, w, h = ann['bbox']
                        img_w = img_info['width']
                        img_h = img_info['height']
                        
                        # Normalize to [0, 1]
                        x_center = (x + w/2) / img_w
                        y_center = (y + h/2) / img_h
                        w_norm = w / img_w
                        h_norm = h / img_h
                        
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            
            logger.info(f"{split_name}: {len(ids)} images")
    
    def generate_data_yaml(self):
        """Generate data.yaml configuration file."""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = Path('configs/data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Data configuration saved to {yaml_path}")
        return yaml_path
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {}
        
        for split_name, split_dir in [('train', self.train_dir), 
                                       ('val', self.val_dir), 
                                       ('test', self.test_dir)]:
            images = list((split_dir / "images").glob("*"))
            labels = list((split_dir / "labels").glob("*.txt"))
            
            # Count classes
            class_counts = defaultdict(int)
            for label_path in labels:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_name = self.class_names.get(class_id, f'class_{class_id}')
                            class_counts[class_name] += 1
            
            stats[split_name] = {
                'images': len(images),
                'labels': len(labels),
                'class_counts': dict(class_counts)
            }
        
        return stats


def main():
    """Main dataset preparation entry point."""
    parser = argparse.ArgumentParser(description='Prepare helmet detection dataset')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to source dataset (YOLO format directory)')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Output directory for processed dataset')
    parser.add_argument('--generate-config', action='store_true',
                        help='Generate data.yaml configuration file')
    
    args = parser.parse_args()
    
    # Initialize preparer
    preparer = HelmetDatasetPreparer(args.output)
    
    # Prepare dataset
    preparer.prepare_from_yolo(args.source)
    
    # Generate configuration
    if args.generate_config:
        preparer.generate_data_yaml()
    
    # Print statistics
    stats = preparer.get_statistics()
    print("\nDataset Statistics:")
    print("="*50)
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Images: {split_stats['images']}")
        print(f"  Labels: {split_stats['labels']}")
        print(f"  Classes: {split_stats['class_counts']}")
    print("="*50)


if __name__ == '__main__':
    main()
