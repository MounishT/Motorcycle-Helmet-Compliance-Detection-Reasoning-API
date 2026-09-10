"""
RT-DETR Training Script for Helmet Detection
Fine-tunes RT-DETR on custom helmet compliance dataset
"""

import os
import sys
import yaml
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
import numpy as np
from ultralytics import RTDETR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train(config_path: str, epochs: int = 30, batch_size: int = 8,
          img_size: int = 640, device: str = None) -> Dict[str, Any]:
    """
    Train RT-DETR model on helmet detection dataset.
    
    Args:
        config_path: Path to data configuration YAML
        epochs: Number of training epochs
        batch_size: Training batch size
        img_size: Input image size
        device: Training device (auto-detect if None)
    
    Returns:
        Dictionary with training results and metrics
    """
    # Load configuration
    config = load_config(config_path)
    data_yaml = os.path.abspath(config_path)
    
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Initialize RT-DETR model with pretrained weights
    logger.info("Initializing RT-DETR model...")
    model = RTDETR('rtdetr-l.yaml', pretrained=True)
    
    # Update model configuration
    model.num_classes = config['nc']
    
    # Training hyperparameters
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'optimizer': 'AdamW',
        'lr0': 1e-4,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 1e-4,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'cos_lr': True,
        'close_mosaic': 10,
        'amp': True,
        'seed': 42,
        'project': 'runs',
        'name': 'helmet_detection',
        'exist_ok': True,
        'pretrained': True,
        'workers': 8,
        'cache': False,
        'verbose': True,
        'patience': 20,
        'save': True,
        'save_period': -1,
        'val': True,
        'plots': True
    }
    
    logger.info(f"Training configuration: {json.dumps(train_args, indent=2)}")
    
    # Record training start time
    start_time = datetime.now()
    
    # Train the model
    logger.info("Starting training...")
    results = model.train(**train_args)
    
    # Calculate training duration
    end_time = datetime.now()
    training_duration = end_time - start_time
    logger.info(f"Training completed in {training_duration}")
    
    # Extract metrics
    metrics = {
        'best_fitness': float(results.results_dict.get('best_fitness', 0)),
        'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
        'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
        'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
        'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
        'training_duration': str(training_duration),
        'epochs': epochs,
        'batch_size': batch_size,
        'img_size': img_size,
        'device': device
    }
    
    # Save training metadata
    metadata = {
        'config': config,
        'hyperparameters': train_args,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'seed': 42
    }
    
    metadata_path = Path('weights') / 'training_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save best model to weights directory
    best_model_path = Path('runs/helmet_detection/weights/best.pt')
    if best_model_path.exists():
        import shutil
        dest_path = Path('weights/best.pt')
        shutil.copy2(best_model_path, dest_path)
        logger.info(f"Best model saved to {dest_path}")
    
    logger.info(f"Training metrics: {json.dumps(metrics, indent=2)}")
    
    return metrics


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train RT-DETR for helmet detection')
    parser.add_argument('--config', type=str, default='configs/data.yaml',
                        help='Path to data configuration YAML')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--device', type=str, default=None,
                        help='Training device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Run training
    metrics = train(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device
    )
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print(f"Best mAP@0.5: {metrics['mAP50']:.4f}")
    print(f"Best mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    print(f"Training duration: {metrics['training_duration']}")
    print("="*50)


if __name__ == '__main__':
    main()
