"""
Evaluation Script for Helmet Detection Model
Computes mAP@0.5, precision, recall, F1, and confusion matrix
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import RTDETR
from sklearn.metrics import confusion_matrix, classification_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(weights_path: str, device: str = None) -> RTDETR:
    """Load trained RT-DETR model for evaluation."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Loading model from {weights_path}")
    model = RTDETR(weights_path)
    model.to(device)
    return model


def compute_metrics(results, class_names: List[str]) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        results: Ultralytics Results object
        class_names: List of class names
    
    Returns:
        Dictionary with computed metrics
    """
    metrics = {}
    
    # Extract confusion matrix data
    confusion_matrix_data = results.confusion_matrix.matrix
    metrics['confusion_matrix'] = confusion_matrix_data.tolist()
    
    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        tp = confusion_matrix_data[i][i]
        fp = sum(confusion_matrix_data[j][i] for j in range(len(class_names))) - tp
        fn = sum(confusion_matrix_data[i][j] for j in range(len(class_names))) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class_metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    metrics['per_class'] = per_class_metrics
    
    # Overall mAP@0.5
    metrics['mAP50'] = float(results.box.map50)
    metrics['mAP50-95'] = float(results.box.map)
    
    # Overall precision and recall
    metrics['precision'] = float(results.box.mp)
    metrics['recall'] = float(results.box.mr)
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                          save_path: str) -> None:
    """Plot and save confusion matrix visualization."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_per_class_metrics(per_class: Dict[str, Dict], 
                           save_path: str) -> None:
    """Plot per-class precision, recall, and F1 scores."""
    classes = list(per_class.keys())
    precisions = [per_class[c]['precision'] for c in classes]
    recalls = [per_class[c]['recall'] for c in classes]
    f1s = [per_class[c]['f1'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#2196F3')
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#4CAF50')
    bars3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#FF9800')
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title('Per-Class Detection Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Per-class metrics plot saved to {save_path}")


def evaluate(weights_path: str, data_yaml: str, split: str = 'test',
             device: str = None, save_plots: bool = True) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on test dataset.
    
    Args:
        weights_path: Path to trained model weights
        data_yaml: Path to data configuration
        split: Dataset split to evaluate ('val' or 'test')
        device: Evaluation device
        save_plots: Whether to save visualization plots
    
    Returns:
        Dictionary with all evaluation metrics
    """
    # Load model
    model = load_model(weights_path, device)
    
    # Get class names from config
    import yaml
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    class_names = config['names']
    
    logger.info(f"Evaluating on {split} split...")
    logger.info(f"Classes: {class_names}")
    
    # Run evaluation
    results = model.val(
        data=data_yaml,
        split=split,
        conf=0.25,
        iou=0.6,
        device=device or ('cuda' if torch.cuda.is_available() else 'cpu'),
        plots=True
    )
    
    # Compute metrics
    metrics = compute_metrics(results, list(class_names.values()))
    
    # Generate plots if requested
    if save_plots:
        eval_dir = Path('evaluation_results')
        eval_dir.mkdir(exist_ok=True)
        
        # Confusion matrix
        cm = np.array(metrics['confusion_matrix'])
        plot_confusion_matrix(cm, list(class_names.values()),
                             str(eval_dir / 'confusion_matrix.png'))
        
        # Per-class metrics
        plot_per_class_metrics(metrics['per_class'],
                              str(eval_dir / 'per_class_metrics.png'))
        
        # Save metrics to JSON
        metrics_path = eval_dir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"mAP@0.5:     {metrics['mAP50']:.4f}")
    print(f"mAP@0.5:0.95:{metrics['mAP50-95']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print("\nPer-Class Metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1-Score:  {class_metrics['f1']:.4f}")
    print("="*60)
    
    return metrics


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description='Evaluate RT-DETR helmet detection model')
    parser.add_argument('--weights', type=str, default='weights/best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--data', type=str, default='configs/data.yaml',
                        help='Path to data configuration YAML')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--device', type=str, default=None,
                        help='Evaluation device (cuda/cpu)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plot generation')
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = evaluate(
        weights_path=args.weights,
        data_yaml=args.data,
        split=args.split,
        device=args.device,
        save_plots=not args.no_plots
    )
    
    return metrics


if __name__ == '__main__':
    main()
