"""
Project verification script
Checks that all required files are present and valid
"""

import os
import sys
from pathlib import Path


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists and print status."""
    if Path(filepath).exists():
        print(f"  [OK] {description}: {filepath}")
        return True
    else:
        print(f"  [MISSING] {description}: {filepath}")
        return False


def main():
    """Verify project structure."""
    print("="*60)
    print("PROJECT STRUCTURE VERIFICATION")
    print("="*60)
    
    # Required files
    required_files = [
        ("requirements.txt", "Dependencies"),
        ("Dockerfile", "Docker configuration"),
        ("docker-compose.yml", "Docker Compose"),
        ("README.md", "Documentation"),
        ("memo.md", "2-page memo"),
        ("LICENSE", "License file"),
        ("Makefile", "Make commands"),
        ("setup.cfg", "Pytest/Black config"),
        (".gitignore", "Git ignore rules"),
        ("CHANGELOG.md", "Change log"),
        ("REPRODUCE.md", "Reproduction guide"),
        ("submission.txt", "Submission summary"),
        ("configs/data.yaml", "Dataset configuration"),
        ("src/__init__.py", "Source package"),
        ("src/train.py", "Training script"),
        ("src/evaluate.py", "Evaluation script"),
        ("src/inference.py", "Inference module"),
        ("src/reasoning.py", "Reasoning layer"),
        ("src/prepare_dataset.py", "Dataset preparation"),
        ("api/__init__.py", "API package"),
        ("api/main.py", "FastAPI application"),
        ("tests/__init__.py", "Tests package"),
        ("tests/test_inference.py", "Inference tests"),
        ("tests/test_api.py", "API tests"),
        ("tests/create_sample_images.py", "Sample image generator"),
        ("scripts/download_weights.py", "Weight downloader"),
        ("notebooks/eda_analysis.ipynb", "EDA notebook"),
        ("weights/.gitkeep", "Weights directory"),
        ("data/raw/.gitkeep", "Raw data directory"),
        ("data/processed/.gitkeep", "Processed data directory"),
    ]
    
    # Check all files
    all_files_ok = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_files_ok = False
    
    print("\n" + "="*60)
    
    if all_files_ok:
        print("SUCCESS: All required files are present!")
    else:
        print("WARNING: Some files are missing.")
        print("Please ensure all files are present before submission.")
    
    print("="*60)
    
    # Check file sizes
    print("\nFile sizes:")
    for filepath, _ in required_files:
        if Path(filepath).exists():
            size = Path(filepath).stat().st_size
            if size > 1024:
                print(f"  {filepath}: {size/1024:.1f} KB")
            else:
                print(f"  {filepath}: {size} bytes")
    
    print("\n" + "="*60)
    print("Verification complete!")
    print("="*60)


if __name__ == '__main__':
    main()
