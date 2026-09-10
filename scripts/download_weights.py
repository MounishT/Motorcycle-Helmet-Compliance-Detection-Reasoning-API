"""
Download Model Weights Script
Downloads pretrained or trained model weights from GitHub Releases
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, dest_path: str, chunk_size: int = 8192) -> bool:
    """
    Download file from URL with progress bar.
    
    Args:
        url: Download URL
        dest_path: Destination file path
        chunk_size: Download chunk size
    
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Create progress bar
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
    
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def main():
    """Download model weights."""
    # Weights directory
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    # Model weights URLs - hosted on GitHub Releases
    weights_urls = {
        "best.pt": "https://github.com/MounishT/Motorcycle-Helmet-Compliance-Detection-Reasoning-API/releases/download/v1.0.0/best.pt",
    }
    
    print("Downloading model weights...")
    print("="*50)
    
    for filename, url in weights_urls.items():
        dest_path = weights_dir / filename
        
        if dest_path.exists():
            print(f"{filename} already exists, skipping...")
            continue
        
        print(f"Downloading {filename}...")
        print(f"URL: {url}")
        if download_file(url, str(dest_path)):
            print(f"Successfully downloaded {filename}")
        else:
            print(f"Failed to download {filename}")
            print("")
            print("Manual download instructions:")
            print(f"1. Go to: https://github.com/MounishT/Motorcycle-Helmet-Compliance-Detection-Reasoning-API/releases")
            print(f"2. Download '{filename}' from the latest release")
            print(f"3. Place it in: {weights_dir.absolute()}")
            sys.exit(1)
    
    print("="*50)
    print("All weights downloaded successfully!")
    print("\nYou can now run the API:")
    print("  uvicorn api.main:app --host 0.0.0.0 --port 8000")


if __name__ == '__main__':
    main()
