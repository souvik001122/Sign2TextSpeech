#!/usr/bin/env python3
"""
Download helper for Sign Language Recognition assets.
Downloads the trained model and dataset from GitHub Releases or cloud storage.
"""

import os
import sys
import urllib.request
from pathlib import Path
import zipfile

# Asset information
ASSETS = {
    "model": {
        "filename": "cnn8grps_rad1_model.h5",
        "url": "https://github.com/souvik001122/Sign2TextSpeech/releases/download/v1.0/cnn8grps_rad1_model.h5",
        "size_mb": 13,
        "required": True
    },
    "dataset": {
        "filename": "AtoZ_3.1.zip",
        "url": "https://github.com/souvik001122/Sign2TextSpeech/releases/download/v1.0/AtoZ_3.1.zip",
        "size_mb": 72,
        "required": False
    }
}


def download_file(url, filename, show_progress=True):
    """Download a file with progress indication."""
    print(f"Downloading {filename}...")
    
    def report_progress(block_num, block_size, total_size):
        if show_progress and total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f'\r[{bar}] {percent:.1f}% ({downloaded/(1024*1024):.1f}MB)', end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, filename, report_progress)
        print(f"\n✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"\n✗ Failed to download {filename}: {e}")
        return False


def extract_zip(zip_path, extract_to="."):
    """Extract a ZIP file."""
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"✗ Failed to extract {zip_path}: {e}")
        return False


def main():
    """Main download logic."""
    print("=" * 60)
    print("Sign Language Recognition - Asset Download Helper")
    print("=" * 60)
    
    # Determine what to download
    download_dataset = False
    if len(sys.argv) > 1 and sys.argv[1] == "--dataset":
        download_dataset = True
        print("\nMode: Downloading model + dataset")
    else:
        print("\nMode: Downloading model only")
        print("(Use --dataset flag to also download the dataset)")
    
    print()
    
    # Download model
    model_info = ASSETS["model"]
    if not os.path.exists(model_info["filename"]):
        print(f"Model file not found. Downloading (~{model_info['size_mb']}MB)...")
        if not download_file(model_info["url"], model_info["filename"]):
            print("\n⚠ Model download failed. Please download manually from:")
            print(f"   {model_info['url']}")
            return 1
    else:
        print(f"✓ Model already exists: {model_info['filename']}")
    
    # Download dataset if requested
    if download_dataset:
        dataset_info = ASSETS["dataset"]
        zip_path = dataset_info["filename"]
        extract_folder = "AtoZ_3.1"
        
        if os.path.exists(extract_folder):
            print(f"✓ Dataset already exists: {extract_folder}/")
        else:
            if not os.path.exists(zip_path):
                print(f"\nDataset not found. Downloading (~{dataset_info['size_mb']}MB)...")
                if not download_file(dataset_info["url"], zip_path):
                    print("\n⚠ Dataset download failed. Please download manually from:")
                    print(f"   {dataset_info['url']}")
                    return 1
            
            # Extract
            if extract_zip(zip_path, "."):
                print(f"✓ Dataset ready: {extract_folder}/")
                # Optionally clean up zip
                # os.remove(zip_path)
    
    print("\n" + "=" * 60)
    print("✓ Setup complete! You can now run the application.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
