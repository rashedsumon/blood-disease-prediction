import os
import shutil
import kagglehub

def download_and_prepare_dataset():
    """
    Downloads the blood cell anomaly dataset using kagglehub
    and returns the local path to the data.
    """
    print("Checking/Downloading dataset from Kaggle...")
    # Download latest version
    download_path = kagglehub.dataset_download("alitaqishah/blood-cell-anomaly-detection-2025")
    print("Path to dataset files:", download_path)
    
    # Return the path so other scripts can access the images
    return download_path

if __name__ == "__main__":
    # Test download if run directly
    path = download_and_prepare_dataset()