# Configuration for Sign Language Recognition Project

import os

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, "cnn8grps_rad1_model.h5")
WHITE_IMAGE_PATH = os.path.join(BASE_DIR, "white.jpg")

# Dataset paths (for training/data collection)
DATASET_ROOT = os.path.join(BASE_DIR, "AtoZ_3.1")

# Training data paths (if different location)
TRAINING_DATA_ROOT = "D:\\sign2text_dataset_3.0\\AtoZ_3.0"
TEST_DATA_ROOT = "D:\\test_data_2.0"

# Image processing settings
IMAGE_SIZE = 400
OFFSET = 29
HAND_OFFSET = 15

# Download URLs (update these with your actual hosting URLs)
MODEL_DOWNLOAD_URL = "https://github.com/souvik001122/Sign2TextSpeech/releases/download/v1.0/cnn8grps_rad1_model.h5"
DATASET_DOWNLOAD_URL = "https://github.com/souvik001122/Sign2TextSpeech/releases/download/v1.0/AtoZ_3.1.zip"
