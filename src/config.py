import os

# Base directory (root of your project)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Data paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "california-housing.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Artifacts directories
PREPROCESSOR_DIR = os.path.join(BASE_DIR, "artifacts", "preprocessor")
