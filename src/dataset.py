import os
from dotenv import load_dotenv
from google.cloud import storage
from pathlib import Path

# Load environment variables
load_dotenv()

def fetch_data_from_gcs():
    """
    Fetch a file from GCS and store it locally.
    Uses ADC (Application Default Credentials) from `gcloud auth login`.
    """
    # Load variables from .env
    bucket_name = os.getenv("GCS_BUCKET")
    source_blob_name = os.getenv("SOURCE_BLOB_NAME")
    local_dir = "/Users/shubham/Desktop/Blog/california-housing-mlops/data/raw"

    if not all([bucket_name, source_blob_name, local_dir]):
        raise ValueError("❌ Missing one or more environment variables in .env file.")

    # Initialize GCS client (no explicit credentials needed)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # Ensure local directory exists
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    # Define local file path
    local_file_path = os.path.join(local_dir, os.path.basename(source_blob_name))
    
    # Download file
    print(f"⬇️ Downloading {source_blob_name} from bucket {bucket_name} ...")
    blob.download_to_filename(local_file_path)
    print(f"✅ File saved at: {local_file_path}")

if __name__ == "__main__":
    fetch_data_from_gcs()
