from google.cloud import storage
from pathlib import Path
import sys

def upload_folder(bucket_name, local_folder):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    folder = Path(local_folder)

    for file_path in folder.glob("*.csv"):
        blob = bucket.blob(f"raw/{file_path.name}")
        blob.upload_from_filename(file_path)
        print(f"Uploaded: {file_path.name}")

if __name__ == "__main__":
    bucket = sys.argv[1]
    folder = sys.argv[2]
    upload_folder(bucket, folder)
