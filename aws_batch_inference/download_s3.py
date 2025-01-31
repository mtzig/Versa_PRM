import os
import boto3
from pathlib import Path

def download_s3_folder(bucket_name, s3_folder, local_dir):
    # Create local directory if it doesn't exist
    local_file_save_dir = os.path.join(local_dir, s3_folder)
    Path(local_file_save_dir).mkdir(parents=True, exist_ok=True)

    s3 = boto3.client('s3')

    # List all objects in the specified S3 folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)
    if 'Contents' not in response:
        print(f"No files found in S3 folder: {s3_folder}")
        return

    for obj in response['Contents']:
        s3_file_path = obj['Key']
        # Skip directories
        if s3_file_path.endswith('/'):
            continue

        file_name = os.path.basename(s3_file_path)
        local_file_path = os.path.join(local_file_save_dir, file_name)

        # Download the file from S3
        print(f"Downloading {s3_file_path} to {local_file_path}...")
        s3.download_file(bucket_name, s3_file_path, local_file_path)

    print(f"Download complete. Files are saved in {local_file_save_dir}")

# Example usage
if __name__ == "__main__":
    output_bucket = ""  # Your output S3 bucket
    local_output_dir = ""  # Local folder to save the downloaded files
    output_folder = ""
    download_s3_folder(output_bucket, output_folder, local_output_dir)

    
