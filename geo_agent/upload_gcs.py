import geopandas as gpd
import pandas as pd
import datetime
import os
import argparse

# Import Google Cloud Storage client library
from google.cloud import storage

# to do in terminal before running this script:
# export GOOGLE_APPLICATION_CREDENTIALS="/home/robin-hamers/.config/gcloud/application_default_credentials.json"
# bucket_name = "dl-test-439308-bucket"
# gcs_prefix = "weo-data/dashboard/"

# example for running this script:
# python3 upload_gcs.py /home/robin-hamers/Downloads/weo-data_dashboard_comments_20250708_160451/comments.zip dl-test-439308-bucket --gcs_prefix weo-data/dashboard/

def upload_gcs(bucket_name, local_file_path, gcs_prefix=""):
    """
    Uploads a local file to a GCS bucket with a timestamp appended to its name.

    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        local_file_path (str): The path to the local file to upload (e.g., 'my_data.csv').
        gcs_prefix (str): An optional prefix/folder path within the GCS bucket
                          (e.g., 'data/', 'exports/').
    Returns:
        bool: True if the upload was successful, False otherwise.
    """
    try:
        # Initialize the GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Get the base name of the local file (e.g., "my_data.csv")
        base_file_name = os.path.basename(local_file_path)

        # Generate a timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Example: 20250708_112721

        # Split the base name to insert timestamp before extension
        name_part, ext_part = os.path.splitext(base_file_name)

        # Construct the new GCS filename with timestamp
        gcs_filename = f"{name_part}_{timestamp}{ext_part}"

        # Combine prefix and filename for the final GCS destination
        if gcs_prefix:
            # Ensure prefix ends with a slash and uses forward slashes for GCS
            if not gcs_prefix.endswith('/'):
                gcs_prefix += '/'
            destination_blob_name = gcs_prefix + gcs_filename
        else:
            destination_blob_name = gcs_filename

        # Get the blob (object) in the bucket
        blob = bucket.blob(destination_blob_name)

        # Upload the file
        blob.upload_from_filename(local_file_path)

        out_path = f"gs://{bucket_name}/{destination_blob_name}"

        print(f"File '{local_file_path}' successfully uploaded {out_path}")

        return out_path

    except Exception as e:
        print(f"Error uploading '{local_file_path}' to GCS: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a local file to GCP bucket with a timestamped name.")
    parser.add_argument("local_file_path", type=str, help="Path to the local file to upload.")
    parser.add_argument("gcs_bucket", type=str, help="Name of the GCP bucket to upload the file to.")
    parser.add_argument("--gcs_prefix", type=str, default="",
                        help="Optional prefix/folder path within the GCS bucket (e.g., 'data/', 'exports/').")

    args = parser.parse_args()

    # Call the upload function
    upload_gcs(args.gcs_bucket, args.local_file_path, args.gcs_prefix)