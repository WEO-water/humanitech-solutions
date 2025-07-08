import os
import pandas as pd
import geopandas as gpd
import sys
from pydantic import BaseModel
from typing import List
import h3pandas
import glob
from functools import reduce
import matplotlib.pyplot as plt
import json
from shapely import wkt
import time


from prompt_func import filter_comments
from upload_gcs import upload_to_gcs_with_timestamp
from tqdm import tqdm
import logging

# Set up logging to file only
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("run_filter_comments.log", mode='a')
    ]
)

def main():

    # Read comments data
    filtered_df = gpd.read_file(COMMENTS_PTH, 
                                dtype={'text': 'str'})

    print(f"Number of rows in filtered DataFrame: {len(filtered_df)}")

    for idx, row in filtered_df.iterrows():
        try:
            logging.debug(f"Processing row {idx}")

            output = filter_comments(
                comment=row['text'],
                print_output=False
            )

            if isinstance(output, str):
                output_dict = json.loads(output)
            else:
                output_dict = output
            
            if output_dict.get('useful', 0) == 0:
                logging.debug(f"Row {idx} is not useful, delete it and continue.")
                filtered_df = filtered_df[filtered_df.index != idx]
                continue

            
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                print(f"RESOURCE_EXHAUSTED error at row {idx}, sleeping 60s and retrying...", file=sys.stderr)
                time.sleep(60)
                continue  # This will re-enter the for loop with the same row (since nothing was added to processed_h3_indices)
            else:
                print(f"Error processing row {idx}: {e}", file=sys.stderr)
                continue
            
    # Save the filtered DataFrame to a GeoJSON file
    filtered_df.to_file(OUT_VECTOR, driver="GeoJSON")
    print(f"GeoJSON file {OUT_VECTOR} created.")


if __name__ == "__main__":

    COMMENTS_TS = '20250708_123244'

    #Inputs
    COMMENTS_PTH = f"gs://dl-test-439308-bucket/weo-data/dashboard/comments_{COMMENTS_TS}.zip"

    #Outputs
    CHECKPOINT_FILE = f"comments_filtering_checkpoint_{COMMENTS_TS}.geojson"
    OUT_VECTOR = f"comments_filtered_{COMMENTS_TS}.geojson"

    main()

    # zip the geojsonfile
    os.system(f"zip comments.zip {OUT_VECTOR}")

    # upload this into GCS bucket:
    upload_to_gcs_with_timestamp(bucket_name="dl-test-439308-bucket", local_file_path="comments.zip", gcs_prefix="weo-data/dashboard/")