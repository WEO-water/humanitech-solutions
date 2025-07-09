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

import asyncio
from tenacity import retry, wait_random_exponential, stop_after_attempt

from prompt_filter_comments import filter_comments
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

async def main():

    # Read comments data
    filtered_df = gpd.read_file(COMMENTS_PTH, 
                                dtype={'text': 'str'})

    print(f"Number of rows in filtered DataFrame: {len(filtered_df)}")

    tasks = []
    for idx, row in filtered_df.iterrows():
        logging.debug(f"Processing row {idx}")
        tasks.append(filter_comments(idx, row['text'], print_output=True))

    print(f"Starting {len(tasks)} asynchronous LLM generation tasks...")

    results = await asyncio.gather(*tasks)
    print("All LLM tasks completed.")

    for idx, response_text in results:
            
        if isinstance(response_text, str):
            output_dict = json.loads(response_text)
        else:
            output_dict = response_text
            
        if output_dict.get('useful', 0) == 0:
            logging.debug(f"Row {idx} is not useful, delete it and continue.")
            filtered_df = filtered_df[filtered_df.index != idx]
            continue

            
    # Save the filtered DataFrame to a GeoJSON file
    #filtered_df.to_file(OUT_VECTOR, driver="GeoJSON")
    #print(f"GeoJSON file {OUT_VECTOR} created.")


if __name__ == "__main__":

    COMMENTS_TS = '20250708_164650'

    #Inputs
    COMMENTS_PTH = f"gs://dl-test-439308-bucket/weo-data/dashboard/comments_{COMMENTS_TS}.zip"

    #Outputs
    CHECKPOINT_FILE = f"comments_filtering_checkpoint_{COMMENTS_TS}.geojson"
    OUT_VECTOR = f"comments_filtered_{COMMENTS_TS}.geojson"

    try:
        asyncio.run(main())
        logging.info("All comments processed successfully.")

    except Exception as e:
        logging.error(f"Error occurred in main: {e}")


    # zip the geojsonfile
    #os.system(f"zip comments.zip {OUT_VECTOR}")

    # upload this into GCS bucket:
    #upload_to_gcs_with_timestamp(bucket_name="dl-test-439308-bucket", local_file_path="comments.zip", gcs_prefix="weo-data/dashboard/")