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
from upload_gcs import upload_gcs
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
    """
    Main async function to filter comments using LLM and save useful ones.
    """
    # Read comments data
    filtered_df = gpd.read_file(COMMENTS_PTH, dtype={'text': 'str'})
    logging.info(f"Number of rows in filtered DataFrame: {len(filtered_df)}")

    tasks = []
    for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Scheduling tasks"):
        logging.debug(f"Processing row {idx}")
        tasks.append(filter_comments(idx, row['text'], print_output=False))

    logging.info(f"Starting {len(tasks)} asynchronous LLM generation tasks...")

    results = await asyncio.gather(*tasks)
    logging.info("All LLM tasks completed.")

    indices_to_drop = []
    for idx, response_text in results:
        try:
            if isinstance(response_text, str):
                output_dict = json.loads(response_text)
            else:
                output_dict = response_text
        except Exception as e:
            logging.error(f"Failed to parse response for row {idx}: {e}")
            indices_to_drop.append(idx)
            continue

        if output_dict.get('useful', 0) == 0:
            logging.debug(f"Row {idx} is not useful, marking for deletion.")
            print(f"Row {idx} is not useful, marking for deletion.")
            print(filtered_df.iloc[idx]['text'])
            print(f"Response: {response_text}")
            print("--------------------------------------------------------")
            indices_to_drop.append(idx)
    keep_comments = input("Do you want to keep some of these comments? (Y/n): ")
    if keep_comments.lower() != "n":
        while True:
            comment_idx = input("Enter the index of the comment you want to keep (or 'done' to finish): ")
            if comment_idx.lower() == "done":
                break
            try:
                comment_idx = int(comment_idx)
                if comment_idx in indices_to_drop:
                    indices_to_drop.remove(comment_idx)
                    logging.info(f"Row {comment_idx} marked for keeping.")
                else:
                    logging.warning(f"Row {comment_idx} is not marked for deletion.")
            except ValueError:
                logging.error("Invalid input, please enter a valid index.")

    # Drop all non-useful rows at once
    filtered_df = filtered_df.drop(indices_to_drop)
    logging.info(f"Filtered DataFrame now has {len(filtered_df)} rows.")

    # Save the filtered DataFrame to a GeoJSON file
    filtered_df.to_file(OUT_VECTOR, driver="GeoJSON")
    logging.info(f"GeoJSON file {OUT_VECTOR} created.")


if __name__ == "__main__":

    COMMENTS_TS = '20250813_110132' # thats the original one for now

    #Inputs
    COMMENTS_PTH = f"gs://dl-test-439308-bucket/weo-data/dashboard/comments_{COMMENTS_TS}.zip"

    #Outputs
    CHECKPOINT_FILE = f"comments_filtering_checkpoint_{COMMENTS_TS}.geojson"
    OUT_VECTOR = f"comments_filtered_{COMMENTS_TS}_cleaned.geojson"

    try:
        asyncio.run(main())
        logging.info("All comments processed successfully.")

    except Exception as e:
        logging.error(f"Error occurred in main: {e}")


    # saving ?
    saving = input("Do you want to save the results? (Y/n): ")
    if saving.lower() != "n":
        # zip the geojsonfile
        os.system(f"zip comments.zip {OUT_VECTOR}")

        # upload this into GCS bucket:
        path = upload_gcs(bucket_name="dl-test-439308-bucket", local_file_path="comments.zip", gcs_prefix="weo-data/dashboard/")

        print("Uploaded to GCS succesfully. Path = " + path)