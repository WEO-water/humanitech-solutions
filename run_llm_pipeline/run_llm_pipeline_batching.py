######### PROBLEM at the moment :
######### there is no possibility to ask for JSON as a response (only while prompting and this is not consistent)
######### - solution 1 : wait google to release this functionnality (should be released later)
######### - solution 2 : use another instance of gemini (lighter one) to transform the response into valid JSON if it is not
######### - solution 3 : same with a small model we deploy on vertex ai (could cost more if we don't use it a lot) 

import os
import sys
import time
import glob
import json
from functools import reduce
import re 

import pandas as pd
import geopandas as gpd
import h3pandas
import matplotlib.pyplot as plt
from shapely import wkt

import google.auth
from google import genai
from google.cloud.aiplatform import BatchPredictionJob
from google.oauth2 import service_account
from google.genai import types
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions, BatchJob, Content, Part

from datetime import datetime
from pydantic import BaseModel
from typing import List

import asyncio
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
import logging

from run_llm_pipeline.prompt_func import generate_risk_actions, map_data, cache_files, prepare_prompt_systemprompt_files_batch, RiskActions, RiskActions_and_explanation
from upload_gcs import upload_to_gcs_without_timestamp

# Constants
LOCAL_TMP_JSONL = "input.jsonl"  # Local temporary JSONL file path
PROJECT_ID = "dl-test-439308"  # 
MODEL_ID = "gemini-2.5-flash" 
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "europe-west1")
BUCKET_NAME = "dl-test-439308-bucket" 
INPUT_PREFIX = "rh_test/t1/"  # Prefix for input files in GCS
OUTPUT_PREFIX = "rh_test/t1/output/"  # Prefix for output files in GCS
INPUT_URI =  f"gs://{BUCKET_NAME}/{INPUT_PREFIX}input.jsonl"
OUTPUT_URI = f"gs://{BUCKET_NAME}/{OUTPUT_PREFIX}"
DATABASE_PTH = 'gs://dl-test-439308-bucket/weo-data/humanitech_dargo_filtered_database_250620.csv'
PDF_URI_LST = [
    'gs://dl-test-439308-bucket/weo-data/2021 Dargo, Census All persons QuickStats _ Australian Bureau of Statistics.pdf',
    'gs://dl-test-439308-bucket/weo-data/climate-ready-communities-a-guide-to-getting-started.pdf',
    'gs://dl-test-439308-bucket/weo-data/Dargo Rear v8 - final.pdf'
]
DATABASE_PTH = 'gs://dl-test-439308-bucket/weo-data/humanitech_dargo_filtered_database_250620.csv'

#Outputs
#CHECKPOINT_FILE = "rh_async_humanitech_dargo_emergency_solutions_v300_checkpoint.geojson"
OUT_VECTOR = "rh_batching_humanitech_dargo_emergency_solutions_v300.geojson"
    
#BUCKET = ""
#INPUT_JSONL_FILE_PATH = ""
#OUTPUT_JSONL_FILE_PATH = ""

EXPLAIN = True


# Set up logging to file only
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("run_llm_pipeline.log", mode='a')
    ]
)

def setup_gemini_client():    
    """Set up the Gemini client with the specified project and location."""

    credentials, project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )


    # Initialize the Gemini client
    client = genai.Client(
        project=PROJECT_ID,
        location=LOCATION,
        credentials=credentials,
        vertexai=True,
        http_options=HttpOptions(api_version='v1')
    )
    return client

def create_gemini_request(system_instruction, prompt, file_list, selected_response_schema):
    """
    Creates the request dictionary for Gemini API, including file data.

    Args:
        system_instruction (str): The system instruction text.
        prompt (str): The user prompt text.
        pdf_uris (str or list): A single PDF URI string or a list of PDF URI strings.

    Returns:
        dict: The populated request dictionary.
    """

    # Initialize the parts list with the prompt
    parts = [
        {
            "text": prompt
        }
    ]

    # Add file data to the parts list if files exist
    if file_list:
        for file_data in file_list:
            parts.append({
                "file_data": file_data
            })

    request_dict = {
        "system_instruction": {
            "parts": [
                {
                    "text": system_instruction
                }
            ]
        },
        "contents": {
            "role": "user",
            "parts": parts
        },
        "generation_config": {
            "temperature": 0.4,
            "topP": 0.95, # Note: in JSONL, it's 'topP', not 'top_p'
            "topK": 20,   # Note: in JSONL, it's 'topK', not 'top_k'
            "candidate_count": 1,
            # 'seed' is typically not a per-request parameter in batch JSONL
            # as it's often a job-level setting or not exposed for batch.
            # If determinism is critical, ensure the batch job itself supports a seed,
            # or consider a smaller number of single API calls.
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        #"response_mime_type": "application/json",
        #"response_schema": selected_response_schema,
    }
    return request_dict

def create_jsonl_file(explain=False):
    filtered_df = gpd.read_file(DATABASE_PTH, 
                                dtype={
            'heat_risk': 'Float64',
            'flood_risk': 'Int64',
            'fire_risk_202502': 'Int64'
        })
    filtered_df['geometry'] = filtered_df['geometry'].apply(wkt.loads)  # Convert WKT strings to geometries
    filtered_df = gpd.GeoDataFrame(filtered_df, crs='EPSG:4326') 
    for col in ['heat_risk', 'flood_risk', 'fire_risk_202502']:
        filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').astype('Int64')
    
    filtered_df = filtered_df.iloc[:100]  # For testing purposes, limit to first 100 rows
    print(f"Number of rows in filtered DataFrame: {len(filtered_df)}")

    # Create the general context string
    general_context_str = '''
        Average fire_risk for the whole area (on a score from 1 to 4): 1.07
        Average heat_risk for the whole area (on a score from 1 to 4): 1.58
        Average flood_risk for the whole area (on a score from 1 to 4): 1.80
        Average tree_count for the whole area: 115.94
    '''

    # --- Pre-generate JSON schemas from Pydantic models ---
    # This is efficient as schemas don't change per request
    risk_actions_schema = RiskActions.model_json_schema()
    risk_actions_and_explanation_schema = RiskActions_and_explanation.model_json_schema()
    # Select the appropriate JSON schema
    selected_response_schema = risk_actions_schema if not explain else risk_actions_and_explanation_schema

    jsonl_lines = []

    for idx, row in filtered_df.iterrows():
        h3_index = row['h3_10']
        
        # Skip already processed rows
        #if h3_index in processed_h3_indices:
        #    continue
        
        logging.debug(f"Processing row {idx} with H3 index {h3_index}")
        heat_risk_idx = 4 - int(row['heat_risk']) if pd.notna(row['heat_risk']) else None
        flood_risk_idx = 4 - int(row['flood_risk']) if pd.notna(row['flood_risk']) else None
        fire_risk_idx = 4 - int(row['fire_risk_202502']) if pd.notna(row['fire_risk_202502']) else None
        pois_cols = ['places_of_interest', 'densely_populated_at_risk_people', 'medical_care']
        pois_list = [
            str(row[col]) for col in pois_cols
            if col in row and pd.notna(row[col])
        ]
        pois_combined = ', '.join(pois_list) if pois_list else 'nan'

        row_id, pdf_files, system_instruction, prompt = prepare_prompt_systemprompt_files_batch(
            row_id=idx,
            municipality_context=general_context_str,
            heat_risk=f"{int(row['heat_risk'])}/4 or {map_data['heat_risk'][heat_risk_idx]}" if heat_risk_idx is not None else 'nan',
            flood_risk=f"{int(row['flood_risk'])}/4 or {map_data['flood_risk'][flood_risk_idx]}" if flood_risk_idx is not None else 'nan',
            fire_risk=f"{int(row['fire_risk_202502'])}/4 or {map_data['fire_risk'][fire_risk_idx]}" if fire_risk_idx is not None else 'nan',
            lst_day=f"{row['lst_day_202502']}°C" if pd.notna(row['lst_day_202502']) else 'nan',
            lst_night=f"{row['lst_night_202502']}°C" if pd.notna(row['lst_night_202502']) else 'nan',
            sealed_surface_pct=row['sealed_surfaces'] if pd.notna(row['sealed_surfaces']) else 'nan',
            canopy_cover_pct='nan',
            elevation='nan',
            river_proximity='nan',
            flood_plain='nan',
            tree_count=row['tree_count_sum'] if pd.notna(row['tree_count_sum']) else 'nan',
            flammability='nan',
            tree_connectivity=row['tree_connectivity'] if pd.notna(row['tree_connectivity']) else 'nan',
            fire_history_info=row['fire_history'] if pd.notna(row['fire_history']) else 'nan',
            population_density='nan',
            vulnerable_groups="NA - to be implemented",
            pois=pois_combined,
            climate_driven_impassable_roads= f"In this area there is a road that might become impassable: {row['climate_driven_impassable_roads']}" if pd.notna(row['climate_driven_impassable_roads']) else 'nan',
            emergency_assemble_areas=f"In this area there is an emergency assemble area: {row['emergency_assemble_areas']}" if pd.notna(row['emergency_assemble_areas']) else 'nan',
            comments=f"Somebody from the community made a comment, take this comment in high regard when coming up with your solutions: {row['comments']}" if pd.notna(row['comments']) else 'nan',
            pdf_uri=PDF_URI_LST,
            explain=EXPLAIN,
            print_output=True # Set to True to print output for debugging
        )
        

        # Fill the 'request' dictionary for the current row
        request_dict = create_gemini_request(system_instruction=system_instruction, 
                                            prompt=prompt, 
                                            file_list=pdf_files,
                                            selected_response_schema=selected_response_schema)


        # Create the full line for the JSONL file, including the 'key'
        jsonl_line = json.dumps({
            "key": row_id,  # Use your row identifier here
            "request": request_dict
        })
        jsonl_lines.append(jsonl_line)

    # Write all lines to the JSONL file at once
    with open(LOCAL_TMP_JSONL, 'w') as f:
        for line in jsonl_lines:
            f.write(line + '\n')

def upload_jsonl_to_gcs():
    """
    Uploads the JSONL file to Google Cloud Storage with a timestamp.
    """
    success = upload_to_gcs_without_timestamp(
        bucket_name=BUCKET_NAME,
        local_file_path=LOCAL_TMP_JSONL,
        gcs_prefix=INPUT_PREFIX
    )
    if success:
        print(f"JSONL file uploaded successfully to {INPUT_URI}")
    else:
        print("Failed to upload JSONL file to GCS.")

def create_batch_job(
    client,
    model = "gemini-2.5-flash",
    input_uri: str = INPUT_URI,
    output_uri: str = OUTPUT_URI
) -> BatchJob:
    """
    Creates a batch prediction job using Google Gemini API.
    
    Args:
        client: Initialized Gemini client
        model: Model name to use for predictions
        input_uri: GCS source URI containing requests
        output_uri: GCS destination URI for results
    
    Returns:
        BatchPredictionJob: Job object containing name and state
    """

    # Create the batch job with configuration
    job = client.batches.create(
        model=model,
        src=input_uri,
        config=types.CreateBatchJobConfig(
            dest=output_uri
        )
    )
    return job

def fetch_job(client, job_name: str) -> JobState:
    """
    Retrieves the current state of a batch prediction job.
    
    Args:
        client: Initialized Gemini API client
        job_name: The unique identifier/name of the batch job
    
    Returns:
        JobState: Object containing job details and current state
        None: If there's an error fetching the job
    
    Raises:
        Exception: When API call fails or job cannot be found
    """
    try:
        job = client.batches.get(name=job_name)
        return job
    except Exception as e:
        print(f"Error fetching job state: {e}")
        return None

def wait_for_job_completion(client, job_name: str, interval: int = 30) -> JobState:
    """
    Waits for a batch job to complete and returns final state.
    
    Args:
        client: Initialized Gemini client
        job_name: Name or ID of the batch job
        interval: Sleep interval between checks in seconds
    """
    completed_states = {
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED, 
        JobState.JOB_STATE_CANCELLED,
        JobState.JOB_STATE_PAUSED
    }
    
    job = fetch_job(client, job_name)
    current_state = job.state 
    
    while current_state not in completed_states:
        time.sleep(interval)
        current_job = fetch_job(client, job_name)
        current_state = current_job.state
        print(f"Job state: {current_state}")
    
    return current_state, job

def get_output_uri_from_job(job: BatchJob) -> str:
    """
    Extracts the output URI from a BatchJob object.

    Args:
        job: The BatchJob object.

    Returns:
        The output URI as a string, or None if not found.
    """
    if job and job.dest and job.dest.gcs_uri:
        return job.dest.gcs_uri
    return None

def read_output_file(output_uri: str, credentials=None) -> dict:
    """
    Reads the output file from GCS and creates a mapping of queries to responses.
    
    Args:
        output_uri: GCS URI where the output files are stored
        credentials: Service account credentials
    
    Returns:
        list: List of dictionaries containing query-answer pairs
    """
    from google.cloud import storage
    import json
    
    credentials, project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )

    results = []
    client = storage.Client(credentials=credentials)

    bucket_name = output_uri.split("/")[2]
    prefix = "/".join(output_uri.split("/")[3:])
    
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    all_timestamped_folders = set()
    for blob in blobs:
        # Find blobs that match the expected folder pattern
        parts = blob.name[len(prefix):].split('/')
        if len(parts) > 1 and parts[0].startswith("prediction-model-") and parts[0].endswith("Z"):
            folder_name = parts[0]
            try:
                timestamp_str = folder_name.replace("prediction-model-", "").replace("Z", "")
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
                all_timestamped_folders.add((timestamp, f"{prefix}{folder_name}/"))
            except ValueError:
                print(f"Skipping folder with malformed timestamp: {folder_name}")
                continue

    all_timestamped_folders = list(all_timestamped_folders)

    if not all_timestamped_folders:
        print(f"No timestamped folders found under '{OUTPUT_PREFIX}' in bucket '{bucket_name}'")
        # You might want to raise an error or return an empty list here
        latest_jsonl_blob = None
    else:
        # Find the latest timestamped folder
        all_timestamped_folders.sort(key=lambda x: x[0], reverse=True)
        latest_folder_timestamp, latest_folder_full_prefix = all_timestamped_folders[0]

        print(f"Processing data from the latest folder: {latest_folder_full_prefix}")

        # List blobs within that specific latest folder
        blobs_in_latest_folder = bucket.list_blobs(prefix=latest_folder_full_prefix)

        # Filter for the last .jsonl file in the latest folder based on creation time
        latest_jsonl_blob = None
        for blob in blobs_in_latest_folder:
            if blob.name.endswith('.jsonl'):
                if latest_jsonl_blob is None or blob.time_created > latest_jsonl_blob.time_created:
                    latest_jsonl_blob = blob

        if latest_jsonl_blob is None:
            print(f"No .jsonl files found in the latest folder: {latest_folder_full_prefix}")
            # You might want to raise an error or set a flag here
        else:
            print(f"Processing the latest .jsonl file: {latest_jsonl_blob.name}")

            content = blob.download_as_string()
            print(f"content = {content[:100]}...")
            for line in content.decode('utf-8').splitlines():
                response = json.loads(line)
                response_data = response.get('response')

                # Check if 'response' and 'candidates' keys exist and are not empty
                if response_data and response_data.get('candidates'):
                    # Safely access the answer
                    answer = response_data['candidates'][0]['content']['parts'][0]['text']
                    # Now you can proceed with the successful response
                    print(f"Successfully extracted answer for key: {response.get('key')}")
                else:
                    # Handle the error case
                    status = response.get('status', 'No status available')
                    print(f"Could not find 'candidates'. The response may be an error. Status: '{status}' for key: {response.get('key')}")
                    answer = None
                #query = response['request']['contents'][0]['parts'][0]['text']
                key = response.get('key')
                #answer = response['response']['candidates'][0]['content']['parts'][0]['text']
                results.append({"key": key, "answer": answer})
    
    return results

def clean_and_parse_json_string(answer_string):
    """
    Attempts to clean a string by removing markdown code block wrappers (```json\n...\n```)
    and then parses it as JSON.

    Args:
        answer_string (str): The string extracted from the model's response.

    Returns:
        dict or None: The parsed JSON dictionary if successful, None otherwise.
    """
    cleaned_string = answer_string.strip()

    # Regex to find content inside ```json ... ``` or ``` ... ```
    # re.DOTALL makes . match newlines as well
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned_string, re.DOTALL)
    
    if match:
        # If a markdown block is found, use its content
        json_content = match.group(1).strip()
    else:
        # Otherwise, assume the entire string should be JSON
        json_content = cleaned_string

    try:
        parsed_json = json.loads(json_content)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print(f"Problematic content after cleaning (first 500 chars):\n{json_content[:500]}...")
        return None

def add_to_geojson(results, output_file=OUT_VECTOR):
    """    Processes the results from the batch job and adds them to a GeoDataFrame.    """
    #Read FELT derived database for risk and other inputs
    filtered_df = gpd.read_file(DATABASE_PTH, 
                                dtype={
            'heat_risk': 'Float64',
            'flood_risk': 'Int64',
            'fire_risk_202502': 'Int64'
        })
    filtered_df['geometry'] = filtered_df['geometry'].apply(wkt.loads)  # Convert WKT strings to geometries
    filtered_df = gpd.GeoDataFrame(filtered_df, crs='EPSG:4326') 
    for col in ['heat_risk', 'flood_risk', 'fire_risk_202502']:
        filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').astype('Int64')
    filtered_df = filtered_df.iloc[:100]  # For testing purposes, limit to first 100 rows

    output_rows = []
    for result in results:
        try:
            idx = int(result['key'])
            if idx < 0 or idx >= len(filtered_df):
                raise IndexError(f"Index {idx} is out of bounds for DataFrame with length {len(filtered_df)}")
            logging.debug(f"Processing output for row {idx}")
            row = filtered_df.iloc[idx]

            answer = result['answer']
            if isinstance(answer, str):
                output_dict = clean_and_parse_json_string(answer) or {}
            else:
                output_dict = answer

            row_dict = {
                "felt:h3_index": row["felt:h3_index"],
                "h3_10": row["h3_10"],
                "geometry": row["geometry"],
            }
            for risk_type in ["fire", "flood", "heat"]:
                solutions = output_dict.get(risk_type, [])
                for i in range(3):
                    key = f"{risk_type}_solution_{i+1}"
                    row_dict[key] = solutions[i] if i < len(solutions) else None
            if EXPLAIN:
                for explanation_type in ['explanation_fire', 'explanation_flood', 'explanation_heat']:
                    explanations = output_dict.get(explanation_type, [])
                    for i in range(3):
                        key = f"{explanation_type}_{i+1}"
                        row_dict[key] = explanations[i] if i < len(explanations) else None

            # Update state
            output_rows.append(row_dict)



            #if EXIT and len(processed_h3_indices) >= EXIT:
            #    logging.debug(f"Exit condition met after processing {len(processed_h3_indices)} rows.")
            #    break
            
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                print(f"RESOURCE_EXHAUSTED error at row {idx}, sleeping 60s and retrying...", file=sys.stderr)
                time.sleep(60)
                continue  # This will re-enter the for loop with the same row (since nothing was added to processed_h3_indices)
            else:
                print(f"Error processing row {idx}: {e}", file=sys.stderr)
                continue
            


    # Flatten output_rows (list of lists) to a single list of dicts
    all_rows = output_rows

    if all_rows:
        output_gdf = gpd.GeoDataFrame(all_rows, geometry="geometry", crs='EPSG:4326')
        output_gdf.to_file(OUT_VECTOR, driver="GeoJSON")
        print(f"GeoJSON file {OUT_VECTOR} created.")
    else:
        print("No output rows to save.")


def main():
    """
    Main function to run the LLM pipeline with batching.
    """
    start_time = time.time()
    # Create the JSONL file
    create_jsonl_file(explain=EXPLAIN)

    # Upload the JSONL file to GCS
    upload_jsonl_to_gcs()

    # Set up the Gemini client
    client = setup_gemini_client()

    # Create a batch prediction job
    batch_job = create_batch_job(
        client=client,
        model=MODEL_ID,
        input_uri=INPUT_URI,
        output_uri=OUTPUT_URI
    )

    # Wait for job completion
    final_state, job = wait_for_job_completion(client, batch_job.name)

    # Get output URI and process results
    output_uri = get_output_uri_from_job(job)
    print(f"Output URI: {output_uri}")
    results = read_output_file(output_uri)
    print(f"job last update time: {job.update_time}")
    print(f"Job completed with state: {final_state}")
    if final_state == JobState.JOB_STATE_FAILED:
        print(f"Job failed. Error details: {job.error}")

    # Add results to GeoDataFrame and save to GeoJSON
    add_to_geojson(results, output_file=OUT_VECTOR)

    elapsed = time.time() - start_time
    print(f"Total processing time: {elapsed:.2f} seconds")



if __name__ == "__main__":
    # Run the main function
    main()