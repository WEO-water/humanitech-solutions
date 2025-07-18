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
import h3
import asyncio
from tenacity import retry, wait_random_exponential, stop_after_attempt

from .prompt_func import generate_risk_actions, map_data, cache_files
from tqdm import tqdm
import logging

import get_data

# Set up logging to file only
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("new_comments_run_llm_pipeline_async.log", mode='a')
    ]
)

FELT_DATA_DIR = 'gs://dl-test-439308-bucket/weo-data/dashboard'#'/mnt/fvw/data/tmp/humanitech/dashboard'
COMMENTS_TS = '20250708_164650'
DB_PTH_DCT = {
    
    'metrics': 'Heat-Risk-.zip',
    'medical_care': 'medical_care.geojson',
    'climate_driven_impassable_roads': 'climate_driven_impassable_roads.geojson',
    'densely_populated_at_risk_people': 'densely_populated_at_risk_people.geojson',
    'emergency_assemble_areas': 'emergency_assemble_areas.geojson',
    'places_of_interest': 'places_of_interest.geojson',
    'comments': f'comments_{COMMENTS_TS}.zip',

}

INDEX = column_to_merge_on = 'h3_10'

NEW_COMMENTS = True
GET_DATA = False
COMMENTS_PTH = "gs://dl-test-439308-bucket/weo-data/dashboard/comments_20250708_160451.zip"

    #Inputs
DATABASE_PTH = 'gs://dl-test-439308-bucket/weo-data/humanitech_dargo_filtered_database_250620.csv'
PDF_URI_LST = [
        'gs://dl-test-439308-bucket/weo-data/2021 Dargo, Census All persons QuickStats _ Australian Bureau of Statistics.pdf',
        'gs://dl-test-439308-bucket/weo-data/climate-ready-communities-a-guide-to-getting-started.pdf',
        'gs://dl-test-439308-bucket/weo-data/Dargo Rear v8 - final.pdf'
    ]

    #Outputs
CHECKPOINT_FILE = "rh_async_humanitech_dargo_emergency_solutions_v300_checkpoint.geojson"
OUT_VECTOR = "rh_async_humanitech_dargo_emergency_solutions_v300.geojson"

    # Configuration
BATCH_SIZE = 10  # Save after every 10 rows
EXIT = 10 # Exit after processing this many rows
EXPLAIN = True  


def save_checkpoint(rows, checkpoint_file, geometry_col="geometry", crs="EPSG:4326", subset_col="h3_10"):
    """Save a checkpoint GeoDataFrame, merging with existing if present."""
    tmp_gdf = gpd.GeoDataFrame(rows, geometry=geometry_col, crs=crs)
    if os.path.exists(checkpoint_file):
        existing_gdf = gpd.read_file(checkpoint_file)
        combined_gdf = pd.concat([existing_gdf, tmp_gdf], ignore_index=True)
        combined_gdf = combined_gdf.drop_duplicates(subset=subset_col)
        combined_gdf.to_file(checkpoint_file, driver="GeoJSON")
    else:
        tmp_gdf.drop_duplicates(subset=subset_col).to_file(checkpoint_file, driver="GeoJSON")

def get_comments_h3(path):

    comments = gpd.read_file(path)

    # Handle different geometry types for H3 assignment
    if comments.geometry.iloc[0].geom_type == "Point":
        # For Point geometries, assign H3 index directly
        comments = comments.h3.geo_to_h3(resolution=10, set_index=False)
    elif comments.geometry.iloc[0].geom_type == "MultiPoint":
        # For MultiPoint geometries, calculate centroid and assign H3 index
        comments = comments.explode(ignore_index=True)
        comments = comments.h3.geo_to_h3(resolution=10, set_index=False)
    elif comments.geometry.iloc[0].geom_type == "Polygon":
        comments = comments.h3.polyfill(10+4, explode=True).set_index('h3_polyfill').h3.h3_to_parent_aggregate(10, operation = {'emergency_assemble_areas': 'first',})  # Take the first value in each group# Add other columns as needed, e.g., 'count': 'sum'
        comments = comments.reset_index()
    else: 
        print(f"Unsupported geometry type {comments.geometry.iloc[0].geom_type} in {path}. Skipping H3 assignment.", file=sys.stderr)
    
    comments['h3_10_int'] = comments['h3_10'].apply(lambda x: int(x, 16) if pd.notna(x) else None)

    #pd.set_option('display.max_columns', None)
    #print("comments columns = ")
    #print(comments.columns)
    print("comments head")
    print(comments.head())
    list = comments['h3_10'].tolist()

    return list

def main(NEW_COMMENTS, GET_DATA, COMMENTS_PTH, DATABASE_PTH, PDF_URI_LST, CHECKPOINT_FILE, OUT_VECTOR, BATCH_SIZE, EXIT, EXPLAIN):
    start_time = time.time()

    # Initialize checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint_gdf = gpd.read_file(CHECKPOINT_FILE)
        processed_h3_indices = set(checkpoint_gdf["h3_10"])
    else:
        processed_h3_indices = set()
        # output_gdf = gpd.GeoDataFrame()
        # output_gdf.to_file(CHECKPOINT_FILE, driver="GeoJSON")

    logging.debug(f"Already processed {len(processed_h3_indices)} H3 indices stores in {CHECKPOINT_FILE}")

    if GET_DATA:
        database_pth = get_data(data_dir=FELT_DATA_DIR, db_pth_dct=DB_PTH_DCT, index=INDEX)
    else:
        database_pth = DATABASE_PTH
    #Read FELT derived database for risk and other inputs
    filtered_df = gpd.read_file(database_pth, 
                                dtype={
            'heat_risk': 'Float64',
            'flood_risk': 'Int64',
            'fire_risk_202502': 'Int64'
        })
    
    #print("filtered df head = ")
    #print(filtered_df.head())
    filtered_df['geometry'] = filtered_df['geometry'].apply(wkt.loads)  # Convert WKT strings to geometries
    filtered_df = gpd.GeoDataFrame(filtered_df, crs='EPSG:4326') 
    for col in ['heat_risk', 'flood_risk', 'fire_risk_202502']:
        filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').astype('Int64')
    
    filtered_df = filtered_df.iloc[:10]  # For testing purposes, limit to first 100 rows
    print(f"Number of rows in filtered DataFrame: {len(filtered_df)}")

    if not filtered_df['h3_10'].is_unique:
        logging.warning("h3_10 column contains duplicates. This might lead to unexpected behavior when mapping results.")
        # Consider how to handle duplicates: first, last, or error.
        # For this example, we'll assume the first occurrence is used.
        filtered_df = filtered_df.drop_duplicates(subset=['h3_10'])


    # Create the general context string
    general_context_str = '''
    Average fire_risk for the whole area (on a score from 1 to 4): 1.07
    Average heat_risk for the whole area (on a score from 1 to 4): 1.58
    Average flood_risk for the whole area (on a score from 1 to 4): 1.80
    Average tree_count for the whole area: 115.94
    '''

    # Process rows with checkpointing
    total_rows = len(filtered_df)
    unprocessed_count = total_rows - len(processed_h3_indices)
    pbar = tqdm(total=unprocessed_count, desc="Processing rows")

    # cache files gemini
    cached_files = cache_files(pdf_uri=PDF_URI_LST, explain=EXPLAIN)

    output_rows_all = []
    output_rows = []

    tasks = []

    if NEW_COMMENTS: 
        logging.info("NEW_COMMENTS flag is True. Processing only H3 indices with new comments.")
        new_comments_h3_indices = get_comments_h3(FELT_DATA_DIR + "/" + DB_PTH_DCT.get('comments'))
        print(f"Number of rows in new comments list: {len(new_comments_h3_indices)}")
        print(new_comments_h3_indices)
        filtered_df['h3_10'] = filtered_df['h3_10'].astype(str)
        #print(filtered_df.head())
        # Filter the main DataFrame to get only the rows corresponding to new comments
        # Use .loc for index lookup
        rows_to_process_df = filtered_df[filtered_df['h3_10'].isin(new_comments_h3_indices)]
        #print(rows_to_process_df.head())
        print(f"Number of rows in rows to process: {len(rows_to_process_df)}")

        processed_h3_indices.difference_update(new_comments_h3_indices)
        logging.debug(f"Removed {len(new_comments_h3_indices)} H3 indices with new comments from processed set for re-processing.")

        if rows_to_process_df.empty:
            print("No matching H3 indices found in the main database for the new comments. Exiting.")
            return # Exit if no relevant rows to process
    else:
        logging.info("NEW_COMMENTS flag is False. Processing all unprocessed H3 indices.")
        # Filter out already processed rows if NEW_COMMENTS is False
        rows_to_process_df = filtered_df[~filtered_df['h3_10'].isin(processed_h3_indices)]

    h3_lookup = rows_to_process_df.set_index('h3_10')
    for idx, row in rows_to_process_df.iterrows():
        h3_index = row['h3_10']
        
        try:
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
            tasks.append(generate_risk_actions(
                row_id=h3_index,
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
                cache=cached_files,
                explain=EXPLAIN,
                print_output=False # Set to True to print output for debugging
            ))

        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                print(f"RESOURCE_EXHAUSTED error at row {idx}, sleeping 60s and retrying...", file=sys.stderr)
                time.sleep(60)
                continue  # This will re-enter the for loop with the same row (since nothing was added to processed_h3_indices)
            else:
                print(f"Error processing row {idx}: {e}", file=sys.stderr)
                continue

    print(f"Starting {len(tasks)} asynchronous LLM generation tasks...")

    async def run_all_tasks(tasks_list):
        return await asyncio.gather(*tasks_list)

    results = asyncio.run(run_all_tasks(tasks))
    print("All LLM tasks completed.")

    #print(len(results))

    for h3_index, output in results: 
        try:
            logging.debug(f"Processing output for H3 index {h3_index}")
            row = h3_lookup.loc[h3_index]
            # Check if output is a string and parse it as JSON
            
            if isinstance(output, str):
                output_dict = json.loads(output)
            else:
                output_dict = output

            row_dict = {
                "felt:h3_index": row["felt:h3_index"],
                "h3_10": h3_index,
                "geometry": row["geometry"] if "geometry" in row else None,
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
            processed_h3_indices.add(h3_index) # processed_h3_indices.add(h3_index)
            pbar.update(1)

            # Save checkpoint periodically
            if len(output_rows) % BATCH_SIZE == 0:
                save_checkpoint(output_rows, CHECKPOINT_FILE)
                logging.debug(f"Checkpoint saved after processing {len(output_rows)} rows.")
                
                output_rows_all.append(output_rows)
                output_rows = []

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
            
    pbar.close()

    # Save any remaining rows in output_rows after the loop finishes
    if output_rows:
        save_checkpoint(output_rows, CHECKPOINT_FILE)
        logging.debug(f"Final checkpoint saved for remaining {len(output_rows)} rows.")
        output_rows_all.append(output_rows)        


    # Flatten output_rows_all (list of lists) to a single list of dicts
    all_rows = [row for batch in output_rows_all for row in batch] + output_rows

    if all_rows:
        output_gdf = gpd.GeoDataFrame(all_rows, geometry="geometry", crs='EPSG:4326')
        if os.path.exists(CHECKPOINT_FILE):
            final_output_gdf = gpd.read_file(CHECKPOINT_FILE)
            final_output_gdf.to_file(OUT_VECTOR, driver="GeoJSON")
            print(f"GeoJSON file {OUT_VECTOR} created from checkpoint.")
        else:
            # This case should ideally not happen if processing successfully started
            output_gdf.to_file(OUT_VECTOR, driver="GeoJSON")
            print(f"GeoJSON file {OUT_VECTOR} created (from current run's results).")
    else:
        print("No output rows to save in the final output file.")

    elapsed = time.time() - start_time
    print(f"Total processing time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    NEW_COMMENTS = False
    GET_DATA = False
    COMMENTS_PTH = "gs://dl-test-439308-bucket/weo-data/dashboard/comments_20250708_160451.zip"

    #Inputs
    DATABASE_PTH = 'gs://dl-test-439308-bucket/weo-data/humanitech_dargo_filtered_database_250620.csv'
    PDF_URI_LST = [
        'gs://dl-test-439308-bucket/weo-data/2021 Dargo, Census All persons QuickStats _ Australian Bureau of Statistics.pdf',
        'gs://dl-test-439308-bucket/weo-data/climate-ready-communities-a-guide-to-getting-started.pdf',
        'gs://dl-test-439308-bucket/weo-data/Dargo Rear v8 - final.pdf'
    ]

    #Outputs
    CHECKPOINT_FILE = "rh_async_humanitech_dargo_emergency_solutions_v300_checkpoint.geojson"
    OUT_VECTOR = "rh_async_humanitech_dargo_emergency_solutions_v300.geojson"

    # Configuration
    BATCH_SIZE = 10  # Save after every 10 rows
    EXIT = 10 # Exit after processing this many rows
    EXPLAIN = True  

    main(NEW_COMMENTS, GET_DATA, COMMENTS_PTH, DATABASE_PTH, PDF_URI_LST, CHECKPOINT_FILE, OUT_VECTOR, BATCH_SIZE, EXIT, EXPLAIN)


