# price indication : 
# IF FLASH LITE VERSION :
# 0.89$ per 1000 rows
# it is 2.31$ for these 2600 rows

# FlASH VERSION :
# 3.88$ for 1000 rows OR 2.949$ with batching
# it is 10.09$ for 2600 rows OR 7.66$ with batching


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

from prompt_func import generate_risk_actions, map_data, cache_files
from tqdm import tqdm
import logging

# Set up logging to file only
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("run_llm_pipeline.log", mode='a')
    ]
)



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


def main():
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
    
    filtered_df = filtered_df.iloc[:2]  # For testing purposes, limit to first 100 rows
    print(f"Number of rows in filtered DataFrame: {len(filtered_df)}")

    if not filtered_df['h3_10'].is_unique:
        logging.warning("h3_10 column contains duplicates. This might lead to unexpected behavior when mapping results.")
        # Consider how to handle duplicates: first, last, or error.
        # For this example, we'll assume the first occurrence is used.
        filtered_df = filtered_df.drop_duplicates(subset=['h3_10'])

    h3_lookup = filtered_df.set_index('h3_10')

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
    cached_files = cache_files(pdf_uri=PDF_URI_LST, explain=EXPLAIN, time="20000s")

    output_rows_all = []
    output_rows = []

    tasks = []
    for idx, row in filtered_df.iterrows():
        h3_index = row['h3_10']
        
        # Skip already processed rows
        if h3_index in processed_h3_indices:
            continue
        
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
                print_output=True # Set to True to print output for debugging
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

    async def run_all_tasks(tasks):
        return await asyncio.gather(*tasks)

    results = asyncio.run(run_all_tasks(tasks))
    print("All LLM tasks completed.")

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
                "h3_10": row["h3_10"],
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

            if EXIT and len(processed_h3_indices) >= EXIT:
                logging.debug(f"Exit condition met after processing {len(processed_h3_indices)} rows.")
                break
            
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                print(f"RESOURCE_EXHAUSTED error at row {idx}, sleeping 60s and retrying...", file=sys.stderr)
                time.sleep(60)
                continue  # This will re-enter the for loop with the same row (since nothing was added to processed_h3_indices)
            #else:
                #print(f"Error processing row {idx}: {e}", file=sys.stderr)
                #continue
            


    # Flatten output_rows_all (list of lists) to a single list of dicts
    all_rows = [row for batch in output_rows_all for row in batch] + output_rows

    if all_rows:
        output_gdf = gpd.GeoDataFrame(all_rows, geometry="geometry", crs='EPSG:4326')
        output_gdf.to_file(OUT_VECTOR, driver="GeoJSON")
        print(f"GeoJSON file {OUT_VECTOR} created.")
    else:
        print("No output rows to save.")

    elapsed = time.time() - start_time
    print(f"Total processing time: {elapsed:.2f} seconds")

if __name__ == "__main__":

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

    main()


