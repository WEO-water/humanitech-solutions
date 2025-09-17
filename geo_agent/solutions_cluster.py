import pandas as pd
import geopandas as gpd
import numpy as np
import vertexai
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import DBSCAN
import asyncio
import json
import sys
import os
from google import genai

from prompt_func import generate_risk_actions, cache_files, map_data


# --- CONFIGURATION & CONSTANTS ---
PROJECT_ID = "dl-test-439308"
LOCATION = "europe-west1"
DATABASE_PATH = "gs://dl-test-439308-bucket/weo-data/dashboard/df_export_20250912_090128.csv"
VALIDATED_DATA_PATH = 'your_data_validated.csv'
FINAL_SUMMARY_PATH = 'final_summary_gdf.csv'
FINAL_COMPLETE_PATH_CSV = "final_complete_gdf.csv"
FINAL_COMPLETE_PATH_GEOJSON = "final_data.geojson"

RUN_SOLUTION_GENERATION_LLM = True 
EXPLAIN = True # Controls if explanations are requested from the LLM

GENERAL_CONTEXT_STR = '''
    Average fire_risk for the whole area (on a score from 1 to 4): 1.07
    Average heat_risk for the whole area (on a score from 1 to 4): 1.58
    Average flood_risk for the whole area (on a score from 1 to 4): 1.80
    Average tree_count for the whole area: 115.94
    '''
PDF_URI_LST = [
        'gs://dl-test-439308-bucket/weo-data/2021 Dargo, Census All persons QuickStats _ Australian Bureau of Statistics.pdf',
        'gs://dl-test-439308-bucket/weo-data/climate-ready-communities-a-guide-to-getting-started.pdf',
        'gs://dl-test-439308-bucket/weo-data/Dargo Rear v8 - final.pdf'
    ]

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV, cleans it by aggregating rows with the same h3_10 index."""
    print("--- 1. Loading and Cleaning Data ---")
    df = pd.read_csv(file_path)
    agg_dict = {'comments': lambda x: ' '.join(x.dropna().astype(str))}
    for col in df.columns:
        if col not in ['comments', 'h3_10']:
            agg_dict[col] = 'first'
    df_cleaned = df.groupby('h3_10', as_index=False).agg(agg_dict)
    df_cleaned.set_index('h3_10', inplace=True)
    print(f"Data loaded and cleaned. Final row count: {len(df_cleaned)}")
    return df_cleaned

def update_risk_value(df_main: pd.DataFrame, h3_index: str, field_to_update: str, new_value: float) -> str:
    """Updates a specific risk field for a given H3 index in the provided DataFrame."""
    if h3_index not in df_main.index: return f"Error: H3 index '{h3_index}' not found."
    if field_to_update not in ['flood_risk', 'heat_risk', 'fire_risk_202502']: return f"Error: Invalid field '{field_to_update}'."
    original_comment = df_main.loc[h3_index, 'comments']
    if "[PROCESSED]" in str(original_comment): return f"No action taken. H3 index '{h3_index}' has already been processed."
    old_value = df_main.loc[h3_index, field_to_update]
    df_main.loc[h3_index, field_to_update] = new_value
    df_main.loc[h3_index, 'comments'] = f"[PROCESSED] Original Comment: {original_comment}"
    return f"Action successful. Updated '{field_to_update}' for H3 index '{h3_index}' from '{old_value}' to '{new_value}'."

def validate_risk_with_llm(df_main: pd.DataFrame) -> pd.DataFrame:
    """Uses a Gemini model with function calling to validate risk scores based on user comments."""
    print("\n--- 2. Validating Risk with LLM ---")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    def update_risk_wrapper(h3_index: str, field_to_update: str, new_value: float) -> str:
        return update_risk_value(df_main, h3_index, field_to_update, new_value)
    model = GenerativeModel("gemini-2.5-pro")
    tool_config = Tool.from_function_declarations([FunctionDeclaration.from_func(update_risk_wrapper)])
    available_tools = {"update_risk_wrapper": update_risk_wrapper}
    df_to_process = df_main[df_main['comments'].str.strip().fillna('') != ''].copy()
    print(f"Found {len(df_to_process)} rows with comments to process.")
    for h3_index, row in df_to_process.iterrows():
        comment = row.get('comments', '')
        if "[PROCESSED]" in comment: continue
        print(f"\n--- Processing H3 Index: {h3_index} ---")
        prompt = f"""You are a risk data validation analyst. Analyze the user's comment and data to determine if a risk score needs correction. Risk scores are 0-5. TASK: 1. Analyze comment nature: Direct Hazard Observation, Contextual Condition, or Explicit Correction. 2. Take action: Update for Direct Observations if score is too low. DO NOT change for Contextual Conditions. Always update for Explicit Corrections. Think step-by-step. DATA: - h3_index: {h3_index} - comment: "{comment}" - flood_risk: {row.get('flood_risk', 'N/A')} - fire_risk: {row.get('fire_risk_202502', 'N/A')} - heat_risk: {row.get('heat_risk', 'N/A')}"""
        try:
            response = model.generate_content(prompt, tools=[tool_config])
            part = response.candidates[0].content.parts[0]
            if hasattr(part, 'text') and part.text: print(f"LLM Reasoning: {part.text}")
            if hasattr(part, 'function_call') and part.function_call:
                function_call = part.function_call
                result = available_tools[function_call.name](**dict(function_call.args))
                print(f"Result: {result}")
        except Exception as e: print(f"An error occurred while processing {h3_index}: {e}")
    print("\nLLM validation complete.")
    df_main.to_csv(VALIDATED_DATA_PATH, index=True)
    return df_main

def engineer_features_for_clustering(df: pd.DataFrame) -> (gpd.GeoDataFrame, np.ndarray):
    """Converts DataFrame to GeoDataFrame, scales coordinates and features for clustering."""
    print("\n--- 3. Engineering Features for Clustering ---")
    df.reset_index(inplace=True)
    df.dropna(subset=['geometry'], inplace=True)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['geometry']), crs='EPSG:4326')
    gdf['x'] = gdf.geometry.x
    gdf['y'] = gdf.geometry.y
    coord_scaler = MinMaxScaler()
    gdf[['x_scaled', 'y_scaled']] = coord_scaler.fit_transform(gdf[['x', 'y']])
    feature_cols = ['flood_risk', 'fire_risk_202502', 'heat_risk', 'tree_count_sum', 'tree_connectivity']
    gdf.dropna(subset=feature_cols, inplace=True)
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(gdf[feature_cols])
    spatial_weight = 2.0
    weighted_coords = gdf[['x_scaled', 'y_scaled']].values * spatial_weight
    combined_data = np.hstack([weighted_coords, features_scaled])
    print(f"Feature engineering complete. Shape of data for clustering: {combined_data.shape}")
    return gdf, combined_data

def perform_dbscan_clustering(gdf: gpd.GeoDataFrame, data_for_clustering: np.ndarray) -> gpd.GeoDataFrame:
    """Runs DBSCAN clustering and adds cluster labels to the GeoDataFrame."""
    print("\n--- 4. Performing DBSCAN Clustering ---")
    dbscan = DBSCAN(eps=1.5, min_samples=10)
    clusters = dbscan.fit_predict(data_for_clustering)
    gdf['cluster'] = clusters
    print("Clustering complete. Cluster counts:\n", gdf['cluster'].value_counts())
    return gdf

def summarize_clusters_and_outliers(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Separates outliers, summarizes clean clusters by centroid, and recombines them.
    This version ensures 'h3_10' is a column throughout the process.
    """
    print("\n--- 5. Summarizing Clusters and Outliers ---")
    
    special_outlier_cols = [
        'climate_driven_impassable_roads',
        'emergency_assemble_areas',
        'comments',
        'places_of_interest',
        'densely_populated_at_risk_people',
        'medical_care'
    ]

    # --- START: FIX ---
    # Create a temporary DataFrame for masking to avoid modifying the original gdf yet.
    gdf_for_masking = gdf[special_outlier_cols].copy()
    
    # Identify string columns to clean
    string_cols_to_clean = gdf_for_masking.select_dtypes(include=['object', 'string']).columns
    
    # Replace empty or whitespace-only strings with np.nan
    for col in string_cols_to_clean:
        gdf_for_masking[col] = gdf_for_masking[col].replace(r'^\s*$', np.nan, regex=True)

    # Now create the mask from the cleaned data
    special_outlier_mask = gdf_for_masking.notna().any(axis=1)
    # --- END: FIX ---

    print(f"len gdf = {len(gdf)}")
    print(f"len special_outlier_mask = {len(special_outlier_mask)}")

    # PART 1: Isolate all outliers (original DBSCAN outliers OR special cases)
    outliers_gdf = gdf[(gdf['cluster'] == -1) | (special_outlier_mask)].copy()
    print(f"len outliers_gdf = {len(outliers_gdf)}")
    # Standardize all outliers to have a cluster ID of -1
    outliers_gdf['cluster'] = -1
    
    # PART 2: Isolate the "clean" data to be clustered
    clusters_gdf = gdf[(gdf['cluster'] != -1) & (~special_outlier_mask)].copy()
    print(f"len clusters_gdf = {len(clusters_gdf)}")
    
    
    # 2. SUMMARIZE THE CLEAN CLUSTERS
    # ---------------------------------
    if not clusters_gdf.empty: # Add a check to prevent errors if there are no clusters
        numerical_cols = clusters_gdf.select_dtypes(include=np.number).columns.tolist()
        agg_dict = {col: 'mean' for col in numerical_cols}
        if 'cluster' in agg_dict:
            del agg_dict['cluster']
        
        if 'h3_10' in clusters_gdf.columns:
            agg_dict['h3_10'] = 'first'

        def get_centroid(points):
            return gpd.GeoSeries(points).union_all().centroid
        
        agg_dict['geometry'] = get_centroid
        
        cluster_summary_gdf = clusters_gdf.groupby('cluster').agg(agg_dict).reset_index()
        print(f"len cluster_summary_gdf = {len(cluster_summary_gdf)}")
    else:
        print("len cluster_summary_gdf = 0 (No clean clusters to summarize)")
        cluster_summary_gdf = gpd.GeoDataFrame() # Create empty GeoDataFrame
    
    # 3. COMBINE THE SUMMARIZED CLUSTERS AND THE INDIVIDUAL OUTLIERS
    # ----------------------------------------------------------------
    outliers_gdf.reset_index(drop=True, inplace=True)

    final_summary_gdf = pd.concat([cluster_summary_gdf, outliers_gdf], ignore_index=True)
    final_summary_gdf = gpd.GeoDataFrame(final_summary_gdf, geometry='geometry')

    final_summary_gdf.loc[final_summary_gdf['cluster'] != -1, 'h3_10'] = None

    print("Final summary GeoDataFrame created successfully.")
    print(f"Total rows: {len(final_summary_gdf)}")
    print("\nFinal value counts for the 'cluster' column:")
    print(final_summary_gdf['cluster'].value_counts().sort_index())
    final_summary_gdf.to_csv(FINAL_SUMMARY_PATH, index=False)
    print(f"Cluster summarization complete. Saved to '{FINAL_SUMMARY_PATH}'")
    print(f"Final summary GeoDataFrame shape: {final_summary_gdf.shape}")
    return final_summary_gdf


async def generate_solution_actions_with_llm(summary_gdf: gpd.GeoDataFrame) -> dict:
    """
    Generates solution actions for each cluster/outlier using the imported async function.
    """
    print("\n--- 6. Generating Solution Actions with LLM ---")
    if not RUN_SOLUTION_GENERATION_LLM:
        print("Skipping LLM solution generation as per configuration.")
        return {}
    
    # Initialize a file cache for the LLM if a PDF URI is provided
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    cached_files = cache_files(client, PDF_URI_LST, explain=EXPLAIN) if PDF_URI_LST else None
    
    tasks = []
    for _, row in summary_gdf.iterrows():
        row_id = f"cluster_{int(row['cluster'])}" if row['cluster'] != -1 else row['h3_10']
        
        # Prepare inputs for the LLM, handling potential NaN values
        try:
            heat_risk_val = row.get('heat_risk')
            heat_risk_idx = int(4 - heat_risk_val) if pd.notna(heat_risk_val) else None
            heat_risk_str = f"{int(heat_risk_val)}/4 or {map_data['heat_risk'][heat_risk_idx]}" if heat_risk_idx is not None else 'nan'
            
            flood_risk_val = row.get('flood_risk')
            flood_risk_idx = int(4 - flood_risk_val) if pd.notna(flood_risk_val) else None
            flood_risk_str = f"{int(flood_risk_val)}/4 or {map_data['flood_risk'][flood_risk_idx]}" if flood_risk_idx is not None else 'nan'

            fire_risk_val = row.get('fire_risk_202502')
            fire_risk_idx = int(4 - fire_risk_val) if pd.notna(fire_risk_val) else None
            fire_risk_str = f"{int(fire_risk_val)}/4 or {map_data['fire_risk'][fire_risk_idx]}" if fire_risk_idx is not None else 'nan'
            
            pois_cols = ['places_of_interest', 'densely_populated_at_risk_people', 'medical_care']
            pois_list = [str(row[col]) for col in pois_cols if col in row and pd.notna(row[col])]
            pois_combined = ', '.join(pois_list) if pois_list else 'nan'

            task = generate_risk_actions(
                client=client,
                row_id=row_id,
                municipality_context=GENERAL_CONTEXT_STR,
                heat_risk=heat_risk_str,
                flood_risk=flood_risk_str,
                fire_risk=fire_risk_str,
                lst_day=f"{row.get('lst_day_202502', 'nan')}°C",
                lst_night=f"{row.get('lst_night_202502', 'nan')}°C",
                sealed_surface_pct=row.get('sealed_surfaces', 'nan'),
                canopy_cover_pct='nan', elevation='nan', river_proximity='nan', flood_plain='nan',
                tree_count=row.get('tree_count_sum', 'nan'),
                flammability='nan',
                tree_connectivity=row.get('tree_connectivity', 'nan'),
                fire_history_info=row.get('fire_history', 'nan'),
                population_density='nan', vulnerable_groups="N/A",
                pois=pois_combined,
                climate_driven_impassable_roads=f"Road impassability risk: {row.get('climate_driven_impassable_roads', 'nan')}",
                emergency_assemble_areas=f"Emergency assembly area: {row.get('emergency_assemble_areas', 'nan')}",
                comments=f"Community comment: {row.get('comments', 'nan')}",
                pdf_uri=PDF_URI_LST,
                cache=cached_files,
                explain=EXPLAIN,
                print_output=False
            )
            tasks.append(task)
        except Exception as e:
            print(f"Error preparing task for row {row_id}: {e}", file=sys.stderr)

    print(f"Starting {len(tasks)} asynchronous LLM generation tasks...")
    results = await asyncio.gather(*tasks)
    
    results_dict = {}
    for row_id, output in results:
        try:
            results_dict[row_id] = json.loads(output) if isinstance(output, str) else output
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON for {row_id}. Raw output: {output}")
            results_dict[row_id] = {} # Assign empty dict on failure
        
    print(f"Processed {len(results_dict)} LLM responses.")
    return results_dict

def map_solutions_to_dataframe(original_gdf: gpd.GeoDataFrame, llm_results: dict) -> gpd.GeoDataFrame:
    """Maps the generated LLM solutions back to the full, non-summarized GeoDataFrame."""
    print("\n--- 7. Mapping LLM Solutions to Final DataFrame ---")
    if not llm_results:
        print("No LLM results to map. Returning original clustered DataFrame.")
        return original_gdf
    
    final_gdf = original_gdf.copy()
    
    solution_cols = []
    for risk in ["fire", "flood", "heat"]:
        for i in range(1, 4):
            solution_cols.append(f"{risk}_solution_{i}")
            if EXPLAIN:
                solution_cols.append(f"explanation_{risk}_{i}")
    for col in solution_cols:
        final_gdf[col] = None

    for key, data in llm_results.items():
        if not data: continue # Skip if data is empty from a failed parse
        target_mask = (final_gdf['cluster'] == int(key.split('_')[1])) if key.startswith('cluster_') else (final_gdf.index == key)
            
        for risk in ["fire", "flood", "heat"]:
            solutions = data.get(risk, [])
            for i in range(3):
                col_name = f"{risk}_solution_{i+1}"
                final_gdf.loc[target_mask, col_name] = solutions[i] if i < len(solutions) else None
                if EXPLAIN:
                    exp_col = f"explanation_{risk}_{i+1}"
                    explanations = data.get(f"explanation_{risk}", [])
                    final_gdf.loc[target_mask, exp_col] = explanations[i] if i < len(explanations) else None

    print("LLM responses have been mapped back to the full GeoDataFrame.")
    return final_gdf

# --- MAIN EXECUTION ---
async def main_async():
    """Main async function to orchestrate the entire data processing pipeline."""
    df_cleaned = load_and_clean_data(DATABASE_PATH)
    df_validated = validate_risk_with_llm(df_cleaned)
    gdf, data_for_clustering = engineer_features_for_clustering(df_validated)
    gdf_clustered = perform_dbscan_clustering(gdf, data_for_clustering)
    gdf_summary = summarize_clusters_and_outliers(gdf_clustered)
    
    solution_results = await generate_solution_actions_with_llm(gdf_summary)
    
    final_gdf = map_solutions_to_dataframe(gdf_clustered, solution_results)

    print(f"\n--- 8. Saving Final Outputs ---")
    final_gdf.to_csv(FINAL_COMPLETE_PATH_CSV)
    print(f"Final data saved to '{FINAL_COMPLETE_PATH_CSV}'")
    
    final_gdf.columns = final_gdf.columns.astype(str)
    final_gdf.to_file(FINAL_COMPLETE_PATH_GEOJSON, driver="GeoJSON")
    print(f"Final GeoDataFrame saved to '{FINAL_COMPLETE_PATH_GEOJSON}'")

if __name__ == "__main__":
    import time
    start_time = time.time() 
    asyncio.run(main_async())

    end_time = time.time()

    duration = end_time - start_time
    
    print(f"\n--- SCRIPT FINISHED ---")
    print(f"Total execution time: {duration:.2f} seconds")
    print(f"Which is equal to: {duration/60:.2f} minutes")