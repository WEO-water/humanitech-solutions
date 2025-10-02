import pandas as pd
import geopandas as gpd
import numpy as np
import vertexai
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import shap
import asyncio
import json
import sys
import os
from google import genai
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from prompt_func import generate_risk_actions, cache_files, map_data

# --- CONFIGURATION & CONSTANTS ---
PROJECT_ID = "dl-test-439308"
LOCATION = "europe-west1"
DATABASE_PATH = "filtered_df.csv" #"gs://dl-test-439308-bucket/weo-data/dashboard/df_export_20250912_090128.csv"

# Define output directory and create it if it doesn't exist
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALIDATED_DATA_PATH = os.path.join(OUTPUT_DIR, 'your_data_validated.csv')
FINAL_SUMMARY_PATH = os.path.join(OUTPUT_DIR, 'final_summary_gdf.csv')
FINAL_COMPLETE_PATH_CSV = os.path.join(OUTPUT_DIR, "final_complete_gdf.csv")
FINAL_COMPLETE_PATH_GEOJSON = os.path.join(OUTPUT_DIR, "final_data.geojson")
FINAL_COMPLETE_PATH_SHAPE = os.path.join(OUTPUT_DIR, "final_data.shp")
CLUSTER_MAP_PATH = os.path.join(OUTPUT_DIR, 'cluster_map.png')

RUN_SOLUTION_GENERATION_LLM = False 
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
    #df = pd.read_csv(file_path)

    essential_cols = [
        'h3_10', 'geometry', 'comments', 'flood_risk', 'tree_count_sum',
        'fire_risk_202502', 'heat_risk', 'lst_day_202502', 'lst_night_202502',
        'tree_connectivity', 'sealed_surfaces', 'fire_history', 
        'climate_driven_impassable_roads', 'emergency_assemble_areas', 
        'places_of_interest', 'densely_populated_at_risk_people', 'medical_care'
    ]

    # Define the data type for columns that might have mixed types
    # 'comments' is often a good candidate to specify as a string
    dtype_spec = {
        'comments': str,
        'climate_driven_impassable_roads': str,
        'emergency_assemble_areas': str,
        'places_of_interest': str,
        'medical_care': str,
        'densely_populated_at_risk_people': str
    }

    df = pd.read_csv(file_path, usecols=essential_cols, dtype=dtype_spec)

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
    vertexai.init(project=PROJECT_ID, location=LOCATION) #TODO: wil be deprecated, see prompt filter comment for new version
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

    print("\n--- Initial NaN Count Per Column ---")
    nan_counts = df.isnull().sum()
    # Filter to only show columns that actually have missing values
    columns_with_nans = nan_counts[nan_counts > 0]

    nans_columns = os.path.join(OUTPUT_DIR, 'nan_columns.txt')
    with open(nans_columns, 'w') as f:
        f.write(columns_with_nans.to_string())
    print(f"NaN columns saved to '{nans_columns}'")

    df.reset_index(inplace=True)
    df.dropna(subset=['geometry'], inplace=True)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['geometry']), crs='EPSG:4326')
    gdf['x'] = gdf.geometry.x
    gdf['y'] = gdf.geometry.y
    coord_scaler = MinMaxScaler()
    gdf[['x_scaled', 'y_scaled']] = coord_scaler.fit_transform(gdf[['x', 'y']])
    feature_cols = ['flood_risk', 'fire_risk_202502', 'heat_risk', 'tree_count_sum', 'tree_connectivity', 'sealed_surfaces' ]
    print(f"Before feature cols dropna: {gdf.shape}")
    gdf['tree_count_sum'] = gdf['tree_count_sum'].fillna(0)
    gdf.dropna(subset=feature_cols, inplace=True)
    print(f"After feature cols dropna: {gdf.shape}")
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(gdf[feature_cols])
    spatial_weight = 0
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

def perform_hdbscan_clustering(gdf: gpd.GeoDataFrame, data_for_clustering: np.ndarray) -> gpd.GeoDataFrame:
    """Runs HDBSCAN clustering and adds cluster labels to the GeoDataFrame."""
    print("\n--- 4. Performing HDBSCAN Clustering ---")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=1000, min_samples=150)
    clusters = clusterer.fit_predict(data_for_clustering)
    gdf['cluster'] = clusters
    print("Clustering complete. Cluster counts:\n", gdf['cluster'].value_counts())
    return gdf


def explain_clusters_with_xai(gdf: gpd.GeoDataFrame, output_dir: str):
    """
    Trains surrogate models to explain the HDBSCAN clustering results
    and saves the explanation visuals.
    """
    print("\n--- X. Generating XAI Explanations for Clusters ---")

    # Define the features used in clustering
    feature_cols = ['flood_risk', 'fire_risk_202502', 'heat_risk', 'tree_count_sum', 'tree_connectivity', 'sealed_surfaces' ]

    train_data = gdf[gdf['cluster'] != -1].copy()
    print("Data type of 'cluster' column:", train_data['cluster'].dtype)
    print("Unique values in 'cluster' column:", train_data['cluster'].unique())
    
    if train_data.empty:
        print("No clusters found to explain (all points are noise). Skipping XAI.")
        return

    X = train_data[feature_cols]
    y = train_data['cluster']
    y = y.astype(int)
    # --- Explanation 1: Decision Tree Visualization ---
    print("Generating Decision Tree explanation...")
    surrogate_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    surrogate_tree.fit(X, y)

    # --- ADD THIS NEW DIAGNOSTIC CODE ---
    print("\n--- Direct Tree Inspection ---")
    # The 'tree_' attribute holds the underlying tree structure
    # The 'value' attribute of the tree_ object is a 3D numpy array of shape [n_nodes, 1, n_classes]
    tree_values = surrogate_tree.tree_.value
    
    print("Shape of the internal 'value' array:", tree_values.shape)
    print("Data type (dtype) of the internal 'value' array:", tree_values.dtype)
    
    # Squeeze the array to remove the unnecessary dimension for easier viewing
    # and print the values for the first few nodes.
    print("Values for the first 5 nodes:\n", tree_values.squeeze()[:5])
    print("--------------------------\n")
    # --- END OF DIAGNOSTIC CODE ---

    plt.figure(figsize=(60, 25))
    plot_tree(surrogate_tree,
              feature_names=X.columns.tolist(),
              class_names=[str(c) for c in sorted(y.unique())],
              filled=True,
              rounded=True,
              fontsize=10,
             )
    
    tree_plot_path = os.path.join(output_dir, 'cluster_decision_tree_refined.png')
    plt.title("Key Rules for Cluster Assignments (Decision Tree, Depth 3)", fontsize=24)
    plt.savefig(tree_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Refined Decision Tree plot saved to '{tree_plot_path}'")


    # --- Explanation 2: SHAP Summary Plot (Using RandomForest) ---
    print("\nGenerating SHAP explanation...")
    
    # Using RandomForestClassifier to avoid the persistent XGBoost CUDA bug
    surrogate_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    surrogate_rf.fit(X, y)

    # SHAP's TreeExplainer works perfectly with scikit-learn models
    explainer = shap.TreeExplainer(surrogate_rf)
    shap_values = explainer.shap_values(X)

    fig, ax = plt.subplots(figsize=(15, 10))
    shap.summary_plot(shap_values, X, plot_type="bar", class_names=[str(c) for c in sorted(y.unique())], show=False, plot_size=None)
    
    # --- LEGEND FIX STARTS HERE ---
    # Move the legend to the right of the plot
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    # Adjust plot to make space for the legend
    fig.tight_layout()
    
    shap_plot_path = os.path.join(output_dir, 'cluster_shap_summary_dot.png')
    plt.title("Feature Impact on Cluster Assignment (SHAP Summary Plot)", fontsize=18)
    plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Refined SHAP summary plot saved to '{shap_plot_path}'")

def create_and_save_cluster_map(gdf: gpd.GeoDataFrame, output_path: str):
    """
    Generates and saves a map of the clustered points, coloring each point by its cluster ID.
    """
    print("\n--- Creating and Saving Cluster Map ---")
    if 'cluster' not in gdf.columns:
        print("Error: 'cluster' column not found in GeoDataFrame. Cannot create map.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # Separate noise points (cluster == -1) to plot them differently
    noise = gdf[gdf['cluster'] == -1]
    clustered_points = gdf[gdf['cluster'] != -1]
    
    # Plot the actual clustered points with a color map
    if not clustered_points.empty:
        clustered_points.plot(
            column='cluster',
            ax=ax,
            legend=True,
            cmap='viridis', # A visually distinct color map
            markersize=5,
            categorical=True, # Treat cluster IDs as categories
            legend_kwds={'title': "Cluster ID", 'bbox_to_anchor': (1.05, 1), 'loc': 'upper left'}
        )
    
    # Plot noise points in grey with a different marker
    if not noise.empty:
        noise.plot(color='grey', marker='x', markersize=5, ax=ax, label='Noise / Outliers')

    ax.set_title('Geospatial Distribution of Clusters', fontsize=16)
    ax.set_axis_off()
    
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Cluster map successfully saved to '{output_path}'")
    except Exception as e:
        print(f"Error saving map: {e}")
    finally:
        plt.close(fig)

def summarize_clusters_and_outliers(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Separates outliers, summarizes clean clusters by centroid, and recombines them.
    This version adds a descriptive summary for each cluster's features.
    """
    print("\n--- 5. Summarizing Clusters and Outliers ---")

    special_outlier_cols = [
        'climate_driven_impassable_roads', 'emergency_assemble_areas', 'comments',
        'places_of_interest', 'densely_populated_at_risk_people', 'medical_care'
    ]

    gdf_for_masking = gdf[special_outlier_cols].copy()
    string_cols_to_clean = gdf_for_masking.select_dtypes(include=['object', 'string']).columns
    for col in string_cols_to_clean:
        gdf_for_masking[col] = gdf_for_masking[col].replace(r'^\s*$', np.nan, regex=True)
    special_outlier_mask = gdf_for_masking.notna().any(axis=1)

    outliers_gdf = gdf[(gdf['cluster'] == -1) | (special_outlier_mask)].copy()
    outliers_gdf['cluster'] = -1
    
    clusters_gdf = gdf[(gdf['cluster'] != -1) & (~special_outlier_mask)].copy()
    
    if not clusters_gdf.empty:
        print("\n--- Cluster Feature Summary Statistics ---")
        feature_cols_for_summary = [
            'flood_risk', 'fire_risk_202502', 'heat_risk',
            'tree_count_sum', 'tree_connectivity'
        ]
        
        # 1. Get descriptive statistics
        cluster_summary_stats = clusters_gdf.groupby('cluster')[feature_cols_for_summary].describe()
        #print(cluster_summary_stats)
        #print("----------------------------------------\n")
        
        # 2. Save the descriptive statistics to a text file
        summary_filename = os.path.join(OUTPUT_DIR, 'cluster_summary_stats.txt')
        with open(summary_filename, 'w') as f:
            f.write(cluster_summary_stats.to_string())
        print(f"Cluster summary statistics saved to '{summary_filename}'")

        # 3. Create the aggregated summary GeoDataFrame
        numerical_cols = clusters_gdf.select_dtypes(include=np.number).columns.tolist()
        agg_dict = {col: 'mean' for col in numerical_cols}
        if 'cluster' in agg_dict:
            del agg_dict['cluster']
        
        if 'h3_10' in clusters_gdf.columns:
            agg_dict['h3_10'] = 'first'

        def get_centroid(points):
            if all(p is None or p.is_empty for p in points):
                return None
            return gpd.GeoSeries(points).union_all().centroid
        
        agg_dict['geometry'] = get_centroid
        
        cluster_summary_df = clusters_gdf.groupby('cluster').agg(agg_dict).reset_index()

        cluster_summary_gdf = gpd.GeoDataFrame(
            cluster_summary_df, 
            geometry='geometry', 
            crs=gdf.crs  # Inherit CRS from the original GeoDataFrame
        )
    else:
        print("No clean clusters were found to summarize.")
        cluster_summary_gdf = gpd.GeoDataFrame()
    
    # PART 4: COMBINE SUMMARIZED CLUSTERS AND INDIVIDUAL OUTLIERS
    outliers_gdf.reset_index(drop=True, inplace=True)
    

    final_summary_gdf = pd.concat([cluster_summary_gdf, outliers_gdf], ignore_index=True)
    final_summary_gdf = gpd.GeoDataFrame(final_summary_gdf, geometry='geometry')
    final_summary_gdf.loc[final_summary_gdf['cluster'] != -1, 'h3_10'] = None

    print("Final summary GeoDataFrame created successfully.")
    print(f"Total rows in summary: {len(final_summary_gdf)}")
    final_summary_gdf.to_csv(FINAL_SUMMARY_PATH, index=False)
    print(f"Cluster summarization complete. Saved to '{FINAL_SUMMARY_PATH}'")
    
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
            await asyncio.sleep(0.5) # Slight delay to avoid overwhelming the LLM
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
    #df_validated = validate_risk_with_llm(df_cleaned)
    gdf, data_for_clustering = engineer_features_for_clustering(df_cleaned)
    gdf_clustered = perform_hdbscan_clustering(gdf, data_for_clustering)
    explain_clusters_with_xai(gdf_clustered, OUTPUT_DIR)
    create_and_save_cluster_map(gdf_clustered, CLUSTER_MAP_PATH)
    gdf_summary = summarize_clusters_and_outliers(gdf_clustered)
    
    solution_results = await generate_solution_actions_with_llm(gdf_summary)
    
    final_gdf = map_solutions_to_dataframe(gdf_clustered, solution_results)

    print(f"\n--- 8. Saving Final Outputs ---")
    final_gdf.to_csv(FINAL_COMPLETE_PATH_CSV)
    print(f"Final data saved to '{FINAL_COMPLETE_PATH_CSV}'")
    
    final_gdf.columns = final_gdf.columns.astype(str)
    ##final_gdf.to_file(FINAL_COMPLETE_PATH_GEOJSON, driver="GeoJSON")
    #final_gdf.to_file(FINAL_COMPLETE_PATH_SHAPE)
    #print(f"Final GeoDataFrame saved to '{FINAL_COMPLETE_PATH_SHAPE}'")
    geopackage_path = os.path.join(OUTPUT_DIR, "final_data.gpkg")
    final_gdf.to_file(geopackage_path, driver="GPKG")
    print(f"Final GeoDataFrame saved to '{geopackage_path}'")

if __name__ == "__main__":
    import time
    start_time = time.time() 
    asyncio.run(main_async())

    end_time = time.time()

    duration = end_time - start_time
    
    print(f"\n--- SCRIPT FINISHED ---")
    print(f"Total execution time: {duration:.2f} seconds")
    print(f"Which is equal to: {duration/60:.2f} minutes")