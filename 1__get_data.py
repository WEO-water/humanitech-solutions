import os
import pandas as pd
import geopandas as gpd
import sys
from functools import reduce
import contextily as ctx
import numpy as np
import h3pandas


from prompt_func import generate_risk_actions, map_data
from upload_gcs import upload_to_gcs_with_timestamp


FELT_DATA_DIR = 'gs://dl-test-439308-bucket/weo-data/dashboard'#'/mnt/fvw/data/tmp/humanitech/dashboard'
DB_PTH_DCT = {
    
    'metrics': 'Heat-Risk-.zip',
    'medical_care': 'medical_care.geojson',
    'climate_driven_impassable_roads': 'climate_driven_impassable_roads.geojson',
    'densely_populated_at_risk_people': 'densely_populated_at_risk_people.geojson',
    'emergency_assemble_areas': 'emergency_assemble_areas.geojson',
    'places_of_interest': 'places_of_interest.geojson',
    'comments': 'comments_20250708_123244.zip',

}

INDEX = column_to_merge_on = 'h3_10'#'felt:h3_index'#'h3_13'



def collapse_duplicates(df, index_col):
    def collapse_strings(series):
        if series.dtype == object:
            values = series.dropna().astype(str)
            if values.empty:
                return np.nan
            seen = set()
            unique_values = []
            for v in values:
                if v and v not in seen:
                    unique_values.append(v)
                    seen.add(v)
            return ', '.join(unique_values)
        else:
            return series.dropna().iloc[0] if series.notna().any() else pd.NA

    collapsed_df = df.groupby(index_col).agg(collapse_strings).reset_index()
    return collapsed_df



db_gdfs = {}


for key, filename in DB_PTH_DCT.items():
    ext = os.path.splitext(filename)[1].lower()
    path = os.path.join(FELT_DATA_DIR, filename)
    if ext in [".csv"]:
        df = pd.read_csv(path)
        db_gdfs[key] = df
    elif ext in [".geojson", ".gpkg", ".zip"]:
        try:
            gdf = gpd.read_file(path)

            if gdf.geometry.name != 'geometry':
                raise ValueError(f"GeoDataFrame {gdf.name} does not have a 'geometry' column.")
            if gdf.crs is None:
                raise ValueError(f"GeoDataFrame {gdf.name} does not have a CRS defined.")

            if gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
            if key != 'comments':
                gdf = gdf.rename(columns={'name': key})
            elif key == 'comments':
                gdf = gdf.rename(columns={'text': key})

            # Handle different geometry types for H3 assignment
            if gdf.geometry.iloc[0].geom_type == "Point":
                # For Point geometries, assign H3 index directly
                gdf = gdf.h3.geo_to_h3(resolution=10, set_index=False)
            elif gdf.geometry.iloc[0].geom_type == "MultiPoint":
                # For MultiPoint geometries, calculate centroid and assign H3 index
                gdf = gdf.explode(ignore_index=True)
                gdf = gdf.h3.geo_to_h3(resolution=10, set_index=False)
            elif gdf.geometry.iloc[0].geom_type == "Polygon":
                gdf = gdf.h3.polyfill(10+4, explode=True).set_index('h3_polyfill').h3.h3_to_parent_aggregate(10, operation = {'emergency_assemble_areas': 'first',})  # Take the first value in each group# Add other columns as needed, e.g., 'count': 'sum'
                gdf = gdf.reset_index()
            else: 
                print(f"Unsupported geometry type {gdf.geometry.iloc[0].geom_type} for {key} in {filename}. Skipping H3 assignment.", file=sys.stderr)
            
            gdf['h3_10_int'] = gdf['h3_10'].apply(lambda x: int(x, 16) if pd.notna(x) else None)
            db_gdfs[key] = gdf

        except Exception as e:
            db_gdfs[key] = None
            print(f"Error with {key} DB, {filename}: {e}", file=sys.stderr)
    else:
        db_gdfs[key] = None



# Merge all DataFrames in db_gdfs on the 'INDEX' column
merged_gdf = reduce(lambda left, right: pd.merge(left, right, on=INDEX, how='outer', suffixes=('', '_dup')),  [df for df in db_gdfs.values() if isinstance(df, pd.DataFrame) and INDEX in df.columns])



#Filter the polygons that have a special feature, e.g., 'densely_populated_at_risk_people'
# Get all keys except 'metrics'
non_metrics_keys = [k for k in db_gdfs.keys() if k != 'metrics']
# Only keep columns that exist in merged_gdf
cols_to_check = [k for k in non_metrics_keys if k in merged_gdf.columns]
# # Select rows where any of these columns is notna
pois_df = merged_gdf[merged_gdf[cols_to_check].notna().any(axis=1)]
print(pois_df.shape)
print(f"number of unique h3 indices: {pois_df[INDEX].nunique()}")



filtered_df = merged_gdf[merged_gdf['flood_risk'].notna() & merged_gdf['heat_risk'].notna() & merged_gdf['fire_risk_202502'].notna()]	
print(f"Filtered DataFrame shape: {filtered_df.shape}")
selected_rows = filtered_df[filtered_df['tree_count_sum']<15]



all_in_selected = pois_df[INDEX].isin(selected_rows[INDEX]).all()

if not all_in_selected:
    missing_idx = pois_df[~pois_df[INDEX].isin(selected_rows[INDEX])]
    selected_rows = pd.concat([selected_rows, missing_idx], ignore_index=True)
    print(f"Added {len(missing_idx)} missing rows from pois_df to selected_rows.")



selected_rows.to_csv("df_export.csv", index=False)
upload_to_gcs_with_timestamp(bucket_name="dl-test-439308-bucket", local_file_path="df_export.csv", gcs_prefix="weo-data/dashboard/")







for key, gdf in db_gdfs.items():
    if gdf is not None:
        print(f"{key} unique polygons on total rows in DB: {gdf[INDEX].nunique()}/{len(gdf)}.")
    else:
        print(f"{key} DB is None or empty.")



ax = selected_rows.to_crs(epsg=3857).plot(figsize=(10, 10), alpha=0.5)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_axis_off()









