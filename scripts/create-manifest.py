# scripts/create-manifest.py

import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys
import json

def create_manifest():
    """
    Reads the main ARTS GeoJSON dataset, filters for features marked as 'Positive',
    extracts their unique UIDs, and creates a master CSV manifest file to track
    the labeling progress.
    """
    try:
        # --- 1. Define File Paths ---
        project_root = Path(__file__).parent.parent
        geojson_path = project_root / "data" / "raw" / "ARTS_main_dataset_v.3.1.0.geojson"
        manifest_path = project_root / "manifest.csv"

        # --- 2. Check if Manifest Already Exists ---
        if manifest_path.exists():
            print(f"Manifest file already exists at: {manifest_path}")
            overwrite = input("Do you want to overwrite it? (y/n): ").lower()
            if overwrite != 'y':
                print("Operation cancelled by user.")
                return

        # --- 3. Validate and Load the GeoJSON Data ---
        print(f"Reading data from: {geojson_path}")
        if not geojson_path.exists():
            print(f"Error: GeoJSON file not found at {geojson_path}", file=sys.stderr)
            print("Please ensure the dataset is correctly placed in the 'data/raw' directory.", file=sys.stderr)
            return

        # --- DEBUGGING STEP: First, validate the file as a standard JSON ---
        try:
            with open(geojson_path, 'r', encoding='utf-8') as f:
                json.load(f)
            print("File successfully validated as a JSON.")
        except json.JSONDecodeError as e:
            print("\nFATAL ERROR: The file is not a valid JSON.", file=sys.stderr)
            print(f"The error is: {e}", file=sys.stderr)
            print("The file may be corrupted or not a proper GeoJSON. Please check the file contents.", file=sys.stderr)
            return
        except Exception as e:
            print(f"\nFATAL ERROR: Could not read the file. Error: {e}", file=sys.stderr)
            return

        # --- If JSON validation passes, proceed with GeoPandas ---
        gdf = gpd.read_file(geojson_path, engine="pyogrio")
        print("GeoJSON file loaded successfully with GeoPandas.")

        # --- 4. Filter for Positive Training Class ---
        if 'TrainClass' not in gdf.columns:
            print("Error: 'TrainClass' column not found in the GeoJSON file.", file=sys.stderr)
            print("Cannot filter for 'Positive' class. Please check the dataset.", file=sys.stderr)
            return
        
        print(f"Total polygons found in file: {len(gdf)}")
        gdf_positive = gdf[gdf['TrainClass'] == 'Positive'].copy()
        print(f"Found {len(gdf_positive)} polygons marked as 'Positive'.")


        # --- 5. Extract Unique UIDs from the filtered data ---
        if 'UID' not in gdf_positive.columns:
            print("Error: 'UID' column not found in the filtered GeoJSON file.", file=sys.stderr)
            return
            
        unique_uids = gdf_positive['UID'].unique()
        print(f"Found {len(unique_uids)} unique UIDs in the 'Positive' class.")

        # --- 6. Create the Manifest DataFrame ---
        manifest_columns = [
            'uid',
            'labeling_status',
            'worker_id',
            'start_time_utc',
            'end_time_utc',
            'output_filename',
            'notes'
        ]
        
        manifest_df = pd.DataFrame(unique_uids, columns=['uid'])
        manifest_df['labeling_status'] = 'unprocessed'
        
        for col in manifest_columns:
            if col not in manifest_df.columns:
                manifest_df[col] = ''
        
        manifest_df = manifest_df[manifest_columns]

        # --- 7. Save the Manifest File ---
        print(f"Saving manifest file to: {manifest_path}")
        manifest_df.to_csv(manifest_path, index=False)

        print("\nManifest file created successfully!")
        print(f"Total tasks to process: {len(manifest_df)}")

    except Exception as e:
        print(f"\nAn unexpected error occurred during GeoPandas processing: {e}", file=sys.stderr)

if __name__ == "__main__":
    create_manifest()
