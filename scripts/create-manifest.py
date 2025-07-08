# scripts/create-manifest.py

import pandas as pd
from pathlib import Path
import sys

# To import from the src directory, we add the project root to the system path.
# This is a common pattern for making scripts runnable from the command line.
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.rts_labeling_tool.data_management import load_and_filter_arts_data

def create_manifest():
    """
    Creates a master CSV manifest file for 'Positive' class RTS features
    by using the data_management module.
    """
    try:
        # --- 1. Define File Paths ---
        geojson_path = project_root / "data" / "raw" / "ARTS_main_dataset_v.3.1.0.geojson"
        manifest_path = project_root / "manifest.csv"

        # --- 2. Check if Manifest Already Exists ---
        if manifest_path.exists():
            print(f"Manifest file already exists at: {manifest_path}")
            overwrite = input("Do you want to overwrite it? (y/n): ").lower()
            if overwrite != 'y':
                print("Operation cancelled by user.")
                return

        # --- 3. Load and Filter Data using the dedicated function ---
        gdf_positive = load_and_filter_arts_data(geojson_path)

        if gdf_positive is None:
            print("Failed to load or filter data. Aborting manifest creation.", file=sys.stderr)
            return

        # --- 4. Extract Unique UIDs ---
        if 'UID' not in gdf_positive.columns:
            print("Error: 'UID' column not found in the filtered GeoJSON file.", file=sys.stderr)
            return
            
        unique_uids = gdf_positive['UID'].unique()
        print(f"Found {len(unique_uids)} unique UIDs in the 'Positive' class.")

        # --- 5. Create the Manifest DataFrame ---
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

        # --- 6. Save the Manifest File ---
        print(f"Saving manifest file to: {manifest_path}")
        manifest_df.to_csv(manifest_path, index=False)

        print("\nManifest file created successfully!")
        print(f"Total tasks to process: {len(manifest_df)}")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    create_manifest()
