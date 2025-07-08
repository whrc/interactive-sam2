# tests/test_gcs_access.py

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import geopandas as gpd

# Add the project root to the system path to allow imports from src.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.interactive_label_sam2.gcs_utils import (
    load_correspondence_data, 
    get_image_info_for_uid, 
    GCSImageLoader
)

def run_test():
    """
    A test function to verify the entire data access pipeline.
    This version saves its output to the tests/outputs directory.
    """
    print("--- Starting GCS Access Test ---")

    try:
        # --- 1. Initialize the GCSImageLoader ---
        gcs_loader = GCSImageLoader(
            project_id="abruptthawmapping",
            bucket_name="abrupt_thaw",
            search_prefix="planet_basemaps/global_quarterly_COGs"
        )

        # --- 2. Define Paths and Test UID ---
        correspondence_path = project_root / "data" / "raw" / "planet_basemaps_rts_polygon_basemap_correspondence.geojson"
        test_uid = "6aff1955-71f5-5fa5-97d1-d9e006e4ec5c"
        # Define the output directory for test results
        output_dir = project_root / "tests" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[Step 1] Using test UID: {test_uid}")

        # --- 3. Load All Necessary Metadata ---
        print("\n[Step 2] Loading metadata files...")
        correspondence_gdf = load_correspondence_data(correspondence_path)
        if correspondence_gdf is None: return

        # --- 4. Find the Image Search Info ---
        print(f"\n[Step 3] Finding image info for UID: {test_uid}")
        image_info_list = get_image_info_for_uid(test_uid, correspondence_gdf)
        if not image_info_list:
            print("Test failed: Could not find any image info.")
            return

        # --- 5. Loop through each target image and test it individually ---
        for image_info in image_info_list:
            print(f"\n--- Testing Image ID: {image_info['image_id']} for Year: {image_info['year']} ---")

            # --- 5a. Find the specific polygon for THIS image ---
            print("[Step 4a] Finding specific polygon for this image...")
            
            feature_gdf = correspondence_gdf[
                (correspondence_gdf['UID'] == test_uid) &
                (correspondence_gdf['id'] == image_info['image_id']) &
                (correspondence_gdf['planet_basemap_year'] == image_info['year'])
            ]

            if feature_gdf.empty:
                print("Could not find the specific feature in the correspondence file.", file=sys.stderr)
                continue
            
            # --- 5b. Define AOI using a buffered centroid ---
            print("[Step 4b] Defining AOI using a buffered centroid...")
            centroid = feature_gdf.geometry.iloc[0].centroid
            centroid_gs = gpd.GeoSeries([centroid], crs=feature_gdf.crs)
            buffer_distance = 384 
            aoi_polygon = centroid_gs.buffer(buffer_distance)
            aoi_bounds = aoi_polygon.total_bounds
            print(f"Using AOI bounds: {aoi_bounds}")

            # --- 5c. Find exact path using the loader's fast in-memory search ---
            print(f"[Step 4c] Searching for exact GCS path...")
            gcs_paths = gcs_loader.find_image_paths([image_info])
            if not gcs_paths:
                print("Test failed: Could not find any GCS paths in the pre-fetched list.")
                continue 

            # --- 5d. Attempt to Download the Image Tile using the buffered AOI ---
            print("\n[Step 4d] Attempting to download image tile from GCS...")
            tile_data = gcs_loader.get_tile_from_paths(gcs_paths, aoi_bounds)
            
            if tile_data:
                image_array, profile = tile_data
                print(f"Successfully downloaded image tile with shape: {image_array.shape}")
                
                # --- 5e. Save a Preview ---
                print(f"\n[Step 4e] Saving a preview image to '{output_dir.name}/test_tile.png'...")
                rgb_array = image_array[:3]
                
                if np.max(rgb_array) > 0:
                    p2, p98 = np.percentile(rgb_array, (2, 98))
                    rgb_stretched = np.clip((rgb_array - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
                    rgb_transposed = np.transpose(rgb_stretched, (1, 2, 0))
                else:
                    print("Warning: Image data is all zeros. Creating a black preview.")
                    rgb_transposed = np.zeros((image_array.shape[1], image_array.shape[2], 3), dtype=np.uint8)

                transform = profile['transform']
                col, row = ~transform * (centroid.x, centroid.y)

                fig, ax = plt.subplots(1, figsize=(6, 6))
                ax.imshow(rgb_transposed)
                ax.scatter([col], [row], c='red', s=50, marker='+', label='Centroid')
                ax.set_title(f"Test Tile for UID: {test_uid[:8]}")
                ax.legend()
                ax.axis('off')
                
                # Save the file to the correct output directory
                output_image_path = output_dir / "test_tile.png"
                fig.savefig(output_image_path, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
                print("Preview image saved successfully.")

                print("\n--- GCS Access Test Finished Successfully! ---")
                return
            else:
                print("Could not retrieve a tile for this image. Trying next if available.")

        print("\n--- GCS Access Test Failed: Could not retrieve a tile from any of the identified images. ---")

    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred During Test ---")
        print(f"Error: {e}")


if __name__ == "__main__":
    run_test()
