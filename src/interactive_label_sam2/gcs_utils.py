# src/rts_labeling_tool/gcs_utils.py

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Point
from pathlib import Path
import sys

# We will need gcsfs to allow rasterio to read directly from GCS
import gcsfs

def load_correspondence_data(correspondence_path: Path) -> gpd.GeoDataFrame | None:
    """
    Loads the UID-to-basemap correspondence GeoJSON file.

    Args:
        correspondence_path (Path): Path to the correspondence GeoJSON.

    Returns:
        gpd.GeoDataFrame | None: A GeoDataFrame of the correspondence data, or None on error.
    """
    if not correspondence_path.exists():
        print(f"Error: Correspondence file not found at {correspondence_path}", file=sys.stderr)
        return None
    
    try:
        gdf = gpd.read_file(correspondence_path, engine="pyogrio")
        print("Correspondence file loaded successfully.")
        return gdf
    except Exception as e:
        print(f"Error loading correspondence file: {e}", file=sys.stderr)
        return None

def get_image_paths_for_uid(uid: str, correspondence_gdf: gpd.GeoDataFrame) -> list[str]:
    """
    Finds the GCS paths for all basemap images associated with a given UID.

    It handles corner/edge cases with multiple images and selects the latest
    available year, prioritizing a common year if possible.

    Args:
        uid (str): The unique identifier for the RTS feature.
        correspondence_gdf (gpd.GeoDataFrame): The correspondence data.

    Returns:
        list[str]: A list of full GCS paths to the required image tiles.
    """
    # Filter for the specific UID
    uid_features = correspondence_gdf[correspondence_gdf['UID'] == uid].copy()

    if uid_features.empty:
        print(f"Warning: No basemap correspondence found for UID: {uid}", file=sys.stderr)
        return []

    # --- Year Selection Logic ---
    # Find the latest year available for each unique image ID
    uid_features['planet_basemap_year'] = uid_features['planet_basemap_year'].astype(int)
    latest_years = uid_features.groupby('id')['planet_basemap_year'].max()

    # Find the most common latest year among the different image IDs
    if not latest_years.empty:
        common_year = latest_years.mode()[0]
    else:
        # Fallback if no years are found
        print(f"Warning: No years found for UID {uid}. Defaulting to 2023.", file=sys.stderr)
        common_year = 2023

    # Filter the features to use only the selected year
    final_features = uid_features[uid_features['planet_basemap_year'] == common_year]

    # --- Path Construction ---
    gcs_paths = []
    bucket_name = "abrupt_thaw"
    base_path = f"{bucket_name}/planet_basemaps/global_quarterly_COGs"

    for _, row in final_features.iterrows():
        year = int(row['planet_basemap_year'])
        # Assuming quarter is always q3 for simplicity as per example.
        # This could be made more robust if quarter info is in the data.
        quarter_str = f"{year}q3"
        image_id = row['id'] # e.g., '2795-3415'
        
        # Construct the folder and file names based on the user's structure
        mosaic_name = f"global_quarterly_{quarter_str}_mosaic"
        # The sub-subfolder seems to use the image_id with a different format
        # e.g., 10-3133_1000-3037. We will assume the 'id' field is what we need.
        sub_subfolder = f"{mosaic_name}_{image_id}"
        
        # The filename seems to use the image_id and a 'quad' suffix
        file_name = f"{mosaic_name}_{image_id}_quad.tif"

        # This path construction is based on the provided example. It might need adjustment.
        # Example: abrupt_thaw/planet_basemaps/global_quarterly_COGs/global_quarterly_2016q3_mosaic/global_quarterly_2016q3_mosaic_10-3133_1000-3037
        # The user's example path for the sub-subfolder seems different from the filename pattern.
        # We will follow the filename pattern for now as it seems more consistent.
        # A potential issue is the subsubfolder name vs the file name id.
        # Let's assume the 'id' field corresponds to the final part of the filename.
        
        # Re-evaluating the path based on user's example:
        # root: abrupt_thaw/planet_basemaps/global_quarterly_COGs
        # subfolder: global_quarterly_2016q3_mosaic/
        # subsubfolder: global_quarterly_2016q3_mosaic_10-3133_1000-3037/
        # file name: global_quarterly_2016q3_mosaic_1000-2818_quad.tif
        # The subsubfolder ID and file ID don't match in the example. This is a critical ambiguity.
        # For now, we will assume the 'id' field from the correspondence file refers to the *file* ID.
        # We will need to clarify how to derive the subsubfolder name.
        # Let's assume for now the subsubfolder is just the mosaic_name.
        
        full_path = f"gs://{base_path}/{mosaic_name}/{file_name}"
        gcs_paths.append(full_path)

    print(f"For UID {uid}, found {len(gcs_paths)} image paths for year {common_year}: {gcs_paths}")
    return gcs_paths


def get_image_tile(gcs_paths: list[str], center_point: Point, tile_size: int = 256) -> tuple[object, object] | None:
    """
    Reads a tile from a Cloud Optimized GeoTIFF on GCS.
    
    Note: For now, if multiple paths are provided (corner/edge case), it will
    try to use the first one that contains the center_point. A more advanced
    implementation would mosaic the tiles.

    Args:
        gcs_paths (list[str]): List of GCS paths to the source images.
        center_point (Point): The center coordinate for the tile (in the image's CRS).
        tile_size (int): The desired tile size in pixels (e.g., 256 for 256x256).

    Returns:
        tuple | None: A tuple containing:
                      - The image data as a NumPy array.
                      - The rasterio profile (metadata).
                      Returns None on error.
    """
    if not gcs_paths:
        return None

    # Authenticate with GCS. In Colab, this will trigger a user prompt.
    # Locally, it uses default credentials if configured.
    gcs = gcsfs.GCSFileSystem()

    for path in gcs_paths:
        try:
            # Use rasterio to open the file directly from GCS
            with rasterio.open(path, 'r', gcs=gcs) as src:
                # Check if the center point is within the image bounds
                if not (src.bounds.left < center_point.x < src.bounds.right and \
                        src.bounds.bottom < center_point.y < src.bounds.top):
                    continue # Try the next image

                # Get pixel resolution
                x_res, y_res = src.res
                
                # Calculate the geographic bounds of our desired 256x256 tile
                half_width = (tile_size / 2) * x_res
                half_height = (tile_size / 2) * y_res
                
                bounds = (
                    center_point.x - half_width,
                    center_point.y - half_height,
                    center_point.x + half_width,
                    center_point.y + half_height
                )

                # Read the data from that window
                window = from_bounds(*bounds, src.transform)
                data = src.read(window=window)
                
                # Get the profile for this specific window to save later
                profile = src.profile
                profile.update({
                    'height': data.shape[1],
                    'width': data.shape[2],
                    'transform': src.window_transform(window)
                })

                print(f"Successfully read tile from {path}")
                return data, profile

        except Exception as e:
            print(f"Could not read tile from {path}. Error: {e}", file=sys.stderr)
            continue
    
    print(f"Error: Could not read a tile for point {center_point.wkt} from any of the provided paths.", file=sys.stderr)
    return None

