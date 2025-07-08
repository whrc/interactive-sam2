# src/rts_labeling_tool/data_management.py

import geopandas as gpd
import json
import sys
from pathlib import Path
from shapely.geometry import Point

def load_and_filter_arts_data(geojson_path: Path) -> gpd.GeoDataFrame | None:
    """
    Loads the ARTS GeoJSON dataset, validates it, and filters for features
    marked as 'Positive'.

    Args:
        geojson_path (Path): The full path to the GeoJSON file.

    Returns:
        gpd.GeoDataFrame | None: A GeoDataFrame containing only the 'Positive'
                                 class features, or None if an error occurs.
    """
    # --- 1. Validate File Existence ---
    if not geojson_path.exists():
        print(f"Error: GeoJSON file not found at {geojson_path}", file=sys.stderr)
        return None

    # --- 2. Validate as a standard JSON file ---
    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            json.load(f)
        print("File successfully validated as a JSON.")
    except json.JSONDecodeError as e:
        print(f"\nFATAL ERROR: The file is not a valid JSON. Error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"\nFATAL ERROR: Could not read the file. Error: {e}", file=sys.stderr)
        return None

    # --- 3. Load with GeoPandas and Filter ---
    try:
        gdf = gpd.read_file(geojson_path, engine="pyogrio")
        print("GeoJSON file loaded successfully with GeoPandas.")

        if 'TrainClass' not in gdf.columns:
            print("Error: 'TrainClass' column not found.", file=sys.stderr)
            return None
        
        print(f"Total polygons found in file: {len(gdf)}")
        gdf_positive = gdf[gdf['TrainClass'] == 'Positive'].copy()
        print(f"Found {len(gdf_positive)} polygons marked as 'Positive'.")
        
        return gdf_positive

    except Exception as e:
        print(f"\nAn unexpected error occurred during GeoPandas processing: {e}", file=sys.stderr)
        return None

def get_feature_info(uid: str, gdf: gpd.GeoDataFrame) -> tuple[list[gpd.GeoSeries], Point] | None:
    """
    Finds all polygons for a given UID and calculates their combined centroid.

    This is used to get the historical context and the central point for
    displaying the feature on the map.

    Args:
        uid (str): The unique identifier for the RTS feature.
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing all positive features.

    Returns:
        tuple | None: A tuple containing:
                      - A list of historical polygon geometries.
                      - The calculated centroid (Shapely Point).
                      Returns None if the UID is not found.
    """
    # Filter the dataframe to get all entries for the specified UID
    feature_gdf = gdf[gdf['UID'] == uid]

    if feature_gdf.empty:
        print(f"Error: No feature found for UID: {uid}", file=sys.stderr)
        return None

    # Get all the historical polygon geometries
    historical_polygons = list(feature_gdf.geometry)

    # Combine all historical polygons into one single geometry
    # The unary_union is a robust way to merge multiple geometries
    combined_geometry = feature_gdf.geometry.unary_union

    # Calculate the centroid of the combined shape
    centroid = combined_geometry.centroid

    return historical_polygons, centroid
