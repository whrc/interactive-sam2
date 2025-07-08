# src/interactive_label_sam2/gcs_utils.py

import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Point, Polygon
from pathlib import Path
import sys
from typing import Union, List, Tuple

import gcsfs
import google.auth
import google.auth.transport.requests

class GCSImageLoader:
    """
    A class to efficiently find and load Planet image tiles from GCS.
    This design is inspired by the user's successful batch processing script.
    """
    def __init__(self, project_id: str, bucket_name: str, search_prefix: str):
        """
        Initializes the loader by authenticating and pre-fetching a list of all
        relevant blob paths to enable fast, in-memory searching.
        """
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.blob_paths = []

        try:
            print("--- Initializing GCSImageLoader ---")
            scopes = ['https://www.googleapis.com/auth/cloud-platform']
            credentials, discovered_project_id = google.auth.default(scopes=scopes)
            if not credentials.valid:
                credentials.refresh(google.auth.transport.requests.Request())
            
            self.project_id = discovered_project_id if discovered_project_id else self.project_id
            self.gcs = gcsfs.GCSFileSystem(project=self.project_id, token=credentials)
            
            print(f"Pre-fetching all blob paths from gs://{self.bucket_name}/{search_prefix}...")
            full_prefix_path = f"{self.bucket_name}/{search_prefix}"
            self.blob_paths = self.gcs.glob(f"{full_prefix_path}/**/*.tif")
            print(f"Found {len(self.blob_paths)} total .tif files. Ready for fast searching.")
            print("--- GCSImageLoader Initialized Successfully ---\n")

        except Exception as e:
            print(f"FATAL: Failed to initialize GCSImageLoader. Error: {e}", file=sys.stderr)
            raise

    def find_image_paths(self, image_info_list: List[dict]) -> List[str]:
        """
        Searches the pre-fetched in-memory list of paths for the required images.
        """
        all_found_paths = []
        for info in image_info_list:
            year = info['year']
            image_id = info['image_id']
            
            found = [
                path for path in self.blob_paths
                if str(year) in path and image_id in path and 'quad' in path
            ]
            if found:
                all_found_paths.extend(found)
        
        unique_paths = sorted(list(set(all_found_paths)))
        print(f"Found {len(unique_paths)} matching image path(s): {unique_paths}")
        return unique_paths

    def get_tile_from_paths(self, gcs_paths: List[str], aoi_polygon_gdf: gpd.GeoDataFrame) -> Union[Tuple[object, object], None]:
        """
        Reads a tile from a list of GCS paths, using the bounds of the
        provided Area of Interest (AOI) GeoDataFrame. It ensures CRS alignment.
        """
        if not gcs_paths:
            return None
        
        for path in gcs_paths:
            try:
                with self.gcs.open(path, 'rb') as f:
                    with rasterio.open(f) as src:
                        print(f"\n--- Processing Image: {Path(path).name} ---")
                        print(f"Vector CRS: {aoi_polygon_gdf.crs} | Raster CRS: {src.crs}")

                        # Ensure the AOI polygon is in the same CRS as the raster
                        if aoi_polygon_gdf.crs != src.crs:
                            print("CRS mismatch detected. Reprojecting AOI polygon...")
                            aoi_reprojected = aoi_polygon_gdf.to_crs(src.crs)
                        else:
                            aoi_reprojected = aoi_polygon_gdf

                        # Get the bounding box of the (potentially reprojected) AOI
                        aoi_bounds = aoi_reprojected.total_bounds
                        print(f"Reading window with bounds: {aoi_bounds}")

                        # Define the read window using the polygon's bounds
                        window = from_bounds(*aoi_bounds, src.transform)

                        # Read the data from that window
                        data = src.read(window=window)
                        
                        if data.shape[1] == 0 or data.shape[2] == 0:
                            print("Warning: Read an empty tile. The AOI may not overlap with the image data.", file=sys.stderr)
                            continue

                        # Get the profile for this specific window to save later
                        profile = src.profile
                        profile.update({
                            'height': data.shape[1],
                            'width': data.shape[2],
                            'transform': src.window_transform(window),
                            'crs': src.crs
                        })

                        print(f"Successfully read tile from {path}")
                        return data, profile
            except Exception as e:
                print(f"Could not read tile from {path}. Error: {e}", file=sys.stderr)
                continue
        
        print(f"Error: Could not read a tile for the given AOI from any of the provided paths.", file=sys.stderr)
        return None

def load_correspondence_data(correspondence_path: Path) -> Union[gpd.GeoDataFrame, None]:
    """
    Loads the UID-to-basemap correspondence GeoJSON file and enforces EPSG:3413.
    """
    if not correspondence_path.exists():
        print(f"Error: Correspondence file not found at {correspondence_path}", file=sys.stderr)
        return None
    
    try:
        gdf = gpd.read_file(correspondence_path, engine="pyogrio")
        
        # Enforce the correct CRS, as requested by the user.
        if gdf.crs != "EPSG:3413":
            print(f"Warning: Original CRS is {gdf.crs}. Forcing to EPSG:3413.")
            gdf = gdf.set_crs("EPSG:3413", allow_override=True)

        print(f"Correspondence file loaded successfully with CRS: {gdf.crs}")
        return gdf
    except Exception as e:
        print(f"Error loading correspondence file: {e}", file=sys.stderr)
        return None

def get_image_info_for_uid(uid: str, correspondence_gdf: gpd.GeoDataFrame) -> List[dict]:
    """
    Finds the necessary info (year, image_id) to locate basemap images for a given UID.
    """
    uid_features = correspondence_gdf[correspondence_gdf['UID'] == uid].copy()

    if uid_features.empty:
        print(f"Warning: No basemap correspondence found for UID: {uid}", file=sys.stderr)
        return []

    image_id_year_mapping = uid_features.groupby('id')['planet_basemap_year'].apply(list).to_dict()

    if not image_id_year_mapping:
        print(f"Warning: No planet image IDs found for UID {uid}", file=sys.stderr)
        return []

    year_sets = [set(years) for years in image_id_year_mapping.values()]
    common_years = list(set.intersection(*year_sets)) if year_sets else []

    selected_year = None
    if common_years:
        selected_year = max(common_years)
    else:
        all_years = [year for years_list in image_id_year_mapping.values() for year in years_list]
        if all_years:
            selected_year = max(all_years)

    if not selected_year:
        print(f"Error: No available years found for UID {uid}", file=sys.stderr)
        return []

    image_info_list = []
    for image_id, years in image_id_year_mapping.items():
        if selected_year in years:
            image_info_list.append({'year': int(selected_year), 'image_id': image_id})
            
    return image_info_list
