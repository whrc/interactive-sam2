# main.py
# Standalone Python script for the Interactive RTS Labeling Tool.
# This script is intended to be run in an environment that supports ipywidgets,
# such as Jupyter, JupyterLab, or Google Colab.

# ==============================================================================
# PRE-REQUISITES (Run these commands in your terminal/environment first)
# ==============================================================================
#
# pip install torch transformers ipywidgets ipycanvas pandas geopandas rasterio Pillow
#
# # Ensure you have the project repository cloned
# # git clone https://github.com/whrc/interactive-sam2.git
# # cd interactive-sam2
# # git lfs pull
# # cd ..
#
# ==============================================================================
#
# ### --- Section 1: Import Libraries --- ###
import ipywidgets as widgets
from IPython.display import display, clear_output
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import geopandas as gpd
import rasterio
from ipycanvas import Canvas, hold_canvas
import importlib

# ### --- Section 2: Environment-Specific Setup (Colab) --- ###
IS_COLAB = 'google.colab' in sys.modules

if IS_COLAB:
    print("Colab environment detected. Running Colab-specific setup...")
    from google.colab import output, auth, drive
    output.enable_custom_widget_manager()
    try:
        auth.authenticate_user()
        drive.mount('/content/drive')
        print("Google Drive mounted successfully.")
    except Exception as e:
        print(f"Error during Colab setup: {e}")
else:
    print("Non-Colab environment detected. Skipping Colab-specific setup.")

# ### --- Section 3: Add Source Code to Python Path and Import --- ###
# This assumes the script is run from the directory containing 'interactive-sam2'
project_root = Path(".")
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from interactive_label_sam2.model import SAM2Model
    from interactive_label_sam2.gcs_utils import GCSImageLoader, load_correspondence_data, get_image_info_for_uid
    from interactive_label_sam2.data_management import load_and_filter_arts_data
    print(" Custom modules imported successfully.")
except ImportError as e:
    print(f" FATAL: Could not import custom modules. Ensure the 'interactive-sam2/src' directory is correct. Error: {e}")
    # Exit if modules can't be found
    sys.exit(1)


# ### --- Section 4: Application State and Global Constants --- ###
print("\\n--- Defining Application State and UI Components ---")
DRIVE_ROOT = Path("/content/drive/MyDrive/Interactive_sam") if IS_COLAB else Path("./data_output")
OUTPUT_DIR = DRIVE_ROOT / "test"
MANIFEST_PATH = OUTPUT_DIR / "rts_labeling_manifest.csv"

APP_STATE = {
    "worker_id": None, "manifest_df": None, "arts_gdf": None,
    "correspondence_gdf": None, "gcs_loader": None, "sam_model": None,
    "current_uid": None, "current_image_array": None, "current_display_image": None,
    "current_tile_profile": None, "current_mask": None,
    "prompts": [],
    "box_prompts": [],
    "final_boxes": [],
    "mask_visible": True,
    "canvas": None
}

# ### --- Section 5: UI Widget Definitions --- ###
worker_id_input = widgets.Text(value='worker_01', description='Worker ID:', layout=widgets.Layout(width='300px'))
start_button = widgets.Button(description="Start Labeling Session", button_style='success')
next_uid_button = widgets.Button(description="Get Next Unprocessed RTS", icon='forward', layout=widgets.Layout(width='250px'))
uid_display = widgets.HTML(value="<h3>Current UID: None</h3>")
prompt_toggle = widgets.ToggleButtons(options=['Positive', 'Negative', 'Box'], description='Prompt Type:', button_style='info')
run_sam_button = widgets.Button(description="Generate Mask", icon='magic', button_style='primary')
clear_prompts_button = widgets.Button(description="Clear All Prompts", icon='trash')
toggle_mask_visibility = widgets.Checkbox(value=True, description='Show Mask', indent=False)
save_button = widgets.Button(description="Save Final Mask", icon='save', button_style='success')
reject_bad_image_button = widgets.Button(description="Reject (Bad Image)", icon='times-circle', button_style='danger')
reject_no_feature_button = widgets.Button(description="Reject (No Feature)", icon='ban', button_style='warning')

# ### --- Section 6: UI Layout --- ###
controls_col1 = widgets.VBox([uid_display, prompt_toggle])
controls_col2 = widgets.VBox([run_sam_button, clear_prompts_button, toggle_mask_visibility])
prompting_controls = widgets.HBox([controls_col1, controls_col2])
finalization_controls = widgets.HBox([save_button, reject_bad_image_button, reject_no_feature_button])
main_controls = widgets.VBox([next_uid_button, prompting_controls, finalization_controls])
login_ui = widgets.VBox([worker_id_input, start_button])
main_app_ui = widgets.VBox([main_controls])

# ### --- Section 7: Application Logic (Callback Functions) --- ###

def on_mouse_down(x, y):
    """Callback function for when the user clicks on the ipycanvas."""
    ix, iy = int(x), int(y)
    prompt_mode = prompt_toggle.value

    if prompt_mode == 'Box':
        APP_STATE["box_prompts"].append((ix, iy))
        print(f"Added Box point at ({ix}, {iy}).")
        if len(APP_STATE["box_prompts"]) == 2:
            p1, p2 = APP_STATE["box_prompts"]
            x_min, y_min = min(p1[0], p2[0]), min(p1[1], p2[1])
            x_max, y_max = max(p1[0], p2[0]), max(p1[1], p2[1])
            APP_STATE["final_boxes"].append([x_min, y_min, x_max, y_max])
            APP_STATE["box_prompts"] = []
            print(f"Completed box. Ready for next box.")

    elif prompt_mode in ['Positive', 'Negative']:
        label = 1 if prompt_mode == 'Positive' else 0
        APP_STATE["prompts"].append({'coords': (ix, iy), 'label': label})
        print(f"Added {prompt_mode} prompt at pixel ({ix}, {iy})")

    redraw_canvas(image_to_show=APP_STATE["current_display_image"], mask=APP_STATE["current_mask"])

def redraw_canvas(image_to_show=None, mask=None):
    """Clears and redraws the ipycanvas with the image, prompts, and mask."""
    canvas = APP_STATE.get("canvas")
    if not canvas: return
    with hold_canvas(canvas):
        canvas.clear()
        if image_to_show is not None:
            canvas.put_image_data(image_to_show, 0, 0)

            if mask is not None and APP_STATE.get("mask_visible", True):
                mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                mask_rgba[mask == 1] = [255, 0, 0, 100]
                canvas.put_image_data(mask_rgba, 0, 0)

            for prompt in APP_STATE["prompts"]:
                x, y = prompt['coords']
                canvas.fill_style = 'lime' if prompt['label'] == 1 else 'red'
                canvas.fill_rect(x - 3, y - 3, 7, 7)

            canvas.stroke_style = 'blue'
            canvas.line_width = 2
            for box in APP_STATE["final_boxes"]:
                x1, y1, x2, y2 = box
                canvas.stroke_rect(x1, y1, x2 - x1, y2 - y1)

            if len(APP_STATE["box_prompts"]) == 1:
                x, y = APP_STATE["box_prompts"][0]
                canvas.fill_style = 'blue'
                canvas.fill_rect(x - 3, y - 3, 7, 7)

def on_run_sam_button_clicked(b):
    """Runs SAM inference with the current set of prompts."""
    points = [p['coords'] for p in APP_STATE["prompts"]] if APP_STATE["prompts"] else None
    labels = [p['label'] for p in APP_STATE["prompts"]] if APP_STATE["prompts"] else None
    boxes = APP_STATE["final_boxes"] if APP_STATE["final_boxes"] else None

    if not points and not boxes:
        print("Please add at least one point or a box prompt.")
        return

    print("Generating mask...")
    mask_array = APP_STATE["sam_model"].run_inference(
        APP_STATE["current_display_image"][:,:,:3],
        points=points,
        labels=labels,
        boxes=boxes
    )
    APP_STATE["current_mask"] = mask_array
    redraw_canvas(image_to_show=APP_STATE["current_display_image"], mask=mask_array)
    print("Mask generated and added to plot.")

def on_clear_prompts_button_clicked(b):
    """Clears all prompts and the current mask from the state and canvas."""
    APP_STATE["prompts"] = []
    APP_STATE["box_prompts"] = []
    APP_STATE["final_boxes"] = []
    APP_STATE["current_mask"] = None
    if APP_STATE.get("current_display_image") is not None:
        redraw_canvas(image_to_show=APP_STATE["current_display_image"])
    print("All prompts and masks have been cleared.")

def on_toggle_mask_visibility_changed(change):
    """Callback for when the show/hide mask checkbox is changed."""
    APP_STATE["mask_visible"] = change.new
    redraw_canvas(
        image_to_show=APP_STATE.get("current_display_image"),
        mask=APP_STATE.get("current_mask")
    )

def finalize_labeling(status: str):
    """Finalizes the current UID with a given status, saves files, and loads the next UID."""
    uid = APP_STATE.get("current_uid")
    if not uid: return
    print(f"Finalizing UID {uid} with status: {status}")

    if status == 'completed':
        if APP_STATE.get("current_mask") is None:
            print("Error: Cannot save. Please generate a mask first.", file=sys.stderr)
            return
        r, g, b = APP_STATE["current_image_array"][:3]
        mask = APP_STATE["current_mask"]
        output_array = np.stack([r, g, b, mask])
        profile = APP_STATE["current_tile_profile"]
        profile.update(count=4, dtype=rasterio.uint8)
        output_filename = f"{uid}_mask.tif"
        output_path = OUTPUT_DIR / output_filename
        print(f"Saving 4-channel GeoTIFF to: {output_path}")
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(output_array.astype(rasterio.uint8))
        APP_STATE["manifest_df"].loc[APP_STATE["manifest_df"]['uid'] == uid, 'output_filename'] = output_filename
    else:
        APP_STATE["manifest_df"].loc[APP_STATE["manifest_df"]['uid'] == uid, 'output_filename'] = ''

    APP_STATE["manifest_df"].loc[APP_STATE["manifest_df"]['uid'] == uid, 'labeling_status'] = status
    APP_STATE["manifest_df"].loc[APP_STATE["manifest_df"]['uid'] == uid, 'end_time_utc'] = datetime.datetime.utcnow().isoformat()
    APP_STATE["manifest_df"].to_csv(MANIFEST_PATH, index=False)
    print("Manifest updated. Loading next UID...")
    load_next_uid()

def load_next_uid():
    """Loads the next unprocessed UID from the manifest and displays the corresponding image."""
    # Check for fatal initialization error before proceeding
    if APP_STATE.get("arts_gdf") is None or APP_STATE.get("correspondence_gdf") is None:
        print(" Cannot load next UID. Application did not initialize correctly due to missing data files.", file=sys.stderr)
        return
        
    on_clear_prompts_button_clicked(None)
    manifest_df = APP_STATE["manifest_df"]
    worker_id = APP_STATE["worker_id"]
    
    # First, check if this worker has an 'in_progress' item
    worker_in_progress = manifest_df[(manifest_df['labeling_status'] == 'in_progress') & (manifest_df['worker_id'] == worker_id)]
    
    next_uid = None
    if not worker_in_progress.empty:
        next_uid = worker_in_progress.iloc[0]['uid']
    else:
        unprocessed_series = manifest_df[manifest_df['labeling_status'] == 'unprocessed']
        if not unprocessed_series.empty:
            next_uid = unprocessed_series.iloc[0]['uid']
        else:
            uid_display.value = "<h3>All UIDs Processed!</h3>"; redraw_canvas(); return

    feature_polygons = APP_STATE["arts_gdf"][APP_STATE["arts_gdf"]['UID'] == next_uid]
    if feature_polygons.empty:
        finalize_labeling('rejected_not_in_arts'); return

    true_centroid = feature_polygons.geometry.union_all().centroid
    image_info_list = get_image_info_for_uid(next_uid, APP_STATE["correspondence_gdf"])
    if not image_info_list: return

    image_info = image_info_list[0]
    centroid_gs = gpd.GeoSeries([true_centroid], crs=APP_STATE["arts_gdf"].crs)
    buffer_distance = 384
    aoi_gdf = gpd.GeoDataFrame(geometry=centroid_gs.buffer(buffer_distance))

    gcs_paths = APP_STATE["gcs_loader"].find_image_paths([image_info])
    if not gcs_paths:
        print(f"Could not find GCS path for UID {next_uid}. Skipping.")
        finalize_labeling('rejected_no_image_path'); return

    tile_data = APP_STATE["gcs_loader"].get_tile_from_paths(gcs_paths, aoi_gdf)

    if not tile_data:
        finalize_labeling('rejected_bad_imagery'); return

    APP_STATE["current_uid"] = next_uid
    uid_display.value = f"<h3>Current UID: {next_uid}</h3>"
    manifest_df.loc[manifest_df['uid'] == next_uid, 'labeling_status'] = 'in_progress'
    manifest_df.loc[manifest_df['uid'] == next_uid, 'worker_id'] = worker_id
    manifest_df.loc[manifest_df['uid'] == next_uid, 'start_time_utc'] = datetime.datetime.utcnow().isoformat()
    # Save manifest immediately after claiming a UID
    manifest_df.to_csv(MANIFEST_PATH, index=False)

    image_array, profile = tile_data
    APP_STATE["current_image_array"] = image_array
    APP_STATE["current_tile_profile"] = profile

    rgb_array = image_array[:3]
    if np.max(rgb_array) > 0:
        p2, p98 = np.percentile(rgb_array[rgb_array > 0], (2, 98)) # Avoid including no-data values in percentile
        rgb_stretched = np.clip((rgb_array - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    else:
        rgb_stretched = np.zeros_like(rgb_array, dtype=np.uint8)

    rgb_display = np.transpose(rgb_stretched, (1, 2, 0))
    rgba_display = np.dstack((rgb_display, np.full((rgb_display.shape[0], rgb_display.shape[1]), 255, dtype=np.uint8)))

    APP_STATE["current_display_image"] = rgba_display
    APP_STATE["canvas"].width = rgba_display.shape[1]
    APP_STATE["canvas"].height = rgba_display.shape[0]
    redraw_canvas(image_to_show=rgba_display)

def initialize_app():
    """Initializes all backend components of the application."""
    print("--- Initializing Application... ---")
    APP_STATE["worker_id"] = worker_id_input.value

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory set to: {OUTPUT_DIR}")

    if not MANIFEST_PATH.exists():
        print(f"Manifest not found. Copying from repository...")
        repo_manifest = project_root / "manifest.csv"
        if repo_manifest.exists():
            import shutil
            shutil.copy(repo_manifest, MANIFEST_PATH)
            print("Manifest copied successfully.")
        else:
            print(f" FATAL: Could not find manifest.csv in the repository at {repo_manifest}", file=sys.stderr)
            return

    print(f"Loading manifest from: {MANIFEST_PATH}")
    APP_STATE["manifest_df"] = pd.read_csv(MANIFEST_PATH)
    print("Manifest loaded.")

    # --- Robust Data Loading ---
    correspondence_path = project_root / "data" / "raw" / "planet_basemaps_rts_polygon_basemap_correspondence.geojson"
    APP_STATE["correspondence_gdf"] = load_correspondence_data(correspondence_path)
    if APP_STATE["correspondence_gdf"] is None:
        print(f" FATAL: Failed to load correspondence data from {correspondence_path}. Cannot continue.", file=sys.stderr)
        return # Stop initialization

    arts_path = project_root / "data" / "raw" / "ARTS_main_dataset_v.3.1.0.geojson"
    APP_STATE["arts_gdf"] = load_and_filter_arts_data(arts_path)
    if APP_STATE["arts_gdf"] is None:
        print(f" FATAL: Failed to load ARTS data from {arts_path}. Cannot continue.", file=sys.stderr)
        return # Stop initialization

    APP_STATE["gcs_loader"] = GCSImageLoader(project_id="abruptthawmapping", bucket_name="abrupt_thaw", search_prefix="planet_basemaps/global_quarterly_COGs")
    APP_STATE["sam_model"] = SAM2Model(model_name="facebook/sam-vit-base")

    print(f"\\n Initialization complete. Welcome, {APP_STATE['worker_id']}!")
    print("Click 'Get Next Unprocessed RTS' to begin.")

def on_start_button_clicked(b):
    """Callback for the main start button. Clears output and displays the main app."""
    clear_output()
    
    canvas = Canvas(width=800, height=600)
    canvas.on_mouse_down(on_mouse_down)
    APP_STATE["canvas"] = canvas
    
    display(main_app_ui)
    display(canvas)
    
    initialize_app()


# ### --- Section 8: Main Execution Block --- ###
def main():
    """Main function to wire up UI components and display the login screen."""
    start_button.on_click(on_start_button_clicked)
    next_uid_button.on_click(lambda b: load_next_uid())
    run_sam_button.on_click(on_run_sam_button_clicked)
    clear_prompts_button.on_click(on_clear_prompts_button_clicked)
    toggle_mask_visibility.observe(on_toggle_mask_visibility_changed, names='value')
    save_button.on_click(lambda b: finalize_labeling('completed'))
    reject_bad_image_button.on_click(lambda b: finalize_labeling('rejected_bad_imagery'))
    reject_no_feature_button.on_click(lambda b: finalize_labeling('rejected_no_feature'))

    print("\\n--- Displaying User Interface ---")
    display(login_ui)

if __name__ == "__main__":
    main()
