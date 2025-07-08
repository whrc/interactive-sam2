# tests/test_sam_model.py

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add the project root to the system path to allow imports from src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.interactive_label_sam2.model import SAM2Model

def run_model_test():
    """
    A test function to verify that the SAM2Model class can be initialized
    and can run inference on a sample image.
    """
    print("--- Starting SAM Model Test ---")

    try:
        # --- 1. Define Paths ---
        # The input tile is now read from the test outputs directory
        output_dir = project_root / "tests" / "outputs"
        image_path = output_dir / "test_tile.png"
        # Create the output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- 2. Load the sample image tile ---
        if not image_path.exists():
            print(f"Error: Test image '{image_path.name}' not found in '{output_dir}'.", file=sys.stderr)
            print("Please run 'tests/test_gcs_access.py' first to generate it.", file=sys.stderr)
            return

        print(f"1. Loading test image from: {image_path}")
        # Open the image and convert to RGB format
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)

        # --- 3. Initialize the SAM Model ---
        print("\n2. Initializing SAM Model...")
        sam_model = SAM2Model(model_name="facebook/sam-vit-base")

        # --- 4. Define Dummy Prompts ---
        height, width, _ = image_array.shape
        points = [
            (width // 2, height // 2),        # Positive prompt in the center
            (width // 4, height // 4)         # Negative prompt in the top-left
        ]
        labels = [1, 0] # 1 for positive, 0 for negative
        print(f"\n3. Using {len(points)} test prompts.")

        # --- 5. Run Inference ---
        print("\n4. Running model inference...")
        mask_array = sam_model.run_inference(image_array, points, labels)
        print(f"   - Inference complete. Mask shape: {mask_array.shape}")

        # --- 6. Save the Output Mask ---
        output_path = output_dir / "test_mask.png"
        print(f"\n5. Saving output mask to: {output_path}")
        
        mask_image = Image.fromarray(mask_array * 255)
        mask_image.save(output_path)
        print("   - Mask saved successfully.")

        # --- 7. Display and Save the results for visual confirmation ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_array)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(image_array)
        axes[1].imshow(mask_array, cmap='jet', alpha=0.6)
        axes[1].set_title("Mask Overlay")
        axes[1].axis('off')

        axes[2].imshow(mask_array, cmap='gray')
        axes[2].set_title("Generated Mask")
        axes[2].axis('off')
        
        overlay_path = output_dir / "test_overlay.png"
        fig.savefig(overlay_path, bbox_inches='tight')
        plt.close(fig)
        print(f"   - Overlay visualization saved to: {overlay_path}")


        print("\n--- SAM Model Test Finished Successfully! ---")

    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred During Model Test ---")
        print(f"Error: {e}")

if __name__ == "__main__":
    run_model_test()
