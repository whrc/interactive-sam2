# src/interactive_label_sam2/model.py

import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional

class SAM2Model:
    """
    A class to encapsulate the SAM model loading and inference logic.
    """
    def __init__(self, model_name: str = "facebook/sam-vit-base"):
        """
        Initializes the model and processor.

        Args:
            model_name (str): The name of the pre-trained SAM model from Hugging Face.
        """
        try:
            print("--- Initializing SAM Model ---")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")

            print(f"Loading model: {model_name}...")
            self.model = SamModel.from_pretrained(model_name).to(self.device)
            self.processor = SamProcessor.from_pretrained(model_name)
            print("--- SAM Model Initialized Successfully ---\n")

        except Exception as e:
            print(f"FATAL: Failed to initialize SAM Model. Error: {e}")
            raise

    def run_inference(self,
                      image_array: np.ndarray,
                      points: Optional[List[Tuple[int, int]]] = None,
                      labels: Optional[List[int]] = None,
                      box: Optional[List[int]] = None) -> np.ndarray:
        """
        Runs inference on a single image with a set of point and/or box prompts.

        Args:
            image_array (np.ndarray): The input image as a NumPy array (H, W, C).
            points (Optional): A list of (x, y) coordinates for the point prompts.
            labels (Optional): A list of labels for each point (1 for positive, 0 for negative).
            box (Optional): A list representing the bounding box [x_min, y_min, x_max, y_max].

        Returns:
            np.ndarray: A 2D NumPy array representing the segmentation mask.
        """
        image_pil = Image.fromarray(image_array)

        # The processor expects inputs in a specific nested list format.
        input_points = [points] if points else None
        input_labels = [labels] if labels else None
        input_boxes = [[box]] if box else None

        # Process the image and prompts.
        inputs = self.processor(
            image_pil,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            return_tensors="pt"
        ).to(self.device)

        # Run the model in inference mode.
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get the segmentation mask from the model output.
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )

        mask = masks[0][0][0].numpy().astype(np.uint8)

        return mask
