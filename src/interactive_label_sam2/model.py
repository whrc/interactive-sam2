# src/interactive_label_sam2/model.py

import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import numpy as np
from typing import List, Tuple

class SAM2Model:
    """
    A class to encapsulate the SAM2 model loading and inference logic.
    """
    def __init__(self, model_name: str = "facebook/sam2-vision-l"):
        """
        Initializes the model and processor.

        Args:
            model_name (str): The name of the pre-trained SAM2 model from Hugging Face.
        """
        try:
            print("--- Initializing SAM2 Model ---")
            # Check if a GPU is available and set the device accordingly.
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")

            # Load the model and processor from Hugging Face.
            # This will download the model weights the first time it is run.
            print(f"Loading model: {model_name}...")
            self.model = SamModel.from_pretrained(model_name).to(self.device)
            self.processor = SamProcessor.from_pretrained(model_name)
            print("--- SAM2 Model Initialized Successfully ---\n")

        except Exception as e:
            print(f"FATAL: Failed to initialize SAM2 Model. Error: {e}")
            raise

    def run_inference(self, image_array: np.ndarray, points: List[Tuple[int, int]], labels: List[int]) -> np.ndarray:
        """
        Runs inference on a single image with a set of point prompts.

        Args:
            image_array (np.ndarray): The input image as a NumPy array (H, W, C).
            points (List[Tuple[int, int]]): A list of (x, y) coordinates for the prompts.
            labels (List[int]): A list of labels for each point (1 for positive, 0 for negative).

        Returns:
            np.ndarray: A 2D NumPy array representing the segmentation mask.
        """
        # Convert the NumPy array to a PIL Image, as the processor expects this format.
        image_pil = Image.fromarray(image_array)

        # The processor requires the points in a specific nested list format.
        # e.g., [[[x1, y1], [x2, y2]]]
        input_points = [points]

        # The processor also requires the labels in a nested list format.
        # e.g., [[1, 0]]
        input_labels = [labels]

        # Process the image and prompts.
        inputs = self.processor(
            image_pil, 
            input_points=input_points, 
            input_labels=input_labels, 
            return_tensors="pt"
        ).to(self.device)

        # Run the model in inference mode (no gradient calculations).
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get the segmentation mask from the model output.
        # The processor can convert the output to the original image size.
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs["original_sizes"].cpu(), 
            inputs["reshaped_input_sizes"].cpu()
        )
        
        # The output is typically a batch of masks, we take the first one.
        # The mask is a boolean tensor, we convert it to a NumPy array of integers (0 or 1).
        mask = masks[0][0][0].numpy().astype(np.uint8)

        return mask

