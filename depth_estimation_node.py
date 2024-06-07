import os
import numpy as np
import torch
from transformers import pipeline
from PIL import Image, ImageFilter, ImageOps
from comfy.utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management

def ensure_odd(value):
    """Ensure the value is an odd integer."""
    value = int(value)
    return value if value % 2 == 1 else value + 1

def gamma_correction(img, gamma=1.0):
    """Apply gamma correction to the image."""
    inv_gamma = 1.0 / gamma
    table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return Image.fromarray(np.array(img).astype(np.uint8)).point(lambda i: table[i])

def auto_contrast(image):
    """Apply automatic contrast adjustment to the image."""
    return ImageOps.autocontrast(image)

class ComfyUIDepthEstimationNode:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            blur_radius=("FLOAT", {"default": 2.0}),
            median_size=("INT", {"default": 5}),
            device=(["cpu", "gpu"], {"default": "cpu"}),
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def __init__(self):
        pass

    def process_image(self, image, blur_radius, median_size, device):
        device = 0 if device == "gpu" and torch.cuda.is_available() else -1
        pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf", device=device)
        
        if device == 0:
            image = image.convert("RGB")  # Ensure image is in RGB format
            inputs = pipe.feature_extractor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = pipe.model(**inputs)
            result = pipe.post_process(outputs, (image.height, image.width))
        else:
            result = pipe(image)

        # Convert depth data to a NumPy array if not already one
        depth_data = np.array(result["depth"])

        # Normalize and convert to uint8
        depth_normalized = (depth_data - depth_data.min()) / (depth_data.max() - depth_data.min() + 1e-8)  # Avoid zero division
        depth_uint8 = (255 * depth_normalized).astype(np.uint8)

        # Create an image from the processed depth data
        depth_image = Image.fromarray(depth_uint8)

        # Apply a median filter to reduce noise
        depth_image = depth_image.filter(ImageFilter.MedianFilter(size=ensure_odd(median_size)))

        # Enhanced edge detection with more feathering
        edges = depth_image.filter(ImageFilter.FIND_EDGES)
        edges = edges.filter(ImageFilter.GaussianBlur(radius=2 * blur_radius))
        edges = edges.point(lambda x: 255 if x > 20 else 0)  # Adjusted threshold

        # Create a mask from the edges
        mask = edges.convert("L")

        # Blur only the edges using the mask
        blurred_edges = depth_image.filter(ImageFilter.GaussianBlur(radius=blur_radius * 2))

        # Combine the blurred edges with the original depth image using the mask
        combined_image = Image.composite(blurred_edges, depth_image, mask)

        # Apply auto gamma correction with a lower gamma to darken the image
        gamma_corrected_image = gamma_correction(combined_image, gamma=0.7)

        # Apply auto contrast
        final_image = auto_contrast(gamma_corrected_image)

        # Additional post-processing: Sharpen the final image
        final_image = final_image.filter(ImageFilter.SHARPEN)

        return final_image

    def execute(self, image, blur_radius, median_size, device):
        final_image = self.process_image(image, blur_radius, median_size, device)
        return (final_image, )

NODE_CLASS_MAPPINGS = {
    "ComfyUIDepthEstimationNode": ComfyUIDepthEstimationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUIDepthEstimationNode": "ComfyUI Depth Estimation",
}
