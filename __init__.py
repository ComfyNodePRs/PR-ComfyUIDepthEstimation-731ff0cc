from .comfyui_depth_estimation import ComfyUIDepthEstimationNode

NODE_CLASS_MAPPINGS = {
    "ComfyUIDepthEstimationNode": ComfyUIDepthEstimationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUIDepthEstimationNode": "ComfyUI Depth Estimation"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
