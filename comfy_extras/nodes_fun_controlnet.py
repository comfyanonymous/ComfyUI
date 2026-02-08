# Fun ControlNet Custom Nodes for ComfyUI
# Provides the PrepareFunControlNet node to copy weights from main model

import torch
import copy


class PrepareFunControlNet:
    """
    Prepares a Fun ControlNet by copying projection weights from the main Qwen model.
    
    This node is required because the Fun ControlNet "Lite" architecture does not include
    its own input projection layers (img_in, txt_in, txt_norm). Instead, it relies on
    borrowing these weights from the main model.
    
    Connect this node between your Model and the standard ControlNetApply node.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "controlnet": ("CONTROL_NET",),
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)
    RETURN_NAMES = ("controlnet",)
    FUNCTION = "prepare"
    CATEGORY = "conditioning/controlnet"
    DESCRIPTION = "Prepares a Fun ControlNet by copying projection weights from the main Qwen model."

    def prepare(self, model, controlnet):
        """
        Copy img_in, txt_in, and txt_norm weights from the main model to the ControlNet.
        """
        # Get the actual diffusion model from the ModelPatcher
        if hasattr(model, 'model'):
            main_model = model.model
            # Unwrap if needed
            if hasattr(main_model, 'diffusion_model'):
                main_model = main_model.diffusion_model
        else:
            raise ValueError("Could not access diffusion model from the provided model input.")

        # Check if this is a Fun ControlNet
        control_model = controlnet.control_model
        if not hasattr(control_model, 'borrowed_img_in'):
            raise ValueError(
                "The provided ControlNet is not a Fun ControlNet. "
                "This node only works with Qwen Fun ControlNet models."
            )

        # Check if main model has the required layers
        if not hasattr(main_model, 'img_in') or not hasattr(main_model, 'txt_in') or not hasattr(main_model, 'txt_norm'):
            raise ValueError(
                "The main model does not have the required projection layers (img_in, txt_in, txt_norm). "
                "Make sure you are using a Qwen image model."
            )

        # Create a copy of the controlnet to avoid modifying the original
        # (important if user wants to use the same ControlNet with multiple models)
        controlnet_copy = copy.copy(controlnet)
        
        # Copy the weight references (not the actual tensors, just the layer references)
        # This means the ControlNet will use the same layers as the main model
        control_model.borrowed_img_in = main_model.img_in
        control_model.borrowed_txt_in = main_model.txt_in
        control_model.borrowed_txt_norm = main_model.txt_norm

        print(f"[PrepareFunControlNet] Successfully copied projection weights from main model to ControlNet")

        return (controlnet_copy,)


# Register nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "PrepareFunControlNet": PrepareFunControlNet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrepareFunControlNet": "Prepare Fun ControlNet",
}
