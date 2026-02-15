from comfy_api.latest import ComfyExtension, io, node_replace


async def register_replacements():
    """Register all built-in node replacements."""
    register_replacements_longeredge()
    register_replacements_batchimages()
    register_replacements_upscaleimage()
    register_replacements_controlnet()
    register_replacements_load3d()
    register_replacements_preview3d()
    register_replacements_svdimg2vid()
    register_replacements_conditioningavg()

def register_replacements_longeredge():
    # No dynamic inputs here
    node_replace.register(node_replace.NodeReplace(
            new_node_id="ImageScaleToMaxDimension",
            old_node_id="ResizeImagesByLongerEdge",
            old_widget_ids=["longer_edge"],
            input_mapping=[
                {"new_id": "image", "old_id": "images"},
                {"new_id": "largest_size", "old_id": "longer_edge"},
                {"new_id": "upscale_method", "set_value": "lanczos"},
            ],
            # just to test the frontend output_mapping code, does nothing really here
            output_mapping=[{"new_idx": 0, "old_idx": 0}],
        ))

def register_replacements_batchimages():
    # BatchImages node uses Autogrow
    node_replace.register(node_replace.NodeReplace(
            new_node_id="BatchImagesNode",
            old_node_id="ImageBatch",
            input_mapping=[
                {"new_id": "images.image0", "old_id": "image1"},
                {"new_id": "images.image1", "old_id": "image2"},
            ],
        ))

def register_replacements_upscaleimage():
    # ResizeImageMaskNode uses DynamicCombo
    node_replace.register(node_replace.NodeReplace(
            new_node_id="ResizeImageMaskNode",
            old_node_id="ImageScaleBy",
            old_widget_ids=["upscale_method", "scale_by"],
            input_mapping=[
                {"new_id": "input", "old_id": "image"},
                {"new_id": "resize_type", "set_value": "scale by multiplier"},
                {"new_id": "resize_type.multiplier", "old_id": "scale_by"},
                {"new_id": "scale_method", "old_id": "upscale_method"},
            ],
        ))

def register_replacements_controlnet():
    # T2IAdapterLoader → ControlNetLoader
    node_replace.register(node_replace.NodeReplace(
            new_node_id="ControlNetLoader",
            old_node_id="T2IAdapterLoader",
            input_mapping=[
                {"new_id": "control_net_name", "old_id": "t2i_adapter_name"},
            ],
        ))

def register_replacements_load3d():
    # Load3DAnimation merged into Load3D
    node_replace.register(node_replace.NodeReplace(
            new_node_id="Load3D",
            old_node_id="Load3DAnimation",
        ))

def register_replacements_preview3d():
    # Preview3DAnimation merged into Preview3D
    node_replace.register(node_replace.NodeReplace(
            new_node_id="Preview3D",
            old_node_id="Preview3DAnimation",
        ))

def register_replacements_svdimg2vid():
    # Typo fix: SDV → SVD
    node_replace.register(node_replace.NodeReplace(
            new_node_id="SVD_img2vid_Conditioning",
            old_node_id="SDV_img2vid_Conditioning",
        ))

def register_replacements_conditioningavg():
    # Typo fix: trailing space in node name
    node_replace.register(node_replace.NodeReplace(
            new_node_id="ConditioningAverage",
            old_node_id="ConditioningAverage ",
        ))

class NodeReplacementsExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return []

async def comfy_entrypoint() -> NodeReplacementsExtension:
    await register_replacements()
    return NodeReplacementsExtension()
