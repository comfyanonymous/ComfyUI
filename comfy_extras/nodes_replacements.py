from comfy_api.latest import ComfyExtension, io, ComfyAPI

api = ComfyAPI()


async def register_replacements():
    """Register all built-in node replacements."""
    await register_replacements_longeredge()
    await register_replacements_batchimages()
    await register_replacements_upscaleimage()
    await register_replacements_controlnet()
    await register_replacements_load3d()
    await register_replacements_preview3d()
    await register_replacements_svdimg2vid()
    await register_replacements_conditioningavg()
    await register_replacements_nanobanana2()

async def register_replacements_longeredge():
    # No dynamic inputs here
    await api.node_replacement.register(io.NodeReplace(
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

async def register_replacements_batchimages():
    # BatchImages node uses Autogrow
    await api.node_replacement.register(io.NodeReplace(
            new_node_id="BatchImagesNode",
            old_node_id="ImageBatch",
            input_mapping=[
                {"new_id": "images.image0", "old_id": "image1"},
                {"new_id": "images.image1", "old_id": "image2"},
            ],
        ))

async def register_replacements_upscaleimage():
    # ResizeImageMaskNode uses DynamicCombo
    await api.node_replacement.register(io.NodeReplace(
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

async def register_replacements_controlnet():
    # T2IAdapterLoader → ControlNetLoader
    await api.node_replacement.register(io.NodeReplace(
            new_node_id="ControlNetLoader",
            old_node_id="T2IAdapterLoader",
            input_mapping=[
                {"new_id": "control_net_name", "old_id": "t2i_adapter_name"},
            ],
        ))

async def register_replacements_load3d():
    # Load3DAnimation merged into Load3D
    await api.node_replacement.register(io.NodeReplace(
            new_node_id="Load3D",
            old_node_id="Load3DAnimation",
        ))

async def register_replacements_preview3d():
    # Preview3DAnimation merged into Preview3D
    await api.node_replacement.register(io.NodeReplace(
            new_node_id="Preview3D",
            old_node_id="Preview3DAnimation",
        ))

async def register_replacements_svdimg2vid():
    # Typo fix: SDV → SVD
    await api.node_replacement.register(io.NodeReplace(
            new_node_id="SVD_img2vid_Conditioning",
            old_node_id="SDV_img2vid_Conditioning",
        ))

async def register_replacements_conditioningavg():
    # Typo fix: trailing space in node name
    await api.node_replacement.register(io.NodeReplace(
            new_node_id="ConditioningAverage",
            old_node_id="ConditioningAverage ",
        ))

async def register_replacements_nanobanana2():
    # GeminiNanoBanana2 replaced by GeminiNanoBanana2V2, which uses Autogrow for the images input.
    await api.node_replacement.register(io.NodeReplace(
            new_node_id="GeminiNanoBanana2V2",
            old_node_id="GeminiNanoBanana2",
            old_widget_ids=[
                "prompt",
                "model",
                "seed",
                "aspect_ratio",
                "resolution",
                "response_modalities",
                "thinking_level",
                "system_prompt",
            ],
            input_mapping=[
                {"new_id": "prompt", "old_id": "prompt"},
                {"new_id": "model", "old_id": "model"},
                {"new_id": "seed", "old_id": "seed"},
                {"new_id": "aspect_ratio", "old_id": "aspect_ratio"},
                {"new_id": "resolution", "old_id": "resolution"},
                {"new_id": "response_modalities", "old_id": "response_modalities"},
                {"new_id": "thinking_level", "old_id": "thinking_level"},
                {"new_id": "images.image_1", "old_id": "images"},
                {"new_id": "files", "old_id": "files"},
                {"new_id": "system_prompt", "old_id": "system_prompt"},
            ],
        ))

class NodeReplacementsExtension(ComfyExtension):
    async def on_load(self) -> None:
        await register_replacements()

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return []

async def comfy_entrypoint() -> NodeReplacementsExtension:
    return NodeReplacementsExtension()
