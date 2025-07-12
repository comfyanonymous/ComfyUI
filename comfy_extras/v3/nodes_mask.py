from comfy_api.v3 import io, ui


class MaskPreview_V3(io.ComfyNodeV3):
    """Mask Preview - original implement in ComfyUI_essentials.

    https://github.com/cubiq/ComfyUI_essentials/blob/9d9f4bedfc9f0321c19faf71855e228c93bd0dc9/mask.py#L81
    Upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes
    """

    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="MaskPreview_V3",
            display_name="Preview Mask _V3",
            category="mask",
            inputs=[
                io.Mask.Input("masks"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, masks):
        return io.NodeOutput(ui=ui.PreviewMask(masks))


NODES_LIST: list[type[io.ComfyNodeV3]] = [MaskPreview_V3]
