import comfy.utils
from comfy.cldm.control_types import UNION_CONTROLNET_TYPES
from comfy_api.v3 import io


class ControlNetApplyAdvanced_V3(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="ControlNetApplyAdvanced_V3",
            display_name="Apply ControlNet _V3",
            category="conditioning/controlnet",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.ControlNet.Input("control_net"),
                io.Image.Input("image"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001),
                io.Vae.Input("vae", optional=True),
            ],
            outputs=[
                io.Conditioning.Output("positive_out", display_name="positive"),
                io.Conditioning.Output("negative_out", display_name="negative"),
            ],
        )

    @classmethod
    def execute(
        cls, positive, negative, control_net, image, strength, start_percent, end_percent, vae=None, extra_concat=[]
    ) -> io.NodeOutput:
        if strength == 0:
            return io.NodeOutput(positive, negative)

        control_hint = image.movedim(-1, 1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get("control", None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(
                        control_hint, strength, (start_percent, end_percent), vae=vae, extra_concat=extra_concat
                    )
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d["control"] = c_net
                d["control_apply_to_uncond"] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return io.NodeOutput(out[0], out[1])


class SetUnionControlNetType_V3(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="SetUnionControlNetType_V3",
            category="conditioning/controlnet",
            inputs=[
                io.ControlNet.Input("control_net"),
                io.Combo.Input("type", options=["auto"] + list(UNION_CONTROLNET_TYPES.keys())),
            ],
            outputs=[
                io.ControlNet.Output("control_net_out"),
            ],
        )

    @classmethod
    def execute(cls, control_net, type) -> io.NodeOutput:
        control_net = control_net.copy()
        type_number = UNION_CONTROLNET_TYPES.get(type, -1)
        if type_number >= 0:
            control_net.set_extra_arg("control_type", [type_number])
        else:
            control_net.set_extra_arg("control_type", [])

        return io.NodeOutput(control_net)


class ControlNetInpaintingAliMamaApply_V3(ControlNetApplyAdvanced_V3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="ControlNetInpaintingAliMamaApply_V3",
            category="conditioning/controlnet",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.ControlNet.Input("control_net"),
                io.Vae.Input("vae"),
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001),
            ],
            outputs=[
                io.Conditioning.Output("positive_out", display_name="positive"),
                io.Conditioning.Output("negative_out", display_name="negative"),
            ],
        )

    @classmethod
    def execute(
        cls, positive, negative, control_net, vae, image, mask, strength, start_percent, end_percent
    ) -> io.NodeOutput:
        extra_concat = []
        if control_net.concat_mask:
            mask = 1.0 - mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            mask_apply = comfy.utils.common_upscale(mask, image.shape[2], image.shape[1], "bilinear", "center").round()
            image = image * mask_apply.movedim(1, -1).repeat(1, 1, 1, image.shape[3])
            extra_concat = [mask]

        return super().execute(
            positive,
            negative,
            control_net,
            image,
            strength,
            start_percent,
            end_percent,
            vae=vae,
            extra_concat=extra_concat,
        )


NODES_LIST: list[type[io.ComfyNodeV3]] = [
    ControlNetApplyAdvanced_V3,
    SetUnionControlNetType_V3,
    ControlNetInpaintingAliMamaApply_V3,
]
