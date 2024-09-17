from comfy.cldm.control_types import UNION_CONTROLNET_TYPES
import nodes
import comfy.utils

class SetUnionControlNetType:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"control_net": ("CONTROL_NET", ),
                             "type": (["auto"] + list(UNION_CONTROLNET_TYPES.keys()),)
                             }}

    CATEGORY = "conditioning/controlnet"
    RETURN_TYPES = ("CONTROL_NET",)

    FUNCTION = "set_controlnet_type"

    def set_controlnet_type(self, control_net, type):
        control_net = control_net.copy()
        type_number = UNION_CONTROLNET_TYPES.get(type, -1)
        if type_number >= 0:
            control_net.set_extra_arg("control_type", [type_number])
        else:
            control_net.set_extra_arg("control_type", [])

        return (control_net,)

class ControlNetInpaintingAliMamaApply(nodes.ControlNetApplyAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "control_net": ("CONTROL_NET", ),
                             "vae": ("VAE", ),
                             "image": ("IMAGE", ),
                             "mask": ("MASK", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
                             }}

    FUNCTION = "apply_inpaint_controlnet"

    CATEGORY = "conditioning/controlnet"

    def apply_inpaint_controlnet(self, positive, negative, control_net, vae, image, mask, strength, start_percent, end_percent):
        extra_concat = []
        if control_net.concat_mask:
            mask = 1.0 - mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            mask_apply = comfy.utils.common_upscale(mask, image.shape[2], image.shape[1], "bilinear", "center").round()
            image = image * mask_apply.movedim(1, -1).repeat(1, 1, 1, image.shape[3])
            extra_concat = [mask]

        return self.apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent, vae=vae, extra_concat=extra_concat)



NODE_CLASS_MAPPINGS = {
    "SetUnionControlNetType": SetUnionControlNetType,
    "ControlNetInpaintingAliMamaApply": ControlNetInpaintingAliMamaApply,
}
