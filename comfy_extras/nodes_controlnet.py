from comfy.cldm.control_types import UNION_CONTROLNET_TYPES

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

NODE_CLASS_MAPPINGS = {
    "SetUnionControlNetType": SetUnionControlNetType,
}
