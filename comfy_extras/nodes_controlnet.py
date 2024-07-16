
UNION_CONTROLNET_TYPES = {"auto": -1,
                          "openpose": 0,
                          "depth": 1,
                          "hed/pidi/scribble/ted": 2,
                          "canny/lineart/anime_lineart/mlsd": 3,
                          "normal": 4,
                          "segment": 5,
                          "tile": 6,
                          "repaint": 7,
                        }

class SetUnionControlNetType:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"control_net": ("CONTROL_NET", ),
                             "type": (list(UNION_CONTROLNET_TYPES.keys()),)
                             }}

    CATEGORY = "conditioning/controlnet"
    RETURN_TYPES = ("CONTROL_NET",)

    FUNCTION = "set_controlnet_type"

    def set_controlnet_type(self, control_net, type):
        control_net = control_net.copy()
        type_number = UNION_CONTROLNET_TYPES[type]
        if type_number >= 0:
            control_net.set_extra_arg("control_type", [type_number])
        else:
            control_net.set_extra_arg("control_type", [])

        return (control_net,)

NODE_CLASS_MAPPINGS = {
    "SetUnionControlNetType": SetUnionControlNetType,
}
