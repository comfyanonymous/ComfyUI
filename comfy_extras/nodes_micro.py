import nodes

from comfy.micro import codec
from comfy.micro.runtime import call_micro_node
from comfy.micro.server import register_micro_nodes, register_routes
from comfy.micro.wire import BytesPayload, MicroValue


class ToMicro_IMAGE:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("Micro_IMAGE",)
    RETURN_NAMES = ("value",)
    FUNCTION = "to_micro"
    CATEGORY = "micro"

    def to_micro(self, image):
        return (MicroValue("IMAGE", BytesPayload(codec.encode_image(image))),)


class FromMicro_IMAGE:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("Micro_IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "from_micro"
    CATEGORY = "micro"

    def from_micro(self, value):
        if value.type_name != "IMAGE":
            raise ValueError(f"expected MicroValue type IMAGE, got {value.type_name!r}")
        return (codec.decode_image(value.payload.as_bytes()),)


class ToMicro_MASK:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",)}}

    RETURN_TYPES = ("Micro_MASK",)
    RETURN_NAMES = ("value",)
    FUNCTION = "to_micro"
    CATEGORY = "micro"

    def to_micro(self, mask):
        return (MicroValue("MASK", BytesPayload(codec.encode_mask(mask))),)


class FromMicro_MASK:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("Micro_MASK",)}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "from_micro"
    CATEGORY = "micro"

    def from_micro(self, value):
        if value.type_name != "MASK":
            raise ValueError(f"expected MicroValue type MASK, got {value.type_name!r}")
        return (codec.decode_mask(value.payload.as_bytes()),)


class Micro_ScaleImage:
    @classmethod
    def INPUT_TYPES(cls):
        required = dict(nodes.ImageScale.INPUT_TYPES()["required"])
        required["image"] = ("Micro_IMAGE",)
        return {"required": required}

    RETURN_TYPES = ("Micro_IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "scale"
    CATEGORY = "micro/image"

    def scale(self, image, upscale_method, width, height, crop):
        result = call_micro_node("Micro_ScaleImage", {
            "image": image,
            "upscale_method": upscale_method,
            "width": width,
            "height": height,
            "crop": crop,
        })
        return (result["image"],)


MICRO_TO_NATIVE = {
    "Micro_ScaleImage": "ImageScale",
}

NODE_CLASS_MAPPINGS = {
    "ToMicro_IMAGE": ToMicro_IMAGE,
    "FromMicro_IMAGE": FromMicro_IMAGE,
    "ToMicro_MASK": ToMicro_MASK,
    "FromMicro_MASK": FromMicro_MASK,
    "Micro_ScaleImage": Micro_ScaleImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToMicro_IMAGE": "To Micro Image",
    "FromMicro_IMAGE": "From Micro Image",
    "ToMicro_MASK": "To Micro Mask",
    "FromMicro_MASK": "From Micro Mask",
    "Micro_ScaleImage": "Micro Scale Image",
}

register_micro_nodes(MICRO_TO_NATIVE)
register_routes()
