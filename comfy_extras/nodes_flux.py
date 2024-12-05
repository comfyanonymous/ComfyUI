import node_helpers
import torch

from comfy.ldm.flux.layers import SingleStreamBlock, DoubleStreamBlock
from comfy.model_patcher import ModelPatcher

class CLIPTextEncodeFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "clip_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "t5xxl": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning/flux"

    def encode(self, clip, clip_l, t5xxl, guidance):
        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        return (clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance}), )

class FluxGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING", ),
            "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "advanced/conditioning/flux"

    def append(self, conditioning, guidance):
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        return (c, )

class _ReduxAttnWrapper:
    def __init__(self, previous, token_counts, bias=0.0, is_first=False):
        self.previous = previous
        self.token_counts = token_counts
        self.bias = bias
        self.is_first = is_first

    def __call__(self, args, extra_args):
        # args: {"img": img, <"txt": txt>, "vec": vec, "pe": pe}
        if self.is_first:
            self.token_counts["img"] = args["img"].shape[1]

        # determine the total number of tokens in the mask, depending on whether we're wrapping a single block or a double one
        total_tokens = args["img"].shape[1]
        if "txt" in args:
            total_tokens += args["txt"].shape[1]
        # create the mask (or bias map)
        mask = extra_args.get("attn_mask", torch.zeros((total_tokens, total_tokens), device=args["img"].device, dtype=args["img"].dtype))
        # if this wrapper was called by another ReduxAttnWrapper, compute the range of tokens that correspond to our image
        redux_end = extra_args.get("redux_end", -self.token_counts["img"])
        redux_start = redux_end - self.token_counts["redux"]
        # modify the mask
        # first 256 tokens are the text prompt
        mask[:256, redux_start:redux_end] = self.bias
        # last 'img' tokens are the image being generated
        mask[-self.token_counts["img"]:, redux_start:redux_end] = self.bias
        match self.previous:
            case DoubleStreamBlock():
                x, c = self.previous(img=args["img"], txt=args["txt"],vec=args["vec"], pe=args["pe"], attn_mask=mask)
                return {"img": x, "txt": c}
            case SingleStreamBlock():
                x = self.previous(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=mask)
                return {"img": x}
            case _ReduxAttnWrapper():
                # pass along the mask, and tell the next redux what its part of the mask is
                extra_args["attn_mask"] = mask
                extra_args["redux_end"] = redux_start
                return self.previous(args, extra_args)
            case _:
                print(f"Can't wrap {repr(self.previous)} with mask.")
                return self.previous(args, extra_args)

class ReduxApplyWithAttnMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL", ),
            "conditioning": ("CONDITIONING", ),
            "style_model": ("STYLE_MODEL", ),
            "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
            "attn_bias": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }}
    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "apply_stylemodel"

    CATEGORY = "conditioning/style_model"

    def apply_stylemodel(self, model: ModelPatcher, clip_vision_output, style_model, conditioning, attn_bias):
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)

        c = []
        for t in conditioning:
            n = [torch.cat((t[0], cond), dim=1), t[1].copy()]
            c.append(n)

        if attn_bias != 0.0:
            token_counts = {
                "redux": cond.shape[1],
                "img": None
            }

            m = model.clone()
            # patch the model
            previous_patches = m.model_options["transformer_options"].get("patches_replace", {}).get("dit", {})

            for i, block in enumerate(m.model.diffusion_model.double_blocks):
                # is there already a patch there?
                # if so, the attnwrapper can chain off it
                previous = previous_patches.get(("double_block", i), block)
                wrapper = _ReduxAttnWrapper(previous, token_counts, bias=attn_bias, is_first=i==0)
                # I think this properly clones things?
                m.set_model_patch_replace(wrapper, "dit", "double_block", i)

            for i, block in enumerate(m.model.diffusion_model.single_blocks):
                previous = previous_patches.get(("single_block", i), block)
                wrapper = _ReduxAttnWrapper(previous, token_counts, bias=attn_bias)
                m.set_model_patch_replace(wrapper, "dit",  "single_block", i)
        else:
            m = model
        return (m, c)

NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeFlux": CLIPTextEncodeFlux,
    "FluxGuidance": FluxGuidance,
    "ReduxWithAttnMask": ReduxApplyWithAttnMask
}
