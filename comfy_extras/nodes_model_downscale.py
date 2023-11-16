import torch

class PatchModelAddDownscale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "block_number": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
                              "downscale_factor": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 9.0, "step": 0.001}),
                              "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                              "end_percent": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.001}),
                              "downscale_after_skip": ("BOOLEAN", {"default": True}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, block_number, downscale_factor, start_percent, end_percent, downscale_after_skip):
        sigma_start = model.model.model_sampling.percent_to_sigma(start_percent).item()
        sigma_end = model.model.model_sampling.percent_to_sigma(end_percent).item()

        def input_block_patch(h, transformer_options):
            if transformer_options["block"][1] == block_number:
                sigma = transformer_options["sigmas"][0].item()
                if sigma <= sigma_start and sigma >= sigma_end:
                    h = torch.nn.functional.interpolate(h, scale_factor=(1.0 / downscale_factor), mode="bicubic", align_corners=False)
            return h

        def output_block_patch(h, hsp, transformer_options):
            if h.shape[2] != hsp.shape[2]:
                h = torch.nn.functional.interpolate(h, size=(hsp.shape[2], hsp.shape[3]), mode="bicubic", align_corners=False)
            return h, hsp

        m = model.clone()
        if downscale_after_skip:
            m.set_model_input_block_patch_after_skip(input_block_patch)
        else:
            m.set_model_input_block_patch(input_block_patch)
        m.set_model_output_block_patch(output_block_patch)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "PatchModelAddDownscale": PatchModelAddDownscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Sampling
    "PatchModelAddDownscale": "PatchModelAddDownscale (Kohya Deep Shrink)",
}
