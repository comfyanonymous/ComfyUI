import comfy.model_patcher
import comfy.samplers
import re


class SkipLayerGuidanceDiT:
    '''
    Enhance guidance towards detailed dtructure by having another set of CFG negative with skipped layers.
    Inspired by Perturbed Attention Guidance (https://arxiv.org/abs/2403.17377)
    Original experimental implementation for SD3 by Dango233@StabilityAI.
    '''
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                             "double_layers": ("STRING", {"default": "7, 8, 9", "multiline": False}),
                             "single_layers": ("STRING", {"default": "7, 8, 9", "multiline": False}),
                             "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                             "start_percent": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "rescaling_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                                }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "skip_guidance"
    EXPERIMENTAL = True

    DESCRIPTION = "Generic version of SkipLayerGuidance node that can be used on every DiT model."

    CATEGORY = "advanced/guidance"

    def skip_guidance(self, model, scale, start_percent, end_percent, double_layers="", single_layers="", rescaling_scale=0):
        # check if layer is comma separated integers
        def skip(args, extra_args):
            return args

        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)

        double_layers = re.findall(r'\d+', double_layers)
        double_layers = [int(i) for i in double_layers]

        single_layers = re.findall(r'\d+', single_layers)
        single_layers = [int(i) for i in single_layers]

        if len(double_layers) == 0 and len(single_layers) == 0:
            return (model, )

        def post_cfg_function(args):
            model = args["model"]
            cond_pred = args["cond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            x = args["input"]
            model_options = args["model_options"].copy()

            for layer in double_layers:
                model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, skip, "dit", "double_block", layer)

            for layer in single_layers:
                model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, skip, "dit", "single_block", layer)

            model_sampling.percent_to_sigma(start_percent)

            sigma_ = sigma[0].item()
            if scale > 0 and sigma_ >= sigma_end and sigma_ <= sigma_start:
                (slg,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)
                cfg_result = cfg_result + (cond_pred - slg) * scale
                if rescaling_scale != 0:
                    factor = cond_pred.std() / cfg_result.std()
                    factor = rescaling_scale * factor + (1 - rescaling_scale)
                    cfg_result *= factor

            return cfg_result

        m = model.clone()
        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m, )


NODE_CLASS_MAPPINGS = {
    "SkipLayerGuidanceDiT": SkipLayerGuidanceDiT,
}
