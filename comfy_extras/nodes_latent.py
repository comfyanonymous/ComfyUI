import comfy.utils
import comfy_extras.nodes_post_processing
import torch

def reshape_latent_to(target_shape, latent):
    if latent.shape[1:] != target_shape[1:]:
        latent = comfy.utils.common_upscale(latent, target_shape[3], target_shape[2], "bilinear", "center")
    return comfy.utils.repeat_to_batch_size(latent, target_shape[0])


class LatentAdd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples1": ("LATENT",), "samples2": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "op"

    CATEGORY = "latent/advanced"

    def op(self, samples1, samples2):
        samples_out = samples1.copy()

        s1 = samples1["samples"]
        s2 = samples2["samples"]

        s2 = reshape_latent_to(s1.shape, s2)
        samples_out["samples"] = s1 + s2
        return (samples_out,)

class LatentSubtract:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples1": ("LATENT",), "samples2": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "op"

    CATEGORY = "latent/advanced"

    def op(self, samples1, samples2):
        samples_out = samples1.copy()

        s1 = samples1["samples"]
        s2 = samples2["samples"]

        s2 = reshape_latent_to(s1.shape, s2)
        samples_out["samples"] = s1 - s2
        return (samples_out,)

class LatentMultiply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "multiplier": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "op"

    CATEGORY = "latent/advanced"

    def op(self, samples, multiplier):
        samples_out = samples.copy()

        s1 = samples["samples"]
        samples_out["samples"] = s1 * multiplier
        return (samples_out,)

class LatentInterpolate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples1": ("LATENT",),
                              "samples2": ("LATENT",),
                              "ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "op"

    CATEGORY = "latent/advanced"

    def op(self, samples1, samples2, ratio):
        samples_out = samples1.copy()

        s1 = samples1["samples"]
        s2 = samples2["samples"]

        s2 = reshape_latent_to(s1.shape, s2)

        m1 = torch.linalg.vector_norm(s1, dim=(1))
        m2 = torch.linalg.vector_norm(s2, dim=(1))

        s1 = torch.nan_to_num(s1 / m1)
        s2 = torch.nan_to_num(s2 / m2)

        t = (s1 * ratio + s2 * (1.0 - ratio))
        mt = torch.linalg.vector_norm(t, dim=(1))
        st = torch.nan_to_num(t / mt)

        samples_out["samples"] = st * (m1 * ratio + m2 * (1.0 - ratio))
        return (samples_out,)

class LatentBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples1": ("LATENT",), "samples2": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "batch"

    CATEGORY = "latent/batch"

    def batch(self, samples1, samples2):
        samples_out = samples1.copy()
        s1 = samples1["samples"]
        s2 = samples2["samples"]

        if s1.shape[1:] != s2.shape[1:]:
            s2 = comfy.utils.common_upscale(s2, s1.shape[3], s1.shape[2], "bilinear", "center")
        s = torch.cat((s1, s2), dim=0)
        samples_out["samples"] = s
        samples_out["batch_index"] = samples1.get("batch_index", [x for x in range(0, s1.shape[0])]) + samples2.get("batch_index", [x for x in range(0, s2.shape[0])])
        return (samples_out,)

class LatentBatchSeedBehavior:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "seed_behavior": (["random", "fixed"],{"default": "fixed"}),}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "op"

    CATEGORY = "latent/advanced"

    def op(self, samples, seed_behavior):
        samples_out = samples.copy()
        latent = samples["samples"]
        if seed_behavior == "random":
            if 'batch_index' in samples_out:
                samples_out.pop('batch_index')
        elif seed_behavior == "fixed":
            batch_number = samples_out.get("batch_index", [0])[0]
            samples_out["batch_index"] = [batch_number] * latent.shape[0]

        return (samples_out,)

class LatentApplyOperation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                             "operation": ("LATENT_OPERATION",),
                             }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "op"

    CATEGORY = "latent/advanced/operations"
    EXPERIMENTAL = True

    def op(self, samples, operation):
        samples_out = samples.copy()

        s1 = samples["samples"]
        samples_out["samples"] = operation(latent=s1)
        return (samples_out,)

class LatentApplyOperationCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             "operation": ("LATENT_OPERATION",),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "latent/advanced/operations"
    EXPERIMENTAL = True

    def patch(self, model, operation):
        m = model.clone()

        def pre_cfg_function(args):
            conds_out = args["conds_out"]
            if len(conds_out) == 2:
                conds_out[0] = operation(latent=(conds_out[0] - conds_out[1])) + conds_out[1]
            else:
                conds_out[0] = operation(latent=conds_out[0])
            return conds_out

        m.set_model_sampler_pre_cfg_function(pre_cfg_function)
        return (m, )

class LatentOperationTonemapReinhard:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                              }}

    RETURN_TYPES = ("LATENT_OPERATION",)
    FUNCTION = "op"

    CATEGORY = "latent/advanced/operations"
    EXPERIMENTAL = True

    def op(self, multiplier):
        def tonemap_reinhard(latent, **kwargs):
            latent_vector_magnitude = (torch.linalg.vector_norm(latent, dim=(1)) + 0.0000000001)[:,None]
            normalized_latent = latent / latent_vector_magnitude

            mean = torch.mean(latent_vector_magnitude, dim=(1,2,3), keepdim=True)
            std = torch.std(latent_vector_magnitude, dim=(1,2,3), keepdim=True)

            top = (std * 5 + mean) * multiplier

            #reinhard
            latent_vector_magnitude *= (1.0 / top)
            new_magnitude = latent_vector_magnitude / (latent_vector_magnitude + 1.0)
            new_magnitude *= top

            return normalized_latent * new_magnitude
        return (tonemap_reinhard,)

class LatentOperationSharpen:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "sharpen_radius": ("INT", {
                    "default": 9,
                    "min": 1,
                    "max": 31,
                    "step": 1
                }),
                "sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
                "alpha": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01
                }),
                              }}

    RETURN_TYPES = ("LATENT_OPERATION",)
    FUNCTION = "op"

    CATEGORY = "latent/advanced/operations"
    EXPERIMENTAL = True

    def op(self, sharpen_radius, sigma, alpha):
        def sharpen(latent, **kwargs):
            luminance = (torch.linalg.vector_norm(latent, dim=(1)) + 1e-6)[:,None]
            normalized_latent = latent / luminance
            channels = latent.shape[1]

            kernel_size = sharpen_radius * 2 + 1
            kernel = comfy_extras.nodes_post_processing.gaussian_kernel(kernel_size, sigma, device=luminance.device)
            center = kernel_size // 2

            kernel *= alpha * -10
            kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0

            padded_image = torch.nn.functional.pad(normalized_latent, (sharpen_radius,sharpen_radius,sharpen_radius,sharpen_radius), 'reflect')
            sharpened = torch.nn.functional.conv2d(padded_image, kernel.repeat(channels, 1, 1).unsqueeze(1), padding=kernel_size // 2, groups=channels)[:,:,sharpen_radius:-sharpen_radius, sharpen_radius:-sharpen_radius]

            return luminance * sharpened
        return (sharpen,)

NODE_CLASS_MAPPINGS = {
    "LatentAdd": LatentAdd,
    "LatentSubtract": LatentSubtract,
    "LatentMultiply": LatentMultiply,
    "LatentInterpolate": LatentInterpolate,
    "LatentBatch": LatentBatch,
    "LatentBatchSeedBehavior": LatentBatchSeedBehavior,
    "LatentApplyOperation": LatentApplyOperation,
    "LatentApplyOperationCFG": LatentApplyOperationCFG,
    "LatentOperationTonemapReinhard": LatentOperationTonemapReinhard,
    "LatentOperationSharpen": LatentOperationSharpen,
}
