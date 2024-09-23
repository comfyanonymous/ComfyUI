import torch

import comfy.utils
from comfy.component_model.tensor_types import Latent


def reshape_latent_to(target_shape, latent):
    if latent.shape[1:] != target_shape[1:]:
        latent = comfy.utils.common_upscale(latent, target_shape[3], target_shape[2], "bilinear", "center")
    return comfy.utils.repeat_to_batch_size(latent, target_shape[0])


class LatentAdd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples1": ("LATENT",), "samples2": ("LATENT",)}}

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
        return {"required": {"samples1": ("LATENT",), "samples2": ("LATENT",)}}

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
        return {"required": {"samples": ("LATENT",),
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
        return {"required": {"samples1": ("LATENT",),
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
        return {"required": {"samples1": ("LATENT",), "samples2": ("LATENT",)}}

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
        return {"required": {"samples": ("LATENT",),
                             "seed_behavior": (["random", "fixed"], {"default": "fixed"}), }}

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


class LatentAddNoiseChannels:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "std_dev": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "slice_i": ("INT", {"default": 0, "min": -16, "max": 16}),
                "slice_j": ("INT", {"default": 16, "min": -16, "max": 16}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "inject_noise"

    CATEGORY = "latent/advanced"

    def inject_noise(self, samples: Latent, std_dev, seed: int, slice_i: int, slice_j: int):
        s = samples.copy()

        latent = samples["samples"]
        with comfy.utils.seed_for_block(seed):
            if not isinstance(latent, torch.Tensor):
                raise TypeError("Input must be a PyTorch tensor")

            noise = torch.randn_like(latent[:, slice_i:slice_j, :, :]) * std_dev

        noised_latent = latent.clone()
        noised_latent[:, slice_i:slice_j, :, :] += noise
        s["samples"] = noised_latent
        return (s,)


NODE_CLASS_MAPPINGS = {
    "LatentAdd": LatentAdd,
    "LatentSubtract": LatentSubtract,
    "LatentMultiply": LatentMultiply,
    "LatentInterpolate": LatentInterpolate,
    "LatentBatch": LatentBatch,
    "LatentBatchSeedBehavior": LatentBatchSeedBehavior,
    "LatentAddNoiseChannels": LatentAddNoiseChannels,
}
