#Taken from: https://github.com/tfernd/HyperTile/

import math
from einops import rearrange
import random
from functools import cache

def random_divisor(value: int, min_value: int, /, max_options: int = 1, counter = 0) -> int:
    min_value = min(min_value, value)

    # All big divisors of value (inclusive)
    divisors = [i for i in range(min_value, value + 1) if value % i == 0]

    ns = [value // i for i in divisors[:max_options]]  # has at least 1 element

    random.seed(counter)
    idx = random.randint(0, len(ns) - 1)

    return ns[idx]

def iterative_closest_divisors(hw:int, aspect_ratio:float) -> tuple[int, int]:
    """
    Finds h and w such that h*w = hw and h/w = aspect_ratio
    We check all possible divisors of hw and return the closest to the aspect ratio
    """
    divisors = [i for i in range(2, hw + 1) if hw % i == 0] # all divisors of hw
    pairs = [(i, hw // i) for i in divisors] # all pairs of divisors of hw
    ratios = [w/h for h, w in pairs] # all ratios of pairs of divisors of hw
    closest_ratio = min(ratios, key=lambda x: abs(x - aspect_ratio)) # closest ratio to aspect_ratio
    closest_pair = pairs[ratios.index(closest_ratio)] # closest pair of divisors to aspect_ratio
    return closest_pair

@cache
def find_hw_candidates(hw:int, aspect_ratio:float) -> tuple[int, int]:
    """
    Finds h and w such that h*w = hw and h/w = aspect_ratio
    """
    h, w = round(math.sqrt(hw * aspect_ratio)), round(math.sqrt(hw / aspect_ratio))
    # find h and w such that h*w = hw and h/w = aspect_ratio
    if h * w != hw:
        w_candidate = hw / h
        # check if w is an integer
        if not w_candidate.is_integer():
            h_candidate = hw / w
            # check if h is an integer
            if not h_candidate.is_integer():
                return iterative_closest_divisors(hw, aspect_ratio)
            else:
                h = int(h_candidate)
        else:
            w = int(w_candidate)
    return h, w

class HyperTile:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             "tile_size": ("INT", {"default": 256, "min": 1, "max": 2048}),
                             "swap_size": ("INT", {"default": 2, "min": 1, "max": 128}),
                             "max_depth": ("INT", {"default": 0, "min": 0, "max": 10}),
                             "scale_depth": ("BOOLEAN", {"default": False}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, tile_size, swap_size, max_depth, scale_depth):
        model_channels = model.model.model_config.unet_config["model_channels"]

        apply_to = set()
        temp = model_channels
        for x in range(max_depth + 1):
            apply_to.add(temp)
            temp *= 2

        latent_tile_size = max(32, tile_size) // 8
        self.temp = None
        self.counter = 1

        def hypertile_in(q, k, v, extra_options):
            if q.shape[-1] in apply_to:
                shape = extra_options["original_shape"]
                aspect_ratio = shape[-1] / shape[-2]

                hw = q.size(1)
                h, w = find_hw_candidates(hw, aspect_ratio)

                factor = 2**((q.shape[-1] // model_channels) - 1) if scale_depth else 1
                nh = random_divisor(h, latent_tile_size * factor, swap_size, self.counter)
                self.counter += 1
                nw = random_divisor(w, latent_tile_size * factor, swap_size, self.counter)
                self.counter += 1

                if nh * nw > 1:
                    q = rearrange(q, "b (nh h nw w) c -> (b nh nw) (h w) c", h=h // nh, w=w // nw, nh=nh, nw=nw)
                    self.temp = (nh, nw, h, w)
                return q, k, v

            return q, k, v
        def hypertile_out(out, extra_options):
            if self.temp is not None:
                nh, nw, h, w = self.temp
                self.temp = None
                out = rearrange(out, "(b nh nw) hw c -> b nh nw hw c", nh=nh, nw=nw)
                out = rearrange(out, "b nh nw (h w) c -> b (nh h nw w) c", h=h // nh, w=w // nw)
            return out


        m = model.clone()
        m.set_model_attn1_patch(hypertile_in)
        m.set_model_attn1_output_patch(hypertile_out)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "HyperTile": HyperTile,
}
