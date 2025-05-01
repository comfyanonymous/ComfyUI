# from https://github.com/bebebe666/OptimalSteps


import numpy as np
import torch

def loglinear_interp(t_steps, num_steps):
    """
    Performs log-linear interpolation of a given array of decreasing numbers.
    """
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])

    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)

    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys


NOISE_LEVELS = {"FLUX": [0.9968, 0.9886, 0.9819, 0.975, 0.966, 0.9471, 0.9158, 0.8287, 0.5512, 0.2808, 0.001],
"Wan":[1.0, 0.997, 0.995, 0.993, 0.991, 0.989, 0.987, 0.985, 0.98, 0.975, 0.973, 0.968, 0.96, 0.946, 0.927, 0.902, 0.864, 0.776, 0.539, 0.208, 0.001],
"Chroma": [0.9919999837875366, 0.9900000095367432, 0.9879999756813049, 0.9850000143051147, 0.9819999933242798, 0.9779999852180481, 0.9729999899864197, 0.9679999947547913, 0.9610000252723694, 0.953000009059906, 0.9430000185966492, 0.9309999942779541, 0.9169999957084656, 0.8999999761581421, 0.8809999823570251, 0.8579999804496765, 0.8320000171661377, 0.8019999861717224, 0.7689999938011169, 0.7310000061988831, 0.6899999976158142, 0.6460000276565552, 0.5989999771118164, 0.550000011920929, 0.5009999871253967, 0.45100000500679016, 0.4020000100135803, 0.35499998927116394, 0.3109999895095825, 0.27000001072883606, 0.23199999332427979, 0.19900000095367432, 0.16899999976158142, 0.14300000667572021, 0.11999999731779099, 0.10100000351667404, 0.08399999886751175, 0.07000000029802322, 0.057999998331069946, 0.04800000041723251, 0.0],
}

class OptimalStepsScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model_type": (["FLUX", "Wan", "Chroma"], ),
                     "steps": ("INT", {"default": 20, "min": 3, "max": 1000}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, model_type, steps, denoise):
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            total_steps = round(steps * denoise)

        sigmas = NOISE_LEVELS[model_type][:]
        if (steps + 1) != len(sigmas):
            sigmas = loglinear_interp(sigmas, steps + 1)

        sigmas = sigmas[-(total_steps + 1):]
        sigmas[-1] = 0
        return (torch.FloatTensor(sigmas), )

NODE_CLASS_MAPPINGS = {
    "OptimalStepsScheduler": OptimalStepsScheduler,
}
