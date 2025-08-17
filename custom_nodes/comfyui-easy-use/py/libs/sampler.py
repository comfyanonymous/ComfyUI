import comfy
import comfy.model_management
import comfy.samplers
import torch
import numpy as np
import latent_preview
from nodes import MAX_RESOLUTION
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union, Any
from ..modules.brushnet.model_patch import add_model_patch

class easySampler:
    def __init__(self):
        self.last_helds: dict[str, list] = {
            "results": [],
            "pipe_line": [],
        }
        self.device = comfy.model_management.intermediate_device()

    @staticmethod
    def tensor2pil(image: torch.Tensor) -> Image.Image:
        """Convert a torch tensor to a PIL image."""
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    @staticmethod
    def pil2tensor(image: Image.Image) -> torch.Tensor:
        """Convert a PIL image to a torch tensor."""
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def enforce_mul_of_64(d):
        d = int(d)
        if d <= 7:
            d = 8
        leftover = d % 8  # 8 is the number of pixels per byte
        if leftover != 0:  # if the number of pixels is not a multiple of 8
            if (leftover < 4):  # if the number of pixels is less than 4
                d -= leftover  # remove the leftover pixels
            else:  # if the number of pixels is more than 4
                d += 8 - leftover  # add the leftover pixels

        return int(d)

    @staticmethod
    def safe_split(to_split: str, delimiter: str) -> List[str]:
        """Split the input string and return a list of non-empty parts."""
        parts = to_split.split(delimiter)
        parts = [part for part in parts if part not in ('', ' ', '  ')]

        while len(parts) < 2:
            parts.append('None')
        return parts

    def emptyLatent(self, resolution, empty_latent_width, empty_latent_height, batch_size=1, compression=0, model_type='sd', video_length=25):
        if resolution not in ["自定义 x 自定义", 'width x height (custom)']:
            try:
                width, height = map(int, resolution.split(' x '))
                empty_latent_width = width
                empty_latent_height = height
            except ValueError:
                raise ValueError("Invalid base_resolution format.")
        if model_type == 'sd3':
            latent = torch.ones([batch_size, 16, empty_latent_height // 8, empty_latent_width // 8], device=self.device) * 0.0609
            samples = {"samples": latent}
        elif model_type == 'mochi':
            latent = torch.zeros([batch_size, 12, ((video_length - 1) // 6) + 1, empty_latent_height // 8, empty_latent_width // 8], device=self.device)
            samples = {"samples": latent}
        elif compression == 0:
            latent = torch.zeros([batch_size, 4, empty_latent_height // 8, empty_latent_width // 8], device=self.device)
            samples = {"samples": latent}
        else:
            latent_c = torch.zeros(
                [batch_size, 16, empty_latent_height // compression, empty_latent_width // compression])
            latent_b = torch.zeros([batch_size, 4, empty_latent_height // 4, empty_latent_width // 4])

            samples = ({"samples": latent_c}, {"samples": latent_b})
        return samples

    def prepare_noise(self, latent_image, seed, noise_inds=None, noise_device="cpu", incremental_seed_mode="comfy",
                      variation_seed=None, variation_strength=None):
        """
        creates random noise given a latent image and a seed.
        optional arg skip can be used to skip and discard x number of noise generations for a given seed
        """

        latent_size = latent_image.size()
        latent_size_1batch = [1, latent_size[1], latent_size[2], latent_size[3]]

        if variation_strength is not None and variation_strength > 0 or incremental_seed_mode.startswith(
                "variation str inc"):
            if noise_device == "cpu":
                variation_generator = torch.manual_seed(variation_seed)
            else:
                torch.cuda.manual_seed(variation_seed)
                variation_generator = None

            variation_latent = torch.randn(latent_size_1batch, dtype=latent_image.dtype, layout=latent_image.layout,
                                           generator=variation_generator, device=noise_device)
        else:
            variation_latent = None

        def apply_variation(input_latent, strength_up=None):
            if variation_latent is None:
                return input_latent
            else:
                strength = variation_strength

                if strength_up is not None:
                    strength += strength_up

                variation_noise = variation_latent.expand(input_latent.size()[0], -1, -1, -1)
                result = (1 - strength) * input_latent + strength * variation_noise
                return result

        # method: incremental seed batch noise
        if noise_inds is None and incremental_seed_mode == "incremental":
            batch_cnt = latent_size[0]

            latents = None
            for i in range(batch_cnt):
                if noise_device == "cpu":
                    generator = torch.manual_seed(seed + i)
                else:
                    torch.cuda.manual_seed(seed + i)
                    generator = None

                latent = torch.randn(latent_size_1batch, dtype=latent_image.dtype, layout=latent_image.layout,
                                     generator=generator, device=noise_device)

                latent = apply_variation(latent)

                if latents is None:
                    latents = latent
                else:
                    latents = torch.cat((latents, latent), dim=0)

            return latents

        # method: incremental variation batch noise
        elif noise_inds is None and incremental_seed_mode.startswith("variation str inc"):
            batch_cnt = latent_size[0]

            latents = None
            for i in range(batch_cnt):
                if noise_device == "cpu":
                    generator = torch.manual_seed(seed)
                else:
                    torch.cuda.manual_seed(seed)
                    generator = None

                latent = torch.randn(latent_size_1batch, dtype=latent_image.dtype, layout=latent_image.layout,
                                     generator=generator, device=noise_device)

                step = float(incremental_seed_mode[18:])
                latent = apply_variation(latent, step * i)

                if latents is None:
                    latents = latent
                else:
                    latents = torch.cat((latents, latent), dim=0)

            return latents

        # method: comfy batch noise
        if noise_device == "cpu":
            generator = torch.manual_seed(seed)
        else:
            torch.cuda.manual_seed(seed)
            generator = None

        if noise_inds is None:
            latents = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout,
                                  generator=generator, device=noise_device)
            latents = apply_variation(latents)
            return latents

        unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
        noises = []
        for i in range(unique_inds[-1] + 1):
            noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype,
                                layout=latent_image.layout,
                                generator=generator, device=noise_device)
            if i in unique_inds:
                noises.append(noise)
        noises = [noises[i] for i in inverse]
        noises = torch.cat(noises, axis=0)
        return noises

    def common_ksampler(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0,
                        disable_noise=False, start_step=None, last_step=None, force_full_denoise=False,
                        preview_latent=True, disable_pbar=False, noise_device='CPU'):
        device = comfy.model_management.get_torch_device()
        noise_device = 'cpu' if noise_device == 'CPU' else device
        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        preview_format = "JPEG"
        if preview_format not in ["JPEG", "PNG"]:
            preview_format = "JPEG"

        previewer = False

        if preview_latent:
            previewer = latent_preview.get_previewer(device, model.model.latent_format)

        pbar = comfy.utils.ProgressBar(steps)

        def callback(step, x0, x, total_steps):
            preview_bytes = None
            if previewer:
                preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
            pbar.update_absolute(step + 1, total_steps, preview_bytes)

        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout,
                                device=noise_device)
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = self.prepare_noise(latent_image, seed, batch_inds, noise_device=noise_device)

        #######################################################################################
        # add model patch
        # brushnet
        add_model_patch(model)
        #######################################################################################
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative,
                                      latent_image,
                                      denoise=denoise, disable_noise=disable_noise, start_step=start_step,
                                      last_step=last_step,
                                      force_full_denoise=force_full_denoise, noise_mask=noise_mask,
                                      callback=callback,
                                      disable_pbar=disable_pbar, seed=seed)
        out = latent.copy()
        out["samples"] = samples
        return out

    def custom_ksampler(self, model, seed, steps, cfg, _sampler, sigmas, positive, negative, latent,
                        disable_noise=False, preview_latent=True,  disable_pbar=False, noise_device='CPU'):

        device = comfy.model_management.get_torch_device()
        noise_device = 'cpu' if noise_device == 'CPU' else device

        latent_image = latent["samples"]

        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device=noise_device)
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = self.prepare_noise(latent_image, seed, batch_inds, noise_device=noise_device)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        preview_format = "JPEG"
        if preview_format not in ["JPEG", "PNG"]:
            preview_format = "JPEG"

        previewer = False

        if preview_latent:
            previewer = latent_preview.get_previewer(device, model.model.latent_format)

        pbar = comfy.utils.ProgressBar(steps)

        def callback(step, x0, x, total_steps):
            preview_bytes = None
            if previewer:
                preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
            pbar.update_absolute(step + 1, total_steps, preview_bytes)

        samples = comfy.samplers.sample(model, noise, positive, negative, cfg, device, _sampler, sigmas, latent_image=latent_image, model_options=model.model_options,
               denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)

        out = latent.copy()
        out["samples"] = samples
        return out

    def custom_advanced_ksampler(self, guider, sampler, sigmas, latent_image, add_noise='enable', seed=0, preview_latent=False):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

        device = comfy.model_management.get_torch_device()
        noise_device = device if add_noise == 'enable (GPU=A1111)' else 'cpu'

        if add_noise == 'disable':
            noise = torch.zeros(latent_image.shape, dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = self.prepare_noise(latent_image, seed, batch_inds, noise_device=noise_device)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        previewer = False

        model = guider.model_patcher
        steps = sigmas.shape[-1] - 1
        if preview_latent:
            previewer = latent_preview.get_previewer(model.load_device, model.model.latent_format)

        pbar = comfy.utils.ProgressBar(steps)

        preview_format = "JPEG"
        if preview_format not in ["JPEG", "PNG"]:
            preview_format = "JPEG"
        def callback(step, x0, x, total_steps):
            if x0_output is not None:
                x0_output["x0"] = x0

            preview_bytes = None
            if previewer:
                preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
            pbar.update_absolute(step + 1, total_steps, preview_bytes)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(noise, latent_image, sampler, sigmas, denoise_mask=noise_mask,
                                callback=callback, disable_pbar=disable_pbar, seed=seed)
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        return (out, out_denoised)

    def get_value_by_id(self, key: str, my_unique_id: Any) -> Optional[Any]:
        """Retrieve value by its associated ID."""
        try:
            for value, id_ in self.last_helds[key]:
                if id_ == my_unique_id:
                    return value
        except KeyError:
            return None

    def update_value_by_id(self, key: str, my_unique_id: Any, new_value: Any) -> Union[bool, None]:
        """Update the value associated with a given ID. Return True if updated, False if appended, None if key doesn't exist."""
        try:
            for i, (value, id_) in enumerate(self.last_helds[key]):
                if id_ == my_unique_id:
                    self.last_helds[key][i] = (new_value, id_)
                    return True
            self.last_helds[key].append((new_value, my_unique_id))
            return False
        except KeyError:
            return False

    def upscale(self, samples, upscale_method, scale_by, crop):
        s = samples.copy()
        width = self.enforce_mul_of_64(round(samples["samples"].shape[3] * scale_by))
        height = self.enforce_mul_of_64(round(samples["samples"].shape[2] * scale_by))

        if (width > MAX_RESOLUTION):
            width = MAX_RESOLUTION
        if (height > MAX_RESOLUTION):
            height = MAX_RESOLUTION

        s["samples"] = comfy.utils.common_upscale(samples["samples"], width, height, upscale_method, crop)
        return (s,)

    def handle_upscale(self, samples: dict, upscale_method: str, factor: float, crop: bool) -> dict:
        """Upscale the samples if the upscale_method is not set to 'None'."""
        if upscale_method != "None":
            samples = self.upscale(samples, upscale_method, factor, crop)[0]
        return samples

    def init_state(self, my_unique_id: Any, key: str, default: Any) -> Any:
        """Initialize the state by either fetching the stored value or setting a default."""
        value = self.get_value_by_id(key, my_unique_id)
        if value is not None:
            return value
        return default

    def get_output(self, pipe: dict,) -> Tuple:
        """Return a tuple of various elements fetched from the input pipe dictionary."""
        return (
            pipe,
            pipe.get("images"),
            pipe.get("model"),
            pipe.get("positive"),
            pipe.get("negative"),
            pipe.get("samples"),
            pipe.get("vae"),
            pipe.get("clip"),
            pipe.get("seed"),
        )

    def get_output_sdxl(self, sdxl_pipe: dict) -> Tuple:
        """Return a tuple of various elements fetched from the input sdxl_pipe dictionary."""
        return (
            sdxl_pipe,
            sdxl_pipe.get("model"),
            sdxl_pipe.get("positive"),
            sdxl_pipe.get("negative"),
            sdxl_pipe.get("vae"),
            sdxl_pipe.get("refiner_model"),
            sdxl_pipe.get("refiner_positive"),
            sdxl_pipe.get("refiner_negative"),
            sdxl_pipe.get("refiner_vae"),
            sdxl_pipe.get("samples"),
            sdxl_pipe.get("clip"),
            sdxl_pipe.get("images"),
            sdxl_pipe.get("seed")
        )

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

class alignYourStepsScheduler:

    NOISE_LEVELS = {
        "SD1": [14.6146412293, 6.4745760956, 3.8636745985, 2.6946151520, 1.8841921177, 1.3943805092, 0.9642583904,
                0.6523686016, 0.3977456272, 0.1515232662, 0.0291671582],
        "SDXL": [14.6146412293, 6.3184485287, 3.7681790315, 2.1811480769, 1.3405244945, 0.8620721141, 0.5550693289,
                 0.3798540708, 0.2332364134, 0.1114188177, 0.0291671582],
        "SVD": [700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002]}

    def get_sigmas(self, model_type, steps, denoise):

        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            total_steps = round(steps * denoise)

        sigmas = self.NOISE_LEVELS[model_type][:]
        if (steps + 1) != len(sigmas):
            sigmas = loglinear_interp(sigmas, steps + 1)

        sigmas = sigmas[-(total_steps + 1):]
        sigmas[-1] = 0
        return (torch.FloatTensor(sigmas),)


class gitsScheduler:

    NOISE_LEVELS = {
        0.80: [
            [14.61464119, 7.49001646, 0.02916753],
            [14.61464119, 11.54541874, 6.77309084, 0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 3.07277966, 0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 5.85520077, 2.05039096, 0.02916753],
            [14.61464119, 12.2308979, 8.75849152, 7.49001646, 5.85520077, 2.05039096, 0.02916753],
            [14.61464119, 12.2308979, 8.75849152, 7.49001646, 5.85520077, 3.07277966, 1.56271636, 0.02916753],
            [14.61464119, 12.96784878, 11.54541874, 8.75849152, 7.49001646, 5.85520077, 3.07277966, 1.56271636,
             0.02916753],
            [14.61464119, 13.76078796, 12.2308979, 10.90732002, 8.75849152, 7.49001646, 5.85520077, 3.07277966,
             1.56271636, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 10.90732002, 8.75849152, 7.49001646, 5.85520077,
             3.07277966, 1.56271636, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646,
             5.85520077, 3.07277966, 1.56271636, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646,
             6.14220476, 4.86714602, 3.07277966, 1.56271636, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.31284904, 9.24142551, 8.30717278,
             7.49001646, 6.14220476, 4.86714602, 3.07277966, 1.56271636, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.24142551,
             8.30717278, 7.49001646, 6.14220476, 4.86714602, 3.07277966, 1.56271636, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.24142551,
             8.75849152, 8.30717278, 7.49001646, 6.14220476, 4.86714602, 3.07277966, 1.56271636, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.24142551,
             8.75849152, 8.30717278, 7.49001646, 6.14220476, 4.86714602, 3.1956799, 1.98035145, 0.86115354, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.75859547,
             9.24142551, 8.75849152, 8.30717278, 7.49001646, 6.14220476, 4.86714602, 3.1956799, 1.98035145, 0.86115354,
             0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.75859547,
             9.24142551, 8.75849152, 8.30717278, 7.49001646, 6.77309084, 5.85520077, 4.65472794, 3.07277966, 1.84880662,
             0.83188516, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.75859547,
             9.24142551, 8.75849152, 8.30717278, 7.88507891, 7.49001646, 6.77309084, 5.85520077, 4.65472794, 3.07277966,
             1.84880662, 0.83188516, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.75859547,
             9.24142551, 8.75849152, 8.30717278, 7.88507891, 7.49001646, 6.77309084, 5.85520077, 4.86714602, 3.75677586,
             2.84484982, 1.78698075, 0.803307, 0.02916753],
        ],
        0.85: [
            [14.61464119, 7.49001646, 0.02916753],
            [14.61464119, 7.49001646, 1.84880662, 0.02916753],
            [14.61464119, 11.54541874, 6.77309084, 1.56271636, 0.02916753],
            [14.61464119, 11.54541874, 7.11996698, 3.07277966, 1.24153244, 0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 5.09240818, 2.84484982, 0.95350921, 0.02916753],
            [14.61464119, 12.2308979, 8.75849152, 7.49001646, 5.09240818, 2.84484982, 0.95350921, 0.02916753],
            [14.61464119, 12.2308979, 8.75849152, 7.49001646, 5.58536053, 3.1956799, 1.84880662, 0.803307, 0.02916753],
            [14.61464119, 12.96784878, 11.54541874, 8.75849152, 7.49001646, 5.58536053, 3.1956799, 1.84880662, 0.803307,
             0.02916753],
            [14.61464119, 12.96784878, 11.54541874, 8.75849152, 7.49001646, 6.14220476, 4.65472794, 3.07277966,
             1.84880662, 0.803307, 0.02916753],
            [14.61464119, 13.76078796, 12.2308979, 10.90732002, 8.75849152, 7.49001646, 6.14220476, 4.65472794,
             3.07277966, 1.84880662, 0.803307, 0.02916753],
            [14.61464119, 13.76078796, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646, 6.14220476,
             4.65472794, 3.07277966, 1.84880662, 0.803307, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646,
             6.14220476, 4.65472794, 3.07277966, 1.84880662, 0.803307, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.31284904, 9.24142551, 8.30717278,
             7.49001646, 6.14220476, 4.65472794, 3.07277966, 1.84880662, 0.803307, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.31284904, 9.24142551, 8.30717278,
             7.49001646, 6.14220476, 4.86714602, 3.60512662, 2.6383388, 1.56271636, 0.72133851, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.31284904, 9.24142551, 8.30717278,
             7.49001646, 6.77309084, 5.85520077, 4.65472794, 3.46139455, 2.45070267, 1.56271636, 0.72133851,
             0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.31284904, 9.24142551, 8.75849152,
             8.30717278, 7.49001646, 6.77309084, 5.85520077, 4.65472794, 3.46139455, 2.45070267, 1.56271636, 0.72133851,
             0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.24142551,
             8.75849152, 8.30717278, 7.49001646, 6.77309084, 5.85520077, 4.65472794, 3.46139455, 2.45070267, 1.56271636,
             0.72133851, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.75859547,
             9.24142551, 8.75849152, 8.30717278, 7.49001646, 6.77309084, 5.85520077, 4.65472794, 3.46139455, 2.45070267,
             1.56271636, 0.72133851, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.75859547,
             9.24142551, 8.75849152, 8.30717278, 7.88507891, 7.49001646, 6.77309084, 5.85520077, 4.65472794, 3.46139455,
             2.45070267, 1.56271636, 0.72133851, 0.02916753],
        ],
        0.90: [
            [14.61464119, 6.77309084, 0.02916753],
            [14.61464119, 7.49001646, 1.56271636, 0.02916753],
            [14.61464119, 7.49001646, 3.07277966, 0.95350921, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 2.54230714, 0.89115214, 0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 4.86714602, 2.54230714, 0.89115214, 0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 5.09240818, 3.07277966, 1.61558151, 0.69515091, 0.02916753],
            [14.61464119, 12.2308979, 8.75849152, 7.11996698, 4.86714602, 3.07277966, 1.61558151, 0.69515091,
             0.02916753],
            [14.61464119, 12.2308979, 8.75849152, 7.49001646, 5.85520077, 4.45427561, 2.95596409, 1.61558151,
             0.69515091, 0.02916753],
            [14.61464119, 12.2308979, 8.75849152, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.19988537, 1.24153244,
             0.57119018, 0.02916753],
            [14.61464119, 12.96784878, 10.90732002, 8.75849152, 7.49001646, 5.85520077, 4.45427561, 3.1956799,
             2.19988537, 1.24153244, 0.57119018, 0.02916753],
            [14.61464119, 12.96784878, 11.54541874, 9.24142551, 8.30717278, 7.49001646, 5.85520077, 4.45427561,
             3.1956799, 2.19988537, 1.24153244, 0.57119018, 0.02916753],
            [14.61464119, 12.96784878, 11.54541874, 9.24142551, 8.30717278, 7.49001646, 6.14220476, 4.86714602,
             3.75677586, 2.84484982, 1.84880662, 1.08895338, 0.52423614, 0.02916753],
            [14.61464119, 13.76078796, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646, 6.14220476,
             4.86714602, 3.75677586, 2.84484982, 1.84880662, 1.08895338, 0.52423614, 0.02916753],
            [14.61464119, 13.76078796, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646, 6.44769001,
             5.58536053, 4.45427561, 3.32507086, 2.45070267, 1.61558151, 0.95350921, 0.45573691, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646,
             6.44769001, 5.58536053, 4.45427561, 3.32507086, 2.45070267, 1.61558151, 0.95350921, 0.45573691,
             0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646,
             6.77309084, 5.85520077, 4.86714602, 3.91689563, 3.07277966, 2.27973175, 1.56271636, 0.95350921, 0.45573691,
             0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.31284904, 9.24142551, 8.30717278,
             7.49001646, 6.77309084, 5.85520077, 4.86714602, 3.91689563, 3.07277966, 2.27973175, 1.56271636, 0.95350921,
             0.45573691, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.31284904, 9.24142551, 8.75849152,
             8.30717278, 7.49001646, 6.77309084, 5.85520077, 4.86714602, 3.91689563, 3.07277966, 2.27973175, 1.56271636,
             0.95350921, 0.45573691, 0.02916753],
            [14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.31284904, 9.24142551, 8.75849152,
             8.30717278, 7.49001646, 6.77309084, 5.85520077, 5.09240818, 4.45427561, 3.60512662, 2.95596409, 2.19988537,
             1.51179266, 0.89115214, 0.43325692, 0.02916753],
        ],
        0.95: [
            [14.61464119, 6.77309084, 0.02916753],
            [14.61464119, 6.77309084, 1.56271636, 0.02916753],
            [14.61464119, 7.49001646, 2.84484982, 0.89115214, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 2.36326075, 0.803307, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 2.95596409, 1.56271636, 0.64427125, 0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 4.86714602, 2.95596409, 1.56271636, 0.64427125, 0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 4.86714602, 3.07277966, 1.91321158, 1.08895338, 0.50118381,
             0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.45427561, 3.07277966, 1.91321158, 1.08895338,
             0.50118381, 0.02916753],
            [14.61464119, 12.2308979, 8.75849152, 7.49001646, 5.85520077, 4.45427561, 3.07277966, 1.91321158,
             1.08895338, 0.50118381, 0.02916753],
            [14.61464119, 12.2308979, 8.75849152, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.19988537, 1.41535246,
             0.803307, 0.38853383, 0.02916753],
            [14.61464119, 12.2308979, 8.75849152, 7.49001646, 5.85520077, 4.65472794, 3.46139455, 2.6383388, 1.84880662,
             1.24153244, 0.72133851, 0.34370604, 0.02916753],
            [14.61464119, 12.96784878, 10.90732002, 8.75849152, 7.49001646, 5.85520077, 4.65472794, 3.46139455,
             2.6383388, 1.84880662, 1.24153244, 0.72133851, 0.34370604, 0.02916753],
            [14.61464119, 12.96784878, 10.90732002, 8.75849152, 7.49001646, 6.14220476, 4.86714602, 3.75677586,
             2.95596409, 2.19988537, 1.56271636, 1.05362725, 0.64427125, 0.32104823, 0.02916753],
            [14.61464119, 12.96784878, 10.90732002, 8.75849152, 7.49001646, 6.44769001, 5.58536053, 4.65472794,
             3.60512662, 2.95596409, 2.19988537, 1.56271636, 1.05362725, 0.64427125, 0.32104823, 0.02916753],
            [14.61464119, 12.96784878, 11.54541874, 9.24142551, 8.30717278, 7.49001646, 6.44769001, 5.58536053,
             4.65472794, 3.60512662, 2.95596409, 2.19988537, 1.56271636, 1.05362725, 0.64427125, 0.32104823,
             0.02916753],
            [14.61464119, 12.96784878, 11.54541874, 9.24142551, 8.30717278, 7.49001646, 6.44769001, 5.58536053,
             4.65472794, 3.75677586, 3.07277966, 2.45070267, 1.78698075, 1.24153244, 0.83188516, 0.50118381, 0.22545385,
             0.02916753],
            [14.61464119, 12.96784878, 11.54541874, 9.24142551, 8.30717278, 7.49001646, 6.77309084, 5.85520077,
             5.09240818, 4.45427561, 3.60512662, 2.95596409, 2.36326075, 1.72759056, 1.24153244, 0.83188516, 0.50118381,
             0.22545385, 0.02916753],
            [14.61464119, 13.76078796, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646, 6.77309084,
             5.85520077, 5.09240818, 4.45427561, 3.60512662, 2.95596409, 2.36326075, 1.72759056, 1.24153244, 0.83188516,
             0.50118381, 0.22545385, 0.02916753],
            [14.61464119, 13.76078796, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646, 6.77309084,
             5.85520077, 5.09240818, 4.45427561, 3.75677586, 3.07277966, 2.45070267, 1.91321158, 1.46270394, 1.05362725,
             0.72133851, 0.43325692, 0.19894916, 0.02916753],
        ],
        1.00: [
            [14.61464119, 1.56271636, 0.02916753],
            [14.61464119, 6.77309084, 0.95350921, 0.02916753],
            [14.61464119, 6.77309084, 2.36326075, 0.803307, 0.02916753],
            [14.61464119, 7.11996698, 3.07277966, 1.56271636, 0.59516323, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 2.84484982, 1.41535246, 0.57119018, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 2.84484982, 1.61558151, 0.86115354, 0.38853383, 0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 4.86714602, 2.84484982, 1.61558151, 0.86115354, 0.38853383,
             0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 4.86714602, 3.07277966, 1.98035145, 1.24153244, 0.72133851,
             0.34370604, 0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.45427561, 3.07277966, 1.98035145, 1.24153244,
             0.72133851, 0.34370604, 0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.27973175, 1.51179266,
             0.95350921, 0.54755926, 0.25053367, 0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.36326075, 1.61558151,
             1.08895338, 0.72133851, 0.41087446, 0.17026083, 0.02916753],
            [14.61464119, 11.54541874, 8.75849152, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.36326075,
             1.61558151, 1.08895338, 0.72133851, 0.41087446, 0.17026083, 0.02916753],
            [14.61464119, 11.54541874, 8.75849152, 7.49001646, 5.85520077, 4.65472794, 3.60512662, 2.84484982,
             2.12350607, 1.56271636, 1.08895338, 0.72133851, 0.41087446, 0.17026083, 0.02916753],
            [14.61464119, 11.54541874, 8.75849152, 7.49001646, 5.85520077, 4.65472794, 3.60512662, 2.84484982,
             2.19988537, 1.61558151, 1.162866, 0.803307, 0.50118381, 0.27464288, 0.09824532, 0.02916753],
            [14.61464119, 11.54541874, 8.75849152, 7.49001646, 5.85520077, 4.65472794, 3.75677586, 3.07277966,
             2.45070267, 1.84880662, 1.36964464, 1.01931262, 0.72133851, 0.45573691, 0.25053367, 0.09824532,
             0.02916753],
            [14.61464119, 11.54541874, 8.75849152, 7.49001646, 6.14220476, 5.09240818, 4.26497746, 3.46139455,
             2.84484982, 2.19988537, 1.67050016, 1.24153244, 0.92192322, 0.64427125, 0.43325692, 0.25053367, 0.09824532,
             0.02916753],
            [14.61464119, 11.54541874, 8.75849152, 7.49001646, 6.14220476, 5.09240818, 4.26497746, 3.60512662,
             2.95596409, 2.45070267, 1.91321158, 1.51179266, 1.12534678, 0.83188516, 0.59516323, 0.38853383, 0.22545385,
             0.09824532, 0.02916753],
            [14.61464119, 12.2308979, 9.24142551, 8.30717278, 7.49001646, 6.14220476, 5.09240818, 4.26497746,
             3.60512662, 2.95596409, 2.45070267, 1.91321158, 1.51179266, 1.12534678, 0.83188516, 0.59516323, 0.38853383,
             0.22545385, 0.09824532, 0.02916753],
            [14.61464119, 12.2308979, 9.24142551, 8.30717278, 7.49001646, 6.77309084, 5.85520077, 5.09240818,
             4.26497746, 3.60512662, 2.95596409, 2.45070267, 1.91321158, 1.51179266, 1.12534678, 0.83188516, 0.59516323,
             0.38853383, 0.22545385, 0.09824532, 0.02916753],
        ],
        1.05: [
            [14.61464119, 0.95350921, 0.02916753],
            [14.61464119, 6.77309084, 0.89115214, 0.02916753],
            [14.61464119, 6.77309084, 2.05039096, 0.72133851, 0.02916753],
            [14.61464119, 6.77309084, 2.84484982, 1.28281462, 0.52423614, 0.02916753],
            [14.61464119, 6.77309084, 3.07277966, 1.61558151, 0.803307, 0.34370604, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 2.84484982, 1.56271636, 0.803307, 0.34370604, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 2.84484982, 1.61558151, 0.95350921, 0.52423614, 0.22545385,
             0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 3.07277966, 1.98035145, 1.24153244, 0.74807048, 0.41087446,
             0.17026083, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 3.1956799, 2.27973175, 1.51179266, 0.95350921, 0.59516323, 0.34370604,
             0.13792117, 0.02916753],
            [14.61464119, 7.49001646, 5.09240818, 3.46139455, 2.45070267, 1.61558151, 1.08895338, 0.72133851,
             0.45573691, 0.25053367, 0.09824532, 0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 5.09240818, 3.46139455, 2.45070267, 1.61558151, 1.08895338,
             0.72133851, 0.45573691, 0.25053367, 0.09824532, 0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.36326075, 1.61558151,
             1.08895338, 0.72133851, 0.45573691, 0.25053367, 0.09824532, 0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.45070267, 1.72759056,
             1.24153244, 0.86115354, 0.59516323, 0.38853383, 0.22545385, 0.09824532, 0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.65472794, 3.60512662, 2.84484982, 2.19988537,
             1.61558151, 1.162866, 0.83188516, 0.59516323, 0.38853383, 0.22545385, 0.09824532, 0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.65472794, 3.60512662, 2.84484982, 2.19988537,
             1.67050016, 1.28281462, 0.95350921, 0.72133851, 0.52423614, 0.34370604, 0.19894916, 0.09824532,
             0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.65472794, 3.60512662, 2.95596409, 2.36326075,
             1.84880662, 1.41535246, 1.08895338, 0.83188516, 0.61951244, 0.45573691, 0.32104823, 0.19894916, 0.09824532,
             0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.65472794, 3.60512662, 2.95596409, 2.45070267,
             1.91321158, 1.51179266, 1.20157266, 0.95350921, 0.74807048, 0.57119018, 0.43325692, 0.29807833, 0.19894916,
             0.09824532, 0.02916753],
            [14.61464119, 11.54541874, 8.30717278, 7.11996698, 5.85520077, 4.65472794, 3.60512662, 2.95596409,
             2.45070267, 1.91321158, 1.51179266, 1.20157266, 0.95350921, 0.74807048, 0.57119018, 0.43325692, 0.29807833,
             0.19894916, 0.09824532, 0.02916753],
            [14.61464119, 11.54541874, 8.30717278, 7.11996698, 5.85520077, 4.65472794, 3.60512662, 2.95596409,
             2.45070267, 1.98035145, 1.61558151, 1.32549286, 1.08895338, 0.86115354, 0.69515091, 0.54755926, 0.41087446,
             0.29807833, 0.19894916, 0.09824532, 0.02916753],
        ],
        1.10: [
            [14.61464119, 0.89115214, 0.02916753],
            [14.61464119, 2.36326075, 0.72133851, 0.02916753],
            [14.61464119, 5.85520077, 1.61558151, 0.57119018, 0.02916753],
            [14.61464119, 6.77309084, 2.45070267, 1.08895338, 0.45573691, 0.02916753],
            [14.61464119, 6.77309084, 2.95596409, 1.56271636, 0.803307, 0.34370604, 0.02916753],
            [14.61464119, 6.77309084, 3.07277966, 1.61558151, 0.89115214, 0.4783645, 0.19894916, 0.02916753],
            [14.61464119, 6.77309084, 3.07277966, 1.84880662, 1.08895338, 0.64427125, 0.34370604, 0.13792117,
             0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 2.84484982, 1.61558151, 0.95350921, 0.54755926, 0.27464288,
             0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 2.95596409, 1.91321158, 1.24153244, 0.803307, 0.4783645, 0.25053367,
             0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 3.07277966, 2.05039096, 1.41535246, 0.95350921, 0.64427125,
             0.41087446, 0.22545385, 0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 3.1956799, 2.27973175, 1.61558151, 1.12534678, 0.803307, 0.54755926,
             0.36617002, 0.22545385, 0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 3.32507086, 2.45070267, 1.72759056, 1.24153244, 0.89115214,
             0.64427125, 0.45573691, 0.32104823, 0.19894916, 0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 5.09240818, 3.60512662, 2.84484982, 2.05039096, 1.51179266, 1.08895338, 0.803307,
             0.59516323, 0.43325692, 0.29807833, 0.19894916, 0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 5.09240818, 3.60512662, 2.84484982, 2.12350607, 1.61558151, 1.24153244,
             0.95350921, 0.72133851, 0.54755926, 0.41087446, 0.29807833, 0.19894916, 0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.45070267, 1.84880662, 1.41535246, 1.08895338,
             0.83188516, 0.64427125, 0.50118381, 0.36617002, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.45070267, 1.91321158, 1.51179266, 1.20157266,
             0.95350921, 0.74807048, 0.59516323, 0.45573691, 0.34370604, 0.25053367, 0.17026083, 0.09824532,
             0.02916753],
            [14.61464119, 7.49001646, 5.85520077, 4.45427561, 3.46139455, 2.84484982, 2.19988537, 1.72759056,
             1.36964464, 1.08895338, 0.86115354, 0.69515091, 0.54755926, 0.43325692, 0.34370604, 0.25053367, 0.17026083,
             0.09824532, 0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.45427561, 3.46139455, 2.84484982, 2.19988537,
             1.72759056, 1.36964464, 1.08895338, 0.86115354, 0.69515091, 0.54755926, 0.43325692, 0.34370604, 0.25053367,
             0.17026083, 0.09824532, 0.02916753],
            [14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.45427561, 3.46139455, 2.84484982, 2.19988537,
             1.72759056, 1.36964464, 1.08895338, 0.89115214, 0.72133851, 0.59516323, 0.4783645, 0.38853383, 0.29807833,
             0.22545385, 0.17026083, 0.09824532, 0.02916753],
        ],
        1.15: [
            [14.61464119, 0.83188516, 0.02916753],
            [14.61464119, 1.84880662, 0.59516323, 0.02916753],
            [14.61464119, 5.85520077, 1.56271636, 0.52423614, 0.02916753],
            [14.61464119, 5.85520077, 1.91321158, 0.83188516, 0.34370604, 0.02916753],
            [14.61464119, 5.85520077, 2.45070267, 1.24153244, 0.59516323, 0.25053367, 0.02916753],
            [14.61464119, 5.85520077, 2.84484982, 1.51179266, 0.803307, 0.41087446, 0.17026083, 0.02916753],
            [14.61464119, 5.85520077, 2.84484982, 1.56271636, 0.89115214, 0.50118381, 0.25053367, 0.09824532,
             0.02916753],
            [14.61464119, 6.77309084, 3.07277966, 1.84880662, 1.12534678, 0.72133851, 0.43325692, 0.22545385,
             0.09824532, 0.02916753],
            [14.61464119, 6.77309084, 3.07277966, 1.91321158, 1.24153244, 0.803307, 0.52423614, 0.34370604, 0.19894916,
             0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 2.95596409, 1.91321158, 1.24153244, 0.803307, 0.52423614, 0.34370604,
             0.19894916, 0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 3.07277966, 2.05039096, 1.36964464, 0.95350921, 0.69515091, 0.4783645,
             0.32104823, 0.19894916, 0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 3.07277966, 2.12350607, 1.51179266, 1.08895338, 0.803307, 0.59516323,
             0.43325692, 0.29807833, 0.19894916, 0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 3.07277966, 2.12350607, 1.51179266, 1.08895338, 0.803307, 0.59516323,
             0.45573691, 0.34370604, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 3.07277966, 2.19988537, 1.61558151, 1.24153244, 0.95350921,
             0.74807048, 0.59516323, 0.45573691, 0.34370604, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 3.1956799, 2.45070267, 1.78698075, 1.32549286, 1.01931262, 0.803307,
             0.64427125, 0.50118381, 0.38853383, 0.29807833, 0.22545385, 0.17026083, 0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 3.1956799, 2.45070267, 1.78698075, 1.32549286, 1.01931262, 0.803307,
             0.64427125, 0.52423614, 0.41087446, 0.32104823, 0.25053367, 0.19894916, 0.13792117, 0.09824532,
             0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 3.1956799, 2.45070267, 1.84880662, 1.41535246, 1.12534678, 0.89115214,
             0.72133851, 0.59516323, 0.4783645, 0.38853383, 0.32104823, 0.25053367, 0.19894916, 0.13792117, 0.09824532,
             0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 3.1956799, 2.45070267, 1.84880662, 1.41535246, 1.12534678, 0.89115214,
             0.72133851, 0.59516323, 0.50118381, 0.41087446, 0.34370604, 0.27464288, 0.22545385, 0.17026083, 0.13792117,
             0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 4.86714602, 3.1956799, 2.45070267, 1.84880662, 1.41535246, 1.12534678, 0.89115214,
             0.72133851, 0.59516323, 0.50118381, 0.41087446, 0.34370604, 0.29807833, 0.25053367, 0.19894916, 0.17026083,
             0.13792117, 0.09824532, 0.02916753],
        ],
        1.20: [
            [14.61464119, 0.803307, 0.02916753],
            [14.61464119, 1.56271636, 0.52423614, 0.02916753],
            [14.61464119, 2.36326075, 0.92192322, 0.36617002, 0.02916753],
            [14.61464119, 2.84484982, 1.24153244, 0.59516323, 0.25053367, 0.02916753],
            [14.61464119, 5.85520077, 2.05039096, 0.95350921, 0.45573691, 0.17026083, 0.02916753],
            [14.61464119, 5.85520077, 2.45070267, 1.24153244, 0.64427125, 0.29807833, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.45070267, 1.36964464, 0.803307, 0.45573691, 0.25053367, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.84484982, 1.61558151, 0.95350921, 0.59516323, 0.36617002, 0.19894916,
             0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.84484982, 1.67050016, 1.08895338, 0.74807048, 0.50118381, 0.32104823,
             0.19894916, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.95596409, 1.84880662, 1.24153244, 0.83188516, 0.59516323, 0.41087446,
             0.27464288, 0.17026083, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 3.07277966, 1.98035145, 1.36964464, 0.95350921, 0.69515091, 0.50118381,
             0.36617002, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
            [14.61464119, 6.77309084, 3.46139455, 2.36326075, 1.56271636, 1.08895338, 0.803307, 0.59516323, 0.45573691,
             0.34370604, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
            [14.61464119, 6.77309084, 3.46139455, 2.45070267, 1.61558151, 1.162866, 0.86115354, 0.64427125, 0.50118381,
             0.38853383, 0.29807833, 0.22545385, 0.17026083, 0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 4.65472794, 3.07277966, 2.12350607, 1.51179266, 1.08895338, 0.83188516,
             0.64427125, 0.50118381, 0.38853383, 0.29807833, 0.22545385, 0.17026083, 0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 4.65472794, 3.07277966, 2.12350607, 1.51179266, 1.08895338, 0.83188516,
             0.64427125, 0.50118381, 0.41087446, 0.32104823, 0.25053367, 0.19894916, 0.13792117, 0.09824532,
             0.02916753],
            [14.61464119, 7.49001646, 4.65472794, 3.07277966, 2.12350607, 1.51179266, 1.08895338, 0.83188516,
             0.64427125, 0.50118381, 0.41087446, 0.34370604, 0.27464288, 0.22545385, 0.17026083, 0.13792117, 0.09824532,
             0.02916753],
            [14.61464119, 7.49001646, 4.65472794, 3.07277966, 2.19988537, 1.61558151, 1.20157266, 0.92192322,
             0.72133851, 0.57119018, 0.45573691, 0.36617002, 0.29807833, 0.25053367, 0.19894916, 0.17026083, 0.13792117,
             0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 4.65472794, 3.07277966, 2.19988537, 1.61558151, 1.24153244, 0.95350921,
             0.74807048, 0.59516323, 0.4783645, 0.38853383, 0.32104823, 0.27464288, 0.22545385, 0.19894916, 0.17026083,
             0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 7.49001646, 4.65472794, 3.07277966, 2.19988537, 1.61558151, 1.24153244, 0.95350921,
             0.74807048, 0.59516323, 0.50118381, 0.41087446, 0.34370604, 0.29807833, 0.25053367, 0.22545385, 0.19894916,
             0.17026083, 0.13792117, 0.09824532, 0.02916753],
        ],
        1.25: [
            [14.61464119, 0.72133851, 0.02916753],
            [14.61464119, 1.56271636, 0.50118381, 0.02916753],
            [14.61464119, 2.05039096, 0.803307, 0.32104823, 0.02916753],
            [14.61464119, 2.36326075, 0.95350921, 0.43325692, 0.17026083, 0.02916753],
            [14.61464119, 2.84484982, 1.24153244, 0.59516323, 0.27464288, 0.09824532, 0.02916753],
            [14.61464119, 3.07277966, 1.51179266, 0.803307, 0.43325692, 0.22545385, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.36326075, 1.24153244, 0.72133851, 0.41087446, 0.22545385, 0.09824532,
             0.02916753],
            [14.61464119, 5.85520077, 2.45070267, 1.36964464, 0.83188516, 0.52423614, 0.34370604, 0.19894916,
             0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.84484982, 1.61558151, 0.98595673, 0.64427125, 0.43325692, 0.27464288,
             0.17026083, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.84484982, 1.67050016, 1.08895338, 0.74807048, 0.52423614, 0.36617002,
             0.25053367, 0.17026083, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.84484982, 1.72759056, 1.162866, 0.803307, 0.59516323, 0.45573691, 0.34370604,
             0.25053367, 0.17026083, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.95596409, 1.84880662, 1.24153244, 0.86115354, 0.64427125, 0.4783645, 0.36617002,
             0.27464288, 0.19894916, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.95596409, 1.84880662, 1.28281462, 0.92192322, 0.69515091, 0.52423614,
             0.41087446, 0.32104823, 0.25053367, 0.19894916, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.95596409, 1.91321158, 1.32549286, 0.95350921, 0.72133851, 0.54755926,
             0.43325692, 0.34370604, 0.27464288, 0.22545385, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.95596409, 1.91321158, 1.32549286, 0.95350921, 0.72133851, 0.57119018,
             0.45573691, 0.36617002, 0.29807833, 0.25053367, 0.19894916, 0.17026083, 0.13792117, 0.09824532,
             0.02916753],
            [14.61464119, 5.85520077, 2.95596409, 1.91321158, 1.32549286, 0.95350921, 0.74807048, 0.59516323, 0.4783645,
             0.38853383, 0.32104823, 0.27464288, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532,
             0.02916753],
            [14.61464119, 5.85520077, 3.07277966, 2.05039096, 1.41535246, 1.05362725, 0.803307, 0.61951244, 0.50118381,
             0.41087446, 0.34370604, 0.29807833, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532,
             0.02916753],
            [14.61464119, 5.85520077, 3.07277966, 2.05039096, 1.41535246, 1.05362725, 0.803307, 0.64427125, 0.52423614,
             0.43325692, 0.36617002, 0.32104823, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117,
             0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 3.07277966, 2.05039096, 1.46270394, 1.08895338, 0.83188516, 0.66947293,
             0.54755926, 0.45573691, 0.38853383, 0.34370604, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916,
             0.17026083, 0.13792117, 0.09824532, 0.02916753],
        ],
        1.30: [
            [14.61464119, 0.72133851, 0.02916753],
            [14.61464119, 1.24153244, 0.43325692, 0.02916753],
            [14.61464119, 1.56271636, 0.59516323, 0.22545385, 0.02916753],
            [14.61464119, 1.84880662, 0.803307, 0.36617002, 0.13792117, 0.02916753],
            [14.61464119, 2.36326075, 1.01931262, 0.52423614, 0.25053367, 0.09824532, 0.02916753],
            [14.61464119, 2.84484982, 1.36964464, 0.74807048, 0.41087446, 0.22545385, 0.09824532, 0.02916753],
            [14.61464119, 3.07277966, 1.56271636, 0.89115214, 0.54755926, 0.34370604, 0.19894916, 0.09824532,
             0.02916753],
            [14.61464119, 3.07277966, 1.61558151, 0.95350921, 0.61951244, 0.41087446, 0.27464288, 0.17026083,
             0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.45070267, 1.36964464, 0.83188516, 0.54755926, 0.36617002, 0.25053367,
             0.17026083, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.45070267, 1.41535246, 0.92192322, 0.64427125, 0.45573691, 0.34370604,
             0.25053367, 0.17026083, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.6383388, 1.56271636, 1.01931262, 0.72133851, 0.50118381, 0.36617002, 0.27464288,
             0.19894916, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.84484982, 1.61558151, 1.05362725, 0.74807048, 0.54755926, 0.41087446,
             0.32104823, 0.25053367, 0.19894916, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.84484982, 1.61558151, 1.08895338, 0.77538133, 0.57119018, 0.43325692,
             0.34370604, 0.27464288, 0.22545385, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.84484982, 1.61558151, 1.08895338, 0.803307, 0.59516323, 0.45573691, 0.36617002,
             0.29807833, 0.25053367, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.84484982, 1.61558151, 1.08895338, 0.803307, 0.59516323, 0.4783645, 0.38853383,
             0.32104823, 0.27464288, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.84484982, 1.72759056, 1.162866, 0.83188516, 0.64427125, 0.50118381, 0.41087446,
             0.34370604, 0.29807833, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532,
             0.02916753],
            [14.61464119, 5.85520077, 2.84484982, 1.72759056, 1.162866, 0.83188516, 0.64427125, 0.52423614, 0.43325692,
             0.36617002, 0.32104823, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532,
             0.02916753],
            [14.61464119, 5.85520077, 2.84484982, 1.78698075, 1.24153244, 0.92192322, 0.72133851, 0.57119018,
             0.45573691, 0.38853383, 0.34370604, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083,
             0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.84484982, 1.78698075, 1.24153244, 0.92192322, 0.72133851, 0.57119018, 0.4783645,
             0.41087446, 0.36617002, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083,
             0.13792117, 0.09824532, 0.02916753],
        ],
        1.35: [
            [14.61464119, 0.69515091, 0.02916753],
            [14.61464119, 0.95350921, 0.34370604, 0.02916753],
            [14.61464119, 1.56271636, 0.57119018, 0.19894916, 0.02916753],
            [14.61464119, 1.61558151, 0.69515091, 0.29807833, 0.09824532, 0.02916753],
            [14.61464119, 1.84880662, 0.83188516, 0.43325692, 0.22545385, 0.09824532, 0.02916753],
            [14.61464119, 2.45070267, 1.162866, 0.64427125, 0.36617002, 0.19894916, 0.09824532, 0.02916753],
            [14.61464119, 2.84484982, 1.36964464, 0.803307, 0.50118381, 0.32104823, 0.19894916, 0.09824532, 0.02916753],
            [14.61464119, 2.84484982, 1.41535246, 0.83188516, 0.54755926, 0.36617002, 0.25053367, 0.17026083,
             0.09824532, 0.02916753],
            [14.61464119, 2.84484982, 1.56271636, 0.95350921, 0.64427125, 0.45573691, 0.32104823, 0.22545385,
             0.17026083, 0.09824532, 0.02916753],
            [14.61464119, 2.84484982, 1.56271636, 0.95350921, 0.64427125, 0.45573691, 0.34370604, 0.25053367,
             0.19894916, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 3.07277966, 1.61558151, 1.01931262, 0.72133851, 0.52423614, 0.38853383, 0.29807833,
             0.22545385, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 3.07277966, 1.61558151, 1.01931262, 0.72133851, 0.52423614, 0.41087446, 0.32104823,
             0.25053367, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 3.07277966, 1.61558151, 1.05362725, 0.74807048, 0.54755926, 0.43325692, 0.34370604,
             0.27464288, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 3.07277966, 1.72759056, 1.12534678, 0.803307, 0.59516323, 0.45573691, 0.36617002, 0.29807833,
             0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 3.07277966, 1.72759056, 1.12534678, 0.803307, 0.59516323, 0.4783645, 0.38853383, 0.32104823,
             0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.45070267, 1.51179266, 1.01931262, 0.74807048, 0.57119018, 0.45573691,
             0.36617002, 0.32104823, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532,
             0.02916753],
            [14.61464119, 5.85520077, 2.6383388, 1.61558151, 1.08895338, 0.803307, 0.61951244, 0.50118381, 0.41087446,
             0.34370604, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532,
             0.02916753],
            [14.61464119, 5.85520077, 2.6383388, 1.61558151, 1.08895338, 0.803307, 0.64427125, 0.52423614, 0.43325692,
             0.36617002, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117,
             0.09824532, 0.02916753],
            [14.61464119, 5.85520077, 2.6383388, 1.61558151, 1.08895338, 0.803307, 0.64427125, 0.52423614, 0.45573691,
             0.38853383, 0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083,
             0.13792117, 0.09824532, 0.02916753],
        ],
        1.40: [
            [14.61464119, 0.59516323, 0.02916753],
            [14.61464119, 0.95350921, 0.34370604, 0.02916753],
            [14.61464119, 1.08895338, 0.43325692, 0.13792117, 0.02916753],
            [14.61464119, 1.56271636, 0.64427125, 0.27464288, 0.09824532, 0.02916753],
            [14.61464119, 1.61558151, 0.803307, 0.43325692, 0.22545385, 0.09824532, 0.02916753],
            [14.61464119, 2.05039096, 0.95350921, 0.54755926, 0.34370604, 0.19894916, 0.09824532, 0.02916753],
            [14.61464119, 2.45070267, 1.24153244, 0.72133851, 0.43325692, 0.27464288, 0.17026083, 0.09824532,
             0.02916753],
            [14.61464119, 2.45070267, 1.24153244, 0.74807048, 0.50118381, 0.34370604, 0.25053367, 0.17026083,
             0.09824532, 0.02916753],
            [14.61464119, 2.45070267, 1.28281462, 0.803307, 0.52423614, 0.36617002, 0.27464288, 0.19894916, 0.13792117,
             0.09824532, 0.02916753],
            [14.61464119, 2.45070267, 1.28281462, 0.803307, 0.54755926, 0.38853383, 0.29807833, 0.22545385, 0.17026083,
             0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 2.84484982, 1.41535246, 0.86115354, 0.59516323, 0.43325692, 0.32104823, 0.25053367,
             0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 2.84484982, 1.51179266, 0.95350921, 0.64427125, 0.45573691, 0.34370604, 0.27464288,
             0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 2.84484982, 1.51179266, 0.95350921, 0.64427125, 0.4783645, 0.36617002, 0.29807833, 0.25053367,
             0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 2.84484982, 1.56271636, 0.98595673, 0.69515091, 0.52423614, 0.41087446, 0.34370604,
             0.29807833, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 2.84484982, 1.56271636, 1.01931262, 0.72133851, 0.54755926, 0.43325692, 0.36617002,
             0.32104823, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532,
             0.02916753],
            [14.61464119, 2.84484982, 1.61558151, 1.05362725, 0.74807048, 0.57119018, 0.45573691, 0.38853383,
             0.34370604, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532,
             0.02916753],
            [14.61464119, 2.84484982, 1.61558151, 1.08895338, 0.803307, 0.61951244, 0.50118381, 0.41087446, 0.36617002,
             0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532,
             0.02916753],
            [14.61464119, 2.84484982, 1.61558151, 1.08895338, 0.803307, 0.61951244, 0.50118381, 0.43325692, 0.38853383,
             0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117,
             0.09824532, 0.02916753],
            [14.61464119, 2.84484982, 1.61558151, 1.08895338, 0.803307, 0.64427125, 0.52423614, 0.45573691, 0.41087446,
             0.36617002, 0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083,
             0.13792117, 0.09824532, 0.02916753],
        ],
        1.45: [
            [14.61464119, 0.59516323, 0.02916753],
            [14.61464119, 0.803307, 0.25053367, 0.02916753],
            [14.61464119, 0.95350921, 0.34370604, 0.09824532, 0.02916753],
            [14.61464119, 1.24153244, 0.54755926, 0.25053367, 0.09824532, 0.02916753],
            [14.61464119, 1.56271636, 0.72133851, 0.36617002, 0.19894916, 0.09824532, 0.02916753],
            [14.61464119, 1.61558151, 0.803307, 0.45573691, 0.27464288, 0.17026083, 0.09824532, 0.02916753],
            [14.61464119, 1.91321158, 0.95350921, 0.57119018, 0.36617002, 0.25053367, 0.17026083, 0.09824532,
             0.02916753],
            [14.61464119, 2.19988537, 1.08895338, 0.64427125, 0.41087446, 0.27464288, 0.19894916, 0.13792117,
             0.09824532, 0.02916753],
            [14.61464119, 2.45070267, 1.24153244, 0.74807048, 0.50118381, 0.34370604, 0.25053367, 0.19894916,
             0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 2.45070267, 1.24153244, 0.74807048, 0.50118381, 0.36617002, 0.27464288, 0.22545385,
             0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 2.45070267, 1.28281462, 0.803307, 0.54755926, 0.41087446, 0.32104823, 0.25053367, 0.19894916,
             0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 2.45070267, 1.28281462, 0.803307, 0.57119018, 0.43325692, 0.34370604, 0.27464288, 0.22545385,
             0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 2.45070267, 1.28281462, 0.83188516, 0.59516323, 0.45573691, 0.36617002, 0.29807833,
             0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 2.45070267, 1.28281462, 0.83188516, 0.59516323, 0.45573691, 0.36617002, 0.32104823,
             0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 2.84484982, 1.51179266, 0.95350921, 0.69515091, 0.52423614, 0.41087446, 0.34370604,
             0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532,
             0.02916753],
            [14.61464119, 2.84484982, 1.51179266, 0.95350921, 0.69515091, 0.52423614, 0.43325692, 0.36617002,
             0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532,
             0.02916753],
            [14.61464119, 2.84484982, 1.56271636, 0.98595673, 0.72133851, 0.54755926, 0.45573691, 0.38853383,
             0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117,
             0.09824532, 0.02916753],
            [14.61464119, 2.84484982, 1.56271636, 1.01931262, 0.74807048, 0.57119018, 0.4783645, 0.41087446, 0.36617002,
             0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117,
             0.09824532, 0.02916753],
            [14.61464119, 2.84484982, 1.56271636, 1.01931262, 0.74807048, 0.59516323, 0.50118381, 0.43325692,
             0.38853383, 0.36617002, 0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916,
             0.17026083, 0.13792117, 0.09824532, 0.02916753],
        ],
        1.50: [
            [14.61464119, 0.54755926, 0.02916753],
            [14.61464119, 0.803307, 0.25053367, 0.02916753],
            [14.61464119, 0.86115354, 0.32104823, 0.09824532, 0.02916753],
            [14.61464119, 1.24153244, 0.54755926, 0.25053367, 0.09824532, 0.02916753],
            [14.61464119, 1.56271636, 0.72133851, 0.36617002, 0.19894916, 0.09824532, 0.02916753],
            [14.61464119, 1.61558151, 0.803307, 0.45573691, 0.27464288, 0.17026083, 0.09824532, 0.02916753],
            [14.61464119, 1.61558151, 0.83188516, 0.52423614, 0.34370604, 0.25053367, 0.17026083, 0.09824532,
             0.02916753],
            [14.61464119, 1.84880662, 0.95350921, 0.59516323, 0.38853383, 0.27464288, 0.19894916, 0.13792117,
             0.09824532, 0.02916753],
            [14.61464119, 1.84880662, 0.95350921, 0.59516323, 0.41087446, 0.29807833, 0.22545385, 0.17026083,
             0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 1.84880662, 0.95350921, 0.61951244, 0.43325692, 0.32104823, 0.25053367, 0.19894916,
             0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 2.19988537, 1.12534678, 0.72133851, 0.50118381, 0.36617002, 0.27464288, 0.22545385,
             0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 2.19988537, 1.12534678, 0.72133851, 0.50118381, 0.36617002, 0.29807833, 0.25053367,
             0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 2.36326075, 1.24153244, 0.803307, 0.57119018, 0.43325692, 0.34370604, 0.29807833, 0.25053367,
             0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 2.36326075, 1.24153244, 0.803307, 0.57119018, 0.43325692, 0.34370604, 0.29807833, 0.27464288,
             0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 2.36326075, 1.24153244, 0.803307, 0.59516323, 0.45573691, 0.36617002, 0.32104823, 0.29807833,
             0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 2.36326075, 1.24153244, 0.803307, 0.59516323, 0.45573691, 0.38853383, 0.34370604, 0.32104823,
             0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532,
             0.02916753],
            [14.61464119, 2.45070267, 1.32549286, 0.86115354, 0.64427125, 0.50118381, 0.41087446, 0.36617002,
             0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117,
             0.09824532, 0.02916753],
            [14.61464119, 2.45070267, 1.36964464, 0.92192322, 0.69515091, 0.54755926, 0.45573691, 0.41087446,
             0.36617002, 0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083,
             0.13792117, 0.09824532, 0.02916753],
            [14.61464119, 2.45070267, 1.41535246, 0.95350921, 0.72133851, 0.57119018, 0.4783645, 0.43325692, 0.38853383,
             0.36617002, 0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083,
             0.13792117, 0.09824532, 0.02916753],
        ],
    }

    def get_sigmas(self, coeff, steps, denoise):
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            total_steps = round(steps * denoise)

        if steps <= 20:
            sigmas = self.NOISE_LEVELS[round(coeff, 2)][steps-2][:]
        else:
            sigmas = self.NOISE_LEVELS[round(coeff, 2)][-1][:]
            sigmas = loglinear_interp(sigmas, steps + 1)

        sigmas = sigmas[-(total_steps + 1):]
        sigmas[-1] = 0
        return (torch.FloatTensor(sigmas), )