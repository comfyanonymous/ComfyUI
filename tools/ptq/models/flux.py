"""
FLUX model quantization recipes.

Defines FluxDevRecipe and FluxSchnellRecipe for quantizing FLUX models.
"""
import logging
import comfy.sd
import folder_paths
from dataclasses import dataclass
from typing import Tuple, Callable

from comfy_extras.nodes_custom_sampler import (
    BasicGuider,
    SamplerCustomAdvanced,
    BasicScheduler,
    RandomNoise,
    KSamplerSelect,
)
from comfy_extras.nodes_flux import FluxGuidance
from comfy_extras.nodes_model_advanced import ModelSamplingFlux
from comfy_extras.nodes_sd3 import EmptySD3LatentImage
from nodes import CLIPTextEncode

from . import register_recipe
from .base import ModelRecipe

@dataclass
class SamplerCFG:
    cfg: float
    sampler_name: str
    scheduler: str
    denoise: float
    max_shift: float
    base_shift: float

class FluxT2IPipe:
    def __init__(
            self,
            model,
            clip,
            batch_size,
            width=1024,
            height=1024,
            seed=0,
            sampler_cfg: SamplerCFG = None,
            device="cuda",
    ) -> None:
        self.clip = clip
        self.clip_node = CLIPTextEncode()
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.max_shift = sampler_cfg.max_shift
        self.base_shift = sampler_cfg.base_shift
        self.device = device

        self.seed = seed
        self.cfg = sampler_cfg.cfg
        self.scheduler_name = sampler_cfg.scheduler
        self.denoise = sampler_cfg.denoise

        (self.model,) = ModelSamplingFlux().patch(
            model, self.max_shift, self.base_shift, self.width, self.height
        )
        (self.ksampler,) = KSamplerSelect().get_sampler(sampler_cfg.sampler_name)
        self.latent_node = EmptySD3LatentImage()
        self.guidance = FluxGuidance()
        self.sampler = SamplerCustomAdvanced()
        self.scheduler_node = BasicScheduler()
        self.guider = BasicGuider()
        self.noise_generator = RandomNoise()

    def __call__(self, num_inference_steps, positive_prompt, *args, **kwargs):
        (positive,) = self.clip_node.encode(self.clip, positive_prompt)
        (latent_image,) = self.latent_node.generate(
            self.width, self.height, self.batch_size
        )
        (noise,) = self.noise_generator.get_noise(self.seed)

        (conditioning,) = self.guidance.append(positive, self.cfg)
        (sigmas,) = self.scheduler_node.get_sigmas(
            self.model, self.scheduler_name, num_inference_steps, self.denoise
        )
        (guider,) = self.guider.get_guider(self.model, conditioning)

        out, denoised_out = self.sampler.sample(
            noise, guider, self.ksampler, sigmas, latent_image
        )

        return out["samples"]

class FluxRecipeBase(ModelRecipe):
    """Base class for FLUX model quantization recipes."""

    @classmethod
    def add_model_args(cls, parser):
        """Add FLUX-specific CLI arguments."""
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "--ckpt_path",
            help="Path to full FLUX checkpoint (includes diffusion model + CLIP + T5)"
        )
        group.add_argument(
            "--unet_path",
            help="Path to FLUX diffusion model only (requires --clip_path and --t5_path)"
        )

        parser.add_argument(
            "--clip_path",
            help="Path to CLIP text encoder (required with --unet_path)"
        )
        parser.add_argument(
            "--t5_path",
            help="Path to T5 text encoder (required with --unet_path)"
        )

    def __init__(self, args):
        """Initialize FLUX recipe with CLI args."""
        self.args = args

        # Validate args
        if hasattr(args, 'unet_path') and args.unet_path:
            if not args.clip_path or not args.t5_path:
                raise ValueError("--unet_path requires both --clip_path and --t5_path")

    def load_model(self) -> Tuple:
        """Load FLUX model, CLIP, and VAE."""
        if hasattr(self.args, 'ckpt_path') and self.args.ckpt_path:
            # Load from full checkpoint
            logging.info(f"Loading full checkpoint from {self.args.ckpt_path}")
            model_patcher, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
                self.args.ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=None
            )
        else:
            # Load from separate files
            logging.info(f"Loading diffusion model from {self.args.unet_path}")
            model_options = {}
            clip_type = comfy.sd.CLIPType.FLUX

            clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", self.args.clip_path)
            clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", self.args.t5_path)

            model_patcher = comfy.sd.load_diffusion_model(
                self.args.unet_path,
                model_options=model_options
            )
            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path1, clip_path2],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type,
                model_options=model_options
            )
            vae = None  # Not needed for calibration

        return model_patcher, clip, vae

    def create_calibration_pipeline(self, model_components):
        """Create FluxT2IPipe for calibration."""
        model_patcher, clip, vae = model_components

        return FluxT2IPipe(
            model=model_patcher,
            clip=clip,
            batch_size=1,
            width=self.get_width(),
            height=self.get_height(),
            seed=42,
            sampler_cfg=self.get_sampler_cfg(),
            device="cuda"
        )

    def get_forward_loop(self, calib_pipeline, dataloader) -> Callable:
        """
        Return forward_loop for ModelOptimizer calibration.

        Iterates through the dataloader and runs full sampling
        for each prompt to collect activation statistics.
        """
        num_steps = self.get_inference_steps()

        def forward_loop():
            for i, prompt in enumerate(dataloader):
                # Dataloader returns batches, extract first element
                prompt_text = prompt[0] if isinstance(prompt, (list, tuple)) else prompt

                logging.debug(f"Calibration step {i+1}: '{prompt_text[:50]}...'")

                try:
                    # Run full sampling pipeline
                    calib_pipeline(num_steps, prompt_text)
                except Exception as e:
                    logging.warning(f"Calibration step {i+1} failed: {e}")
                    # Continue with next prompt

        return forward_loop

    # Abstract methods for variants to implement
    def get_width(self) -> int:
        """Image width for calibration."""
        raise NotImplementedError

    def get_height(self) -> int:
        """Image height for calibration."""
        raise NotImplementedError

    def get_sampler_cfg(self) -> SamplerCFG:
        """Sampler configuration."""
        raise NotImplementedError

    def get_inference_steps(self) -> int:
        """Number of sampling steps per calibration iteration."""
        raise NotImplementedError


@register_recipe
class FluxDevRecipe(FluxRecipeBase):
    """FLUX Dev quantization recipe."""

    @classmethod
    def name(cls) -> str:
        return "flux_dev"

    def get_default_calib_steps(self) -> int:
        return 128

    def get_width(self) -> int:
        return 1024

    def get_height(self) -> int:
        return 1024

    def get_inference_steps(self) -> int:
        return 30

    def get_sampler_cfg(self) -> SamplerCFG:
        return SamplerCFG(
            cfg=3.5,
            sampler_name="euler",
            scheduler="simple",
            denoise=1.0,
            max_shift=1.15,
            base_shift=0.5
        )


@register_recipe
class FluxSchnellRecipe(FluxRecipeBase):
    """FLUX Schnell quantization recipe."""

    @classmethod
    def name(cls) -> str:
        return "flux_schnell"

    def get_default_calib_steps(self) -> int:
        return 64

    def get_width(self) -> int:
        return 1024

    def get_height(self) -> int:
        return 1024

    def get_inference_steps(self) -> int:
        return 4

    def get_sampler_cfg(self) -> SamplerCFG:
        return SamplerCFG(
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            denoise=1.0,
            max_shift=1.15,
            base_shift=0.5
        )
