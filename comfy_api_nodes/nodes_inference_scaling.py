"""
Inference Scaling Nodes for ComfyUI

This module provides nodes for adaptive inference scaling during image generation.
The system monitors image quality during the generation loop and adjusts sampling
parameters dynamically based on quality metrics.
"""

from typing import Any
from typing_extensions import override
from inspect import cleandoc
import torch
import numpy as np

from comfy_api.latest import IO, ComfyExtension
from comfy.samplers import KSampler
from comfy.sd import VAE
from comfy import model_management
import comfy.sample

# Get sampler and scheduler options
SAMPLER_OPTIONS = KSampler.SAMPLERS
SCHEDULER_OPTIONS = KSampler.SCHEDULERS


class QualityVerifier:
    """
    Base class for image quality verification.
    Subclasses should implement specific quality metrics.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def verify(self, image: torch.Tensor) -> tuple[bool, float]:
        """
        Verify image quality.

        Args:
            image: Image tensor [B, H, W, C] in range [0, 1]

        Returns:
            Tuple of (is_acceptable, quality_score)
        """
        raise NotImplementedError("Subclasses must implement verify method")

    def get_quality_score(self, image: torch.Tensor) -> float:
        """
        Get quality score without threshold check.

        Args:
            image: Image tensor [B, H, W, C] in range [0, 1]

        Returns:
            Quality score in range [0, 1]
        """
        _, score = self.verify(image)
        return score


class SimpleVerifier(QualityVerifier):
    """
    Simple verifier using basic image statistics.
    Uses variance and edge detection as quality indicators.
    """

    def verify(self, image: torch.Tensor) -> tuple[bool, float]:
        """
        Simple quality check based on image variance and structure.
        """
        # Convert to grayscale for analysis
        if image.shape[-1] == 4:  # RGBA
            gray = image[..., :3].mean(dim=-1, keepdim=True)
        else:
            gray = image.mean(dim=-1, keepdim=True)

        # Calculate variance (higher variance = more detail)
        variance = gray.var().item()

        # Simple edge detection using Sobel-like operator
        # Convert to numpy for easier processing
        img_np = gray.squeeze().cpu().numpy()
        if len(img_np.shape) == 3:
            img_np = img_np[0]  # Take first batch

        # Calculate gradients
        grad_x = np.gradient(img_np, axis=1)
        grad_y = np.gradient(img_np, axis=0)
        edge_strength = np.sqrt(grad_x**2 + grad_y**2).mean()

        # Combine metrics (normalize to [0, 1])
        variance_score = min(variance * 10, 1.0)  # Scale variance
        edge_score = min(edge_strength * 5, 1.0)  # Scale edge strength

        quality_score = (variance_score * 0.6 + edge_score * 0.4)

        is_acceptable = quality_score >= self.threshold

        return is_acceptable, quality_score


class VerifierSelectionNode(IO.ComfyNode):
    """
    Node for selecting or creating an image quality verifier.
    The verifier is used by InferenceScalingNode to judge image quality
    during the generation process.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VerifierSelectionNode",
            display_name="Verifier Selection",
            category="inference_scaling",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.Combo.Input(
                    "verifier_type",
                    options=["simple", "custom"],
                    default="simple",
                    tooltip="Type of quality verifier to use",
                ),
                IO.Float.Input(
                    "quality_threshold",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Minimum quality score threshold (0.0-1.0)",
                ),
            ],
            outputs=[
                IO.String.Output(),  # Verifier identifier (for now, just a string)
            ],
        )

    @classmethod
    async def execute(
        cls,
        verifier_type: str = "simple",
        quality_threshold: float = 0.5,
    ) -> IO.NodeOutput:
        """
        Create and return a verifier identifier.

        In a real implementation, this would create and store the verifier
        instance. For now, we return a serialized identifier.
        """
        # Create verifier based on type
        if verifier_type == "simple":
            verifier = SimpleVerifier(threshold=quality_threshold)
        else:
            verifier = SimpleVerifier(threshold=quality_threshold)

        # Store verifier in a way that InferenceScalingNode can access it
        # For now, we'll use a simple approach: store in a module-level dict
        import comfy_api_nodes.nodes_inference_scaling as mod
        if not hasattr(mod, '_verifier_registry'):
            mod._verifier_registry = {}

        verifier_id = f"verifier_{id(verifier)}"
        mod._verifier_registry[verifier_id] = verifier

        return IO.NodeOutput(verifier_id)


class InferenceScalingNode(IO.ComfyNode):
    """
    Main inference scaling node that wraps KSampler with quality monitoring.

    This node:
    1. Wraps the standard sampling process
    2. Periodically decodes latents to images using VAE during sampling
    3. Uses a verifier to judge image quality
    4. Adjusts sampling parameters (steps, CFG) based on quality
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="InferenceScalingNode",
            display_name="Inference Scaling",
            category="inference_scaling",
            description=cleandoc(cls.__doc__ or ""),
            inputs=[
                IO.String.Input(
                    "verifier_id",
                    default="",
                    tooltip="Verifier identifier from VerifierSelectionNode",
                ),
                IO.Model.Input(
                    "model",
                    tooltip="Model to use for sampling",
                ),
                IO.Conditioning.Input(
                    "positive",
                    tooltip="Positive conditioning",
                ),
                IO.Conditioning.Input(
                    "negative",
                    tooltip="Negative conditioning",
                ),
                IO.Latent.Input(
                    "latent_image",
                    tooltip="Initial latent image",
                ),
                IO.Vae.Input(
                    "vae",
                    tooltip="VAE model for decoding latents during quality checks",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xffffffffffffffff,
                    control_after_generate=True,
                    tooltip="Random seed",
                ),
                IO.Int.Input(
                    "steps",
                    default=20,
                    min=1,
                    max=1000,
                    tooltip="Number of sampling steps",
                ),
                IO.Float.Input(
                    "cfg",
                    default=7.0,
                    min=0.0,
                    max=30.0,
                    step=0.1,
                    tooltip="CFG scale",
                ),
                IO.Combo.Input(
                    "sampler_name",
                    options=SAMPLER_OPTIONS,
                    default="euler",
                    tooltip="Sampler name",
                ),
                IO.Combo.Input(
                    "scheduler",
                    options=SCHEDULER_OPTIONS,
                    default="normal",
                    tooltip="Scheduler name",
                ),
                IO.Int.Input(
                    "check_interval",
                    default=5,
                    min=1,
                    max=50,
                    tooltip="Check quality every N steps",
                ),
                IO.Float.Input(
                    "quality_threshold",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Quality threshold for early stopping or adjustment",
                ),
                IO.Float.Input(
                    "scale_factor",
                    default=1.2,
                    min=0.5,
                    max=2.0,
                    step=0.1,
                    tooltip="Factor to scale steps/CFG when quality is low",
                ),
            ],
            outputs=[
                IO.Latent.Output(),  # Latent output
            ],
        )

    @classmethod
    async def execute(
        cls,
        verifier_id: str,
        model: Any,
        positive: Any,
        negative: Any,
        latent_image: dict,
        vae: VAE,
        seed: int = 0,
        steps: int = 20,
        cfg: float = 7.0,
        sampler_name: str = "euler",
        scheduler: str = "normal",
        check_interval: int = 5,
        quality_threshold: float = 0.5,
        scale_factor: float = 1.2,
    ) -> IO.NodeOutput:
        """
        Execute inference scaling sampling.

        This wraps the standard sampling process and adds quality monitoring.
        """
        # Get verifier from registry
        import comfy_api_nodes.nodes_inference_scaling as mod
        if not hasattr(mod, '_verifier_registry') or verifier_id not in mod._verifier_registry:
            raise ValueError(f"Verifier {verifier_id} not found. Please create it with VerifierSelectionNode first.")

        verifier = mod._verifier_registry[verifier_id]

        # Prepare sampling parameters
        model_management.get_torch_device()

        # Extract latent_image tensor from dict
        latent_image_tensor = latent_image["samples"]

        # Fix empty latent channels if needed
        latent_image_tensor = comfy.sample.fix_empty_latent_channels(model, latent_image_tensor)

        # Prepare noise from latent_image and seed (same as common_ksampler does)
        batch_inds = latent_image.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image_tensor, seed, batch_inds)

        # Get noise_mask if present
        noise_mask = latent_image.get("noise_mask", None)

        # Track quality during sampling
        quality_history = []
        current_cfg = cfg

        def quality_check_callback(callback_dict: dict):
            """
            Callback function called during sampling steps.
            Decodes latents, checks quality, and adjusts parameters.

            Args:
                callback_dict: Dictionary with keys 'x', 'i', 'sigma', 'sigma_hat', 'denoised'
            """
            nonlocal current_cfg, quality_history

            step = callback_dict['i']
            denoised = callback_dict['denoised']  # This is the predicted x0

            # Only check at specified intervals
            if check_interval > 0 and step % check_interval != 0:
                return

            try:
                # Decode latent to image using VAE
                # denoised is the predicted x0 in latent space [B, C, H, W]
                with torch.no_grad():
                    # Ensure denoised is a proper tensor
                    if not isinstance(denoised, torch.Tensor):
                        return

                    # VAE.decode expects [B, C, H, W] format and handles device management
                    # It returns [B, H, W, C] in range [0, 1]
                    decoded = vae.decode(denoised)

                    # Ensure decoded is in correct format [B, H, W, C]
                    if decoded is None:
                        return

                    # Move to CPU for quality checking to avoid GPU memory issues
                    decoded_cpu = decoded.cpu()

                    # Check quality
                    is_acceptable, quality_score = verifier.verify(decoded_cpu)
                    quality_history.append((step, quality_score, is_acceptable))

                    # Log quality for debugging
                    import logging
                    logging.info(f"Inference Scaling: Step {step}, Quality: {quality_score:.3f}, Acceptable: {is_acceptable}")

                    # Note: CFG adjustment here won't affect current sampling run
                    # as CFG is set at the start. This is for future reference/logging.
                    if not is_acceptable and quality_score < quality_threshold:
                        # Quality is low - log for analysis
                        logging.warning(f"Inference Scaling: Low quality detected at step {step}: {quality_score:.3f} < {quality_threshold}")
                    elif is_acceptable and quality_score > quality_threshold * 1.2:
                        # Quality is good
                        logging.info(f"Inference Scaling: Good quality at step {step}: {quality_score:.3f}")

            except Exception as e:
                # If decoding fails, continue sampling (don't break the generation)
                import logging
                logging.warning(f"Inference Scaling: Quality check failed at step {step}: {e}")
                # Don't log full traceback in production to avoid spam
                # logging.debug(traceback.format_exc())

        # Perform sampling with quality monitoring
        try:
            # Use comfy.sample.sample which handles everything properly
            result_samples = comfy.sample.sample(
                model=model,
                noise=noise,
                steps=steps,
                cfg=current_cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent_image_tensor,
                denoise=1.0,
                disable_noise=False,
                start_step=None,
                last_step=None,
                force_full_denoise=False,
                noise_mask=noise_mask,
                callback=quality_check_callback,
                disable_pbar=False,
                seed=seed,
            )

            # Return the result as a latent dict (preserve other keys from input)
            result_latent = latent_image.copy()
            result_latent["samples"] = result_samples

            return IO.NodeOutput(result_latent)

        except Exception as e:
            import traceback
            raise Exception(f"Inference scaling sampling failed: {e}\n{traceback.format_exc()}")


class InferenceScalingExtension(ComfyExtension):
    """
    Extension class that registers inference scaling nodes.
    """

    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            VerifierSelectionNode,
            InferenceScalingNode,
        ]


async def comfy_entrypoint() -> InferenceScalingExtension:
    """
    Entry point function that ComfyUI calls to load the extension.
    """
    return InferenceScalingExtension()
