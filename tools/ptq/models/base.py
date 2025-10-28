"""
Abstract base class for model quantization recipes.

Each model type (FLUX, SDXL, Qwen, etc.) implements this interface
to define model-specific quantization logic.
"""
from abc import ABC, abstractmethod
import argparse
from typing import Tuple, Any, Callable


class ModelRecipe(ABC):
    """
    Abstract base class for model quantization recipes.

    Each model type implements this interface to define:
    - How to load the model
    - How to create calibration pipeline
    - How to run calibration (forward_loop)
    - Which layers to quantize (filter function)
    - Model-specific hyperparameters
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """
        Unique identifier for this recipe (e.g., 'flux_dev', 'sdxl').
        Used in CLI --model_type argument.
        """
        pass

    @classmethod
    @abstractmethod
    def add_model_args(cls, parser: argparse.ArgumentParser):
        """
        Add model-specific CLI arguments.

        Example:
            parser.add_argument("--ckpt_path", required=True)
            parser.add_argument("--clip_path", help="Optional CLIP path")

        Args:
            parser: ArgumentParser to add arguments to
        """
        pass

    @abstractmethod
    def __init__(self, args):
        """
        Initialize recipe with parsed CLI arguments.
        Store model-specific configuration.

        Args:
            args: Parsed argparse.Namespace
        """
        pass

    @abstractmethod
    def load_model(self) -> Tuple[Any, ...]:
        """
        Load model from checkpoint(s).

        Returns:
            Tuple of (model_patcher, *other_components)
            e.g., (model_patcher, clip, vae) for FLUX

            First element MUST be model_patcher (ComfyUI ModelPatcher)
        """
        pass

    @abstractmethod
    def create_calibration_pipeline(self, model_components) -> Any:
        """
        Create calibration pipeline for running inference.

        The pipeline should have a __call__ method that runs inference
        for one calibration iteration.

        Args:
            model_components: Output from load_model()

        Returns:
            Pipeline object with __call__(steps, prompt, ...) method
        """
        pass

    @abstractmethod
    def get_forward_loop(self, calib_pipeline, dataloader) -> Callable:
        """
        Return forward_loop function for ModelOptimizer calibration.

        The forward_loop is called by mtq.quantize() to collect activation
        statistics. It should iterate through the dataloader and run
        inference using the calibration pipeline.

        Args:
            calib_pipeline: Output from create_calibration_pipeline()
            dataloader: DataLoader with calibration data

        Returns:
            Callable that takes no arguments and runs calibration loop

        Example:
            def forward_loop():
                for prompt in dataloader:
                    calib_pipeline(steps=4, prompt=prompt)
            return forward_loop
        """
        pass


    @abstractmethod
    def get_default_calib_steps(self) -> int:
        """
        Default number of calibration steps for this model.

        Returns:
            Number of calibration iterations (e.g., 128 for FLUX Dev)
        """
        pass



