import torch
import logging
from typing import Dict, Callable
import itertools

import modelopt.torch.quantization as mtq
from tools.ptq.utils import log_quant_summary, save_amax_dict, extract_amax_values

class PTQPipeline:
    def __init__(self, model_patcher, quant_config: dict, filter_func=None):
        self.model_patcher = model_patcher
        self.diffusion_model = model_patcher.model.diffusion_model
        self.quant_config = quant_config
        self.filter_func = filter_func

        logging.debug(f"PTQPipeline initialized with config: {quant_config}")

    @torch.no_grad()
    def calibrate_with_pipeline(
        self,
        calib_pipeline,
        dataloader,
        num_steps: int,
        get_forward_loop: Callable
    ):
        """
        Run calibration using the model-specific forward loop.

        Args:
            calib_pipeline: Calibration pipeline (e.g., FluxT2IPipe)
            dataloader: DataLoader with calibration data
            num_steps: Number of calibration steps
            get_forward_loop: Function that returns forward_loop callable
        """
        logging.info(f"Running calibration with {num_steps} steps...")
        limited_dataloader = itertools.islice(dataloader, num_steps)
        forward_loop = get_forward_loop(calib_pipeline, limited_dataloader)
        try:
            mtq.quantize(self.diffusion_model, self.quant_config, forward_loop=forward_loop)
        except Exception as e:
            logging.error(f"Calibration failed: {e}")
            raise

        try:
            forward_loop()
        except Exception as e:
            logging.error(f"Calibration failed: {e}")
            raise
        logging.info("Calibration complete")
        log_quant_summary(self.diffusion_model)

    def get_amax_dict(self) -> Dict:
        return extract_amax_values(self.diffusion_model)

    def save_amax_values(self, output_path: str, metadata: dict = None):
        amax_dict = self.get_amax_dict()
        save_amax_dict(amax_dict, output_path, metadata=metadata)
        logging.info(f"Saved amax values to {output_path}")

