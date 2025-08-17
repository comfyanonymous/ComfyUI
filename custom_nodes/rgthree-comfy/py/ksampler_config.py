"""Some basic config stuff I use for SDXL."""

from .constants import get_category, get_name
from nodes import MAX_RESOLUTION
import comfy.samplers


class RgthreeKSamplerConfig:
  """Some basic config stuff I started using for SDXL, but useful in other spots too."""

  NAME = get_name('KSampler Config')
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {
        "steps_total": ("INT", {
          "default": 30,
          "min": 1,
          "max": MAX_RESOLUTION,
          "step": 1,
        }),
        "refiner_step": ("INT", {
          "default": 24,
          "min": 1,
          "max": MAX_RESOLUTION,
          "step": 1,
        }),
        "cfg": ("FLOAT", {
          "default": 8.0,
          "min": 0.0,
          "max": 100.0,
          "step": 0.5,
        }),
        "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
        "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
        #"refiner_ascore_pos": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
        #"refiner_ascore_neg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
      },
    }

  RETURN_TYPES = ("INT", "INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS,
                  comfy.samplers.KSampler.SCHEDULERS)
  RETURN_NAMES = ("STEPS", "REFINER_STEP", "CFG", "SAMPLER", "SCHEDULER")
  FUNCTION = "main"

  def main(self, steps_total, refiner_step, cfg, sampler_name, scheduler):
    """main"""
    return (
      steps_total,
      refiner_step,
      cfg,
      sampler_name,
      scheduler,
    )
