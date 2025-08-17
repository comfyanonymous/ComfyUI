"""See node."""
import random
from datetime import datetime

from .constants import get_category, get_name
from .log import log_node_warn, log_node_info

# Some extension must be setting a seed as server-generated seeds were not random. We'll set a new
# seed and use that state going forward.
initial_random_state = random.getstate()
random.seed(datetime.now().timestamp())
rgthree_seed_random_state = random.getstate()
random.setstate(initial_random_state)


def new_random_seed():
  """ Gets a new random seed from the rgthree_seed_random_state and resetting the previous state."""
  global rgthree_seed_random_state
  prev_random_state = random.getstate()
  random.setstate(rgthree_seed_random_state)
  seed = random.randint(1, 1125899906842624)
  rgthree_seed_random_state = random.getstate()
  random.setstate(prev_random_state)
  return seed


class RgthreeSeed:
  """Seed node."""

  NAME = get_name('Seed')
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {
        "seed": ("INT", {
          "default": 0,
          "min": -1125899906842624,
          "max": 1125899906842624
        }),
      },
      "hidden": {
        "prompt": "PROMPT",
        "extra_pnginfo": "EXTRA_PNGINFO",
        "unique_id": "UNIQUE_ID",
      },
    }

  RETURN_TYPES = ("INT",)
  RETURN_NAMES = ("SEED",)
  FUNCTION = "main"

  @classmethod
  def IS_CHANGED(cls, seed, prompt=None, extra_pnginfo=None, unique_id=None):
    """Forces a changed state if we happen to get a special seed, as if from the API directly."""
    if seed in (-1, -2, -3):
      # This isn't used, but a different value than previous will force it to be "changed"
      return new_random_seed()
    return seed

  def main(self, seed=0, prompt=None, extra_pnginfo=None, unique_id=None):
    """Returns the passed seed on execution."""

    # We generate random seeds on the frontend in the seed node before sending the workflow in for
    # many reasons. However, if we want to use this in an API call without changing the seed before
    # sending, then users _could_ pass in "-1" and get a random seed used and added to the metadata.
    # Though, this should likely be discouraged for several reasons (thus, a lot of logging).
    if seed in (-1, -2, -3):
      log_node_warn(self.NAME,
                    f'Got "{seed}" as passed seed. ' +
                    'This shouldn\'t happen when queueing from the ComfyUI frontend.',
                    msg_color="YELLOW")
      if seed in (-2, -3):
        log_node_warn(self.NAME,
                      f'Cannot {"increment" if seed == -2 else "decrement"} seed from ' +
                      'server, but will generate a new random seed.',
                      msg_color="YELLOW")

      original_seed = seed
      seed = new_random_seed()
      log_node_info(self.NAME, f'Server-generated random seed {seed} and saving to workflow.')
      log_node_warn(
        self.NAME,
        'NOTE: Re-queues passing in "{seed}" and server-generated random seed won\'t be cached.',
        msg_color="YELLOW")

      if unique_id is None:
        log_node_warn(
          self.NAME, 'Cannot save server-generated seed to image metadata because ' +
          'the node\'s id was not provided.')
      else:
        if extra_pnginfo is None:
          log_node_warn(
            self.NAME, 'Cannot save server-generated seed to image workflow ' +
            'metadata because workflow was not provided.')
        else:
          workflow_node = next(
            (x for x in extra_pnginfo['workflow']['nodes'] if str(x['id']) == str(unique_id)), None)
          if workflow_node is None or 'widgets_values' not in workflow_node:
            log_node_warn(
              self.NAME, 'Cannot save server-generated seed to image workflow ' +
              'metadata because node was not found in the provided workflow.')
          else:
            for index, widget_value in enumerate(workflow_node['widgets_values']):
              if widget_value == original_seed:
                workflow_node['widgets_values'][index] = seed

        if prompt is None:
          log_node_warn(
            self.NAME, 'Cannot save server-generated seed to image API prompt ' +
            'metadata because prompt was not provided.')
        else:
          prompt_node = prompt[str(unique_id)]
          if prompt_node is None or 'inputs' not in prompt_node or 'seed' not in prompt_node[
              'inputs']:
            log_node_warn(
              self.NAME, 'Cannot save server-generated seed to image workflow ' +
              'metadata because node was not found in the provided workflow.')
          else:
            prompt_node['inputs']['seed'] = seed

    return (seed,)
