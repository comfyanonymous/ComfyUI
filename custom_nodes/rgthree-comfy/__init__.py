"""
@author: rgthree
@title: Comfy Nodes
@nickname: rgthree
@description: A bunch of nodes I created that I also find useful.
"""

from glob import glob
import json
import os
import shutil
import re
import random

import execution

from .py.log import log
from .py.config import get_config_value
from .py.server.rgthree_server import *

from .py.context import RgthreeContext
from .py.context_switch import RgthreeContextSwitch
from .py.context_switch_big import RgthreeContextSwitchBig
from .py.display_any import RgthreeDisplayAny, RgthreeDisplayInt
from .py.lora_stack import RgthreeLoraLoaderStack
from .py.seed import RgthreeSeed
from .py.sdxl_empty_latent_image import RgthreeSDXLEmptyLatentImage
from .py.power_prompt import RgthreePowerPrompt
from .py.power_prompt_simple import RgthreePowerPromptSimple
from .py.image_inset_crop import RgthreeImageInsetCrop
from .py.context_big import RgthreeBigContext
from .py.dynamic_context import RgthreeDynamicContext
from .py.dynamic_context_switch import RgthreeDynamicContextSwitch
from .py.ksampler_config import RgthreeKSamplerConfig
from .py.sdxl_power_prompt_postive import RgthreeSDXLPowerPromptPositive
from .py.sdxl_power_prompt_simple import RgthreeSDXLPowerPromptSimple
from .py.any_switch import RgthreeAnySwitch
from .py.context_merge import RgthreeContextMerge
from .py.context_merge_big import RgthreeContextMergeBig
from .py.image_comparer import RgthreeImageComparer
from .py.power_lora_loader import RgthreePowerLoraLoader
from .py.power_primitive import RgthreePowerPrimitive
from .py.image_or_latent_size import RgthreeImageOrLatentSize
from .py.image_resize import RgthreeImageResize
from .py.power_puter import RgthreePowerPuter

NODE_CLASS_MAPPINGS = {
  RgthreeBigContext.NAME: RgthreeBigContext,
  RgthreeContext.NAME: RgthreeContext,
  RgthreeContextSwitch.NAME: RgthreeContextSwitch,
  RgthreeContextSwitchBig.NAME: RgthreeContextSwitchBig,
  RgthreeContextMerge.NAME: RgthreeContextMerge,
  RgthreeContextMergeBig.NAME: RgthreeContextMergeBig,
  RgthreeDisplayInt.NAME: RgthreeDisplayInt,
  RgthreeDisplayAny.NAME: RgthreeDisplayAny,
  RgthreeLoraLoaderStack.NAME: RgthreeLoraLoaderStack,
  RgthreeSeed.NAME: RgthreeSeed,
  RgthreeImageInsetCrop.NAME: RgthreeImageInsetCrop,
  RgthreePowerPrompt.NAME: RgthreePowerPrompt,
  RgthreePowerPromptSimple.NAME: RgthreePowerPromptSimple,
  RgthreeKSamplerConfig.NAME: RgthreeKSamplerConfig,
  RgthreeSDXLEmptyLatentImage.NAME: RgthreeSDXLEmptyLatentImage,
  RgthreeSDXLPowerPromptPositive.NAME: RgthreeSDXLPowerPromptPositive,
  RgthreeSDXLPowerPromptSimple.NAME: RgthreeSDXLPowerPromptSimple,
  RgthreeAnySwitch.NAME: RgthreeAnySwitch,
  RgthreeImageComparer.NAME: RgthreeImageComparer,
  RgthreePowerLoraLoader.NAME: RgthreePowerLoraLoader,
  RgthreePowerPrimitive.NAME: RgthreePowerPrimitive,
  RgthreeImageOrLatentSize.NAME: RgthreeImageOrLatentSize,
  RgthreeImageResize.NAME: RgthreeImageResize,
  RgthreePowerPuter.NAME: RgthreePowerPuter,
}

if get_config_value('unreleased.dynamic_context.enabled') is True:
  NODE_CLASS_MAPPINGS[RgthreeDynamicContext.NAME] = RgthreeDynamicContext
  NODE_CLASS_MAPPINGS[RgthreeDynamicContextSwitch.NAME] = RgthreeDynamicContextSwitch

# WEB_DIRECTORY is the comfyui nodes directory that ComfyUI will link and auto-load.
WEB_DIRECTORY = "./web/comfyui"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_WEB = os.path.abspath(f'{THIS_DIR}/{WEB_DIRECTORY}')
DIR_PY = os.path.abspath(f'{THIS_DIR}/py')

# remove old directories
OLD_DIRS = [
  os.path.abspath(f'{THIS_DIR}/../../web/extensions/rgthree'),
  os.path.abspath(f'{THIS_DIR}/../../web/extensions/rgthree-comfy'),
]
for old_dir in OLD_DIRS:
  if os.path.exists(old_dir):
    shutil.rmtree(old_dir)

__all__ = ['NODE_CLASS_MAPPINGS', 'WEB_DIRECTORY']

NOT_NODES = ['constants', 'log', 'utils', 'rgthree', 'rgthree_server', 'image_clipbaord', 'config']

nodes = []
for file in glob(os.path.join(DIR_PY, '*.py')) + glob(os.path.join(DIR_WEB, '*.js')):
  name = os.path.splitext(os.path.basename(file))[0]
  if name in NOT_NODES or name in nodes:
    continue
  if name.startswith('_') or name.startswith('base') or 'utils' in name:
    continue
  nodes.append(name)
  if name == 'display_any':
    nodes.append('display_int')

print()
adjs = ['exciting', 'extraordinary', 'epic', 'fantastic', 'magnificent']
log(f'Loaded {len(nodes)} {random.choice(adjs)} nodes. 🎉', color='BRIGHT_GREEN')
print()
