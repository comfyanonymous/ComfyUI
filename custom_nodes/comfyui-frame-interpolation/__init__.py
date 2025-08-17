import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from .other_nodes import Gradually_More_Denoise_KSampler

#Some models are commented out because the code is not completed
#from vfi_models.eisai import EISAI_VFI
from vfi_models.gmfss_fortuna import GMFSS_Fortuna_VFI
from vfi_models.ifrnet import IFRNet_VFI
from vfi_models.ifunet import IFUnet_VFI
from vfi_models.m2m import M2M_VFI
from vfi_models.rife import RIFE_VFI
from vfi_models.sepconv import SepconvVFI
from vfi_models.amt import AMT_VFI
from vfi_models.film import FILM_VFI
from vfi_models.stmfnet import STMFNet_VFI
from vfi_models.flavr import FLAVR_VFI
from vfi_models.cain import CAIN_VFI
from vfi_utils import MakeInterpolationStateList, FloatToInt
    
NODE_CLASS_MAPPINGS = {
    "KSampler Gradually Adding More Denoise (efficient)": Gradually_More_Denoise_KSampler,
#    "EISAI VFI": EISAI_VFI,
    "GMFSS Fortuna VFI": GMFSS_Fortuna_VFI,
    "IFRNet VFI": IFRNet_VFI,
    "IFUnet VFI": IFUnet_VFI,
    "M2M VFI": M2M_VFI,
    "RIFE VFI": RIFE_VFI,
    "Sepconv VFI": SepconvVFI,
    "AMT VFI": AMT_VFI,
    "FILM VFI": FILM_VFI,
    "Make Interpolation State List": MakeInterpolationStateList,
    "STMFNet VFI": STMFNet_VFI,
    "FLAVR VFI": FLAVR_VFI,
    "CAIN VFI": CAIN_VFI,
    "VFI FloatToInt": FloatToInt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RIFE VFI": "RIFE VFI (recommend rife47 and rife49)"
}