from __future__ import annotations
from typing import Any, Literal, TYPE_CHECKING, TypeVar, Callable, Optional, cast, TypedDict, NotRequired
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections import Counter
from comfy_api.v3.resources import Resources, ResourcesLocal
# used for type hinting
import torch
from spandrel import ImageModelDescriptor
from comfy.model_patcher import ModelPatcher
from comfy.samplers import Sampler, CFGGuider
from comfy.sd import CLIP
from comfy.controlnet import ControlNet
from comfy.sd import VAE
from comfy.sd import StyleModel as StyleModel_
from comfy.clip_vision import ClipVisionModel
from comfy.clip_vision import Output as ClipVisionOutput_
from comfy_api.input import VideoInput
from comfy.hooks import HookGroup, HookKeyframeGroup
# from comfy_extras.nodes_images import SVG as SVG_ # NOTE: needs to be moved before can be imported due to circular reference


class FolderType(str, Enum):
    input = "input"
    output = "output"
    temp = "temp"

class RemoteOptions:
    def __init__(self, route: str, refresh_button: bool, control_after_refresh: Literal["first", "last"]="first",
                 timeout: int=None, max_retries: int=None, refresh: int=None):
        self.route = route
        """The route to the remote source."""
        self.refresh_button = refresh_button
        """Specifies whether to show a refresh button in the UI below the widget."""
        self.control_after_refresh = control_after_refresh
        """Specifies the control after the refresh button is clicked. If "first", the first item will be automatically selected, and so on."""
        self.timeout = timeout
        """The maximum amount of time to wait for a response from the remote source in milliseconds."""
        self.max_retries = max_retries
        """The maximum number of retries before aborting the request."""
        self.refresh = refresh
        """The TTL of the remote input's value in milliseconds. Specifies the interval at which the remote input's value is refreshed."""

    def as_dict(self):
        return prune_dict({
            "route": self.route,
            "refresh_button": self.refresh_button,
            "control_after_refresh": self.control_after_refresh,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "refresh": self.refresh,
        })


def is_class(obj):
    '''
    Returns True if is a class type.
    Returns False if is a class instance.
    '''
    return isinstance(obj, type)


def copy_class(cls: type) -> type:
    '''
    Copy a class and its attributes.
    '''
    if cls is None:
        return None
    cls_dict = {
            k: v for k, v in cls.__dict__.items()
            if k not in ('__dict__', '__weakref__', '__module__', '__doc__')
        }
    # new class
    new_cls = type(
        cls.__name__,
        (cls,),
        cls_dict
    )
    # metadata preservation
    new_cls.__module__ = cls.__module__
    new_cls.__doc__ = cls.__doc__
    return new_cls


class NumberDisplay(str, Enum):
    number = "number"
    slider = "slider"


class ComfyType:
    Type = Any
    io_type: str = None
    Input: type[InputV3] = None
    Output: type[OutputV3] = None

# NOTE: this is a workaround to make the decorator return the correct type
T = TypeVar("T", bound=type)
def comfytype(io_type: str, **kwargs):
    '''
    Decorator to mark nested classes as ComfyType; io_type will be bound to the class.
    
    A ComfyType may have the following attributes:
    - Type = <type hint here>
    - class Input(InputV3): ...
    - class Output(OutputV3): ...
    '''
    def decorator(cls: T) -> T:
        if isinstance(cls, ComfyType) or issubclass(cls, ComfyType):
            # clone Input and Output classes to avoid modifying the original class
            new_cls = cls
            new_cls.Input = copy_class(new_cls.Input)
            new_cls.Output = copy_class(new_cls.Output)
        else:
            # copy class attributes except for special ones that shouldn't be in type()
            cls_dict = {
                k: v for k, v in cls.__dict__.items()
                if k not in ('__dict__', '__weakref__', '__module__', '__doc__')
            }
            # new class
            new_cls: ComfyType = type(
                cls.__name__,
                (cls, ComfyType),
                cls_dict
            )
            # metadata preservation
            new_cls.__module__ = cls.__module__
            new_cls.__doc__ = cls.__doc__
            # assign ComfyType attributes, if needed
        # NOTE: do we need __ne__ trick for io_type? (see node_typing.IO.__ne__ for details)
        new_cls.io_type = io_type
        if new_cls.Input is not None:
            new_cls.Input.Parent = new_cls
        if new_cls.Output is not None:
            new_cls.Output.Parent = new_cls
        return new_cls
    return decorator

def Custom(io_type: str) -> type[ComfyType]:
    '''Create a ComfyType for a custom io_type.'''
    @comfytype(io_type=io_type)
    class CustomComfyType(ComfyTypeIO):
        ...
    return CustomComfyType

class IO_V3:
    '''
    Base class for V3 Inputs and Outputs.
    '''
    Parent: ComfyType = None

    def __init__(self):
        pass

    @property
    def io_type(self):
        return self.Parent.io_type

    @property
    def Type(self):
        return self.Parent.Type

class InputV3(IO_V3):
    '''
    Base class for a V3 Input.
    '''
    def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None, extra_dict=None):
        super().__init__()
        self.id = id
        self.display_name = display_name
        self.optional = optional
        self.tooltip = tooltip
        self.lazy = lazy
        self.extra_dict = extra_dict if extra_dict is not None else {}

    def as_dict_V1(self):
        return prune_dict({
            "display_name": self.display_name,
            "tooltip": self.tooltip,
            "lazy": self.lazy,
        }) | prune_dict(self.extra_dict)

    def get_io_type_V1(self):
        return self.io_type

class WidgetInputV3(InputV3):
    '''
    Base class for a V3 Input with widget.
    '''
    def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                 default: Any=None,
                 socketless: bool=None, widgetType: str=None, extra_dict=None):
        super().__init__(id, display_name, optional, tooltip, lazy, extra_dict)
        self.default = default
        self.socketless = socketless
        self.widgetType = widgetType
    
    def as_dict_V1(self):
        return super().as_dict_V1() | prune_dict({
            "default": self.default,
            "socketless": self.socketless,
            "widgetType": self.widgetType,
        })
    
    def get_io_type_V1(self):
        return self.widgetType if self.widgetType is not None else super().get_io_type_V1()

class OutputV3(IO_V3):
    def __init__(self, id: str, display_name: str=None, tooltip: str=None,
                 is_output_list=False):
        self.id = id
        self.display_name = display_name
        self.tooltip = tooltip
        self.is_output_list = is_output_list


class ComfyTypeIO(ComfyType):
    '''ComfyType subclass that has default Input and Output classes; useful for basic Inputs and Outputs.'''
    class Input(InputV3):
        ...
    class Output(OutputV3):
        ...


class NodeState(ABC):
    def __init__(self, node_id: str):
        self.node_id = node_id

    @abstractmethod
    def get_value(self, key: str):
        pass

    @abstractmethod
    def set_value(self, key: str, value: Any):
        pass

    @abstractmethod
    def pop(self, key: str):
        pass

    @abstractmethod
    def __contains__(self, key: str):
        pass


class NodeStateLocal(NodeState):
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.local_state = {}

    def get_value(self, key: str):
        return self.local_state.get(key)

    def set_value(self, key: str, value: Any):
        self.local_state[key] = value

    def pop(self, key: str):
        return self.local_state.pop(key, None)

    def __contains__(self, key: str):
        return key in self.local_state

    def __getattr__(self, key: str):
        local_state = type(self).__getattribute__(self, "local_state")
        if key in local_state:
            return local_state[key]
        return None
        # raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        if key in ['node_id', 'local_state']:
            super().__setattr__(key, value)
        else:
            self.local_state[key] = value
    
    def __setitem__(self, key: str, value: Any):
        self.local_state[key] = value
    
    def __getitem__(self, key: str):
        return self.local_state[key]
    
    def __delitem__(self, key: str):
        del self.local_state[key]


@comfytype(io_type="BOOLEAN")
class Boolean:
    Type = bool
    
    class Input(WidgetInputV3):
        '''Boolean input.'''
        def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                    default: bool=None, label_on: str=None, label_off: str=None,
                    socketless: bool=None):
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, self.io_type)
            self.label_on = label_on
            self.label_off = label_off
            self.default: bool
        
        def as_dict_V1(self):
            return super().as_dict_V1() | prune_dict({
                "label_on": self.label_on,
                "label_off": self.label_off,
            })

    class Output(OutputV3):
        ...

@comfytype(io_type="INT")
class Int:
    Type = int

    class Input(WidgetInputV3):
        '''Integer input.'''
        def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                    default: int=None, min: int=None, max: int=None, step: int=None, control_after_generate: bool=None,
                    display_mode: NumberDisplay=None, socketless: bool=None):
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, self.io_type)
            self.min = min
            self.max = max
            self.step = step
            self.control_after_generate = control_after_generate
            self.display_mode = display_mode
            self.default: int

        def as_dict_V1(self):
            return super().as_dict_V1() | prune_dict({
                "min": self.min,
                "max": self.max,
                "step": self.step,
                "control_after_generate": self.control_after_generate,
                "display": self.display_mode,
            })

    class Output(OutputV3):
        ...

@comfytype(io_type="FLOAT")
class Float(ComfyTypeIO):
    Type = float

    class Input(WidgetInputV3):
        '''Float input.'''
        def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                    default: float=None, min: float=None, max: float=None, step: float=None, round: float=None,
                    display_mode: NumberDisplay=None, socketless: bool=None):
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, self.io_type)
            self.min = min
            self.max = max
            self.step = step
            self.round = round
            self.display_mode = display_mode
            self.default: float

        def as_dict_V1(self):
            return super().as_dict_V1() | prune_dict({
                "min": self.min,
                "max": self.max,
                "step": self.step,
                "round": self.round,
                "display": self.display_mode,
            })

@comfytype(io_type="STRING")
class String(ComfyTypeIO):
    Type = str

    class Input(WidgetInputV3):
        '''String input.'''
        def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                    multiline=False, placeholder: str=None, default: int=None,
                    socketless: bool=None):
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, self.io_type)
            self.multiline = multiline
            self.placeholder = placeholder
            self.default: str
        
        def as_dict_V1(self):
            return super().as_dict_V1() | prune_dict({
                "multiline": self.multiline,
                "placeholder": self.placeholder,
            })

@comfytype(io_type="COMBO")
class Combo(ComfyType):
    Type = str
    class Input(WidgetInputV3):
        '''Combo input (dropdown).'''
        Type = str
        def __init__(self, id: str, options: list[str]=None, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                    default: str=None, control_after_generate: bool=None,
                    image_upload: bool=None, image_folder: FolderType=None,
                    remote: RemoteOptions=None,
                    socketless: bool=None):
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, self.io_type)
            self.multiselect = False
            self.options = options
            self.control_after_generate = control_after_generate
            self.image_upload = image_upload
            self.image_folder = image_folder
            self.remote = remote
            self.default: str

        def as_dict_V1(self):
            return super().as_dict_V1() | prune_dict({
                "multiselect": self.multiselect,
                "options": self.options,
                "control_after_generate": self.control_after_generate,
                "image_upload": self.image_upload,
                "image_folder": self.image_folder.value if self.image_folder else None,
                "remote": self.remote.as_dict() if self.remote else None,
            })


@comfytype(io_type="COMBO")
class MultiCombo(ComfyType):
    '''Multiselect Combo input (dropdown for selecting potentially more than one value).'''
    # TODO: something is wrong with the serialization, frontend does not recognize it as multiselect
    Type = list[str]
    class Input(Combo.Input):
        def __init__(self, id: str, options: list[str], display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                    default: list[str]=None, placeholder: str=None, chip: bool=None, control_after_generate: bool=None,
                    socketless: bool=None, widgetType: str=None):
            super().__init__(id, options, display_name, optional, tooltip, lazy, default, control_after_generate, socketless, widgetType)
            self.multiselect = True
            self.placeholder = placeholder
            self.chip = chip
            self.default: list[str]

        def as_dict_V1(self):
            to_return = super().as_dict_V1() | prune_dict({
                "multiselect": self.multiselect,
                "placeholder": self.placeholder,
                "chip": self.chip,
            })
            return to_return

@comfytype(io_type="IMAGE")
class Image(ComfyTypeIO):
    Type = torch.Tensor

@comfytype(io_type="MASK")
class Mask(ComfyTypeIO):
    Type = torch.Tensor

@comfytype(io_type="LATENT")
class Latent(ComfyTypeIO):
    '''Latents are stored as a dictionary.'''
    class LatentDict(TypedDict):
        samples: torch.Tensor
        '''Latent tensors.'''
        noise_mask: NotRequired[torch.Tensor]
        batch_index: NotRequired[list[int]]
        type: NotRequired[str]
        '''Only needed if dealing with these types: audio, hunyuan3dv2'''
    Type = LatentDict

@comfytype(io_type="CONDITIONING")
class Conditioning(ComfyTypeIO):
    class PooledDict(TypedDict):
        pooled_output: torch.Tensor
        '''Pooled output from CLIP.'''
        control: NotRequired[ControlNet]
        '''ControlNet to apply to conditioning.'''
        control_apply_to_uncond: NotRequired[bool]
        '''Whether to apply ControlNet to matching negative conditioning at sample time, if applicable.'''
        cross_attn_controlnet: NotRequired[torch.Tensor]
        '''CrossAttn from CLIP to use for controlnet only.'''
        pooled_output_controlnet: NotRequired[torch.Tensor]
        '''Pooled output from CLIP to use for controlnet only.'''
        gligen: NotRequired[tuple[str, Gligen, list[tuple[torch.Tensor, int, ...]]]]
        '''GLIGEN to apply to conditioning.'''
        area: NotRequired[tuple[int, ...] | tuple[str, float, ...]]
        '''Set area of conditioning. First half of values apply to dimensions, the second half apply to coordinates.
        By default, the dimensions are based on total pixel amount, but the first value can be set to "percentage" to use a percentage of the image size instead.

        (1024, 1024, 0, 0) would apply conditioning to the top-left 1024x1024 pixels.
        
        ("percentage", 0.5, 0.5, 0, 0) would apply conditioning to the top-left 50% of the image.''' # TODO: verify its actually top-left
        strength: NotRequired[float]
        '''Strength of conditioning. Default strength is 1.0.'''
        mask: NotRequired[torch.Tensor]
        '''Mask to apply conditioning to.'''
        mask_strength: NotRequired[float]
        '''Strength of conditioning mask. Default strength is 1.0.'''
        set_area_to_bounds: NotRequired[bool]
        '''Whether conditioning mask should determine bounds of area - if set to false, latents are sampled at full resolution and result is applied in mask.'''
        concat_latent_image: NotRequired[torch.Tensor]
        '''Used for inpainting and specific models.'''
        concat_mask: NotRequired[torch.Tensor]
        '''Used for inpainting and specific models.'''
        concat_image: NotRequired[torch.Tensor]
        '''Used by SD_4XUpscale_Conditioning.'''
        noise_augmentation: NotRequired[float]
        '''Used by SD_4XUpscale_Conditioning.'''
        hooks: NotRequired[HookGroup]
        '''Applies hooks to conditioning.'''
        default: NotRequired[bool]
        '''Whether to this conditioning is 'default'; default conditioning gets applied to any areas of the image that have no masks/areas applied, assuming at least one area/mask is present during sampling.'''
        start_percent: NotRequired[float]
        '''Determines relative step to begin applying conditioning, expressed as a float between 0.0 and 1.0.'''
        end_percent: NotRequired[float]
        '''Determines relative step to end applying conditioning, expressed as a float between 0.0 and 1.0.'''
        clip_start_percent: NotRequired[float]
        '''Internal variable for conditioning scheduling - start of application, expressed as a float between 0.0 and 1.0.'''
        clip_end_percent: NotRequired[float]
        '''Internal variable for conditioning scheduling - end of application, expressed as a float between 0.0 and 1.0.'''
        attention_mask: NotRequired[torch.Tensor]
        '''Masks text conditioning; used by StyleModel among others.'''
        attention_mask_img_shape: NotRequired[tuple[int, ...]]
        '''Masks text conditioning; used by StyleModel among others.'''
        unclip_conditioning: NotRequired[list[dict]]
        '''Used by unCLIP.'''
        conditioning_lyrics: NotRequired[torch.Tensor]
        '''Used by AceT5Model.'''
        seconds_start: NotRequired[float]
        '''Used by StableAudio.'''
        seconds_total: NotRequired[float]
        '''Used by StableAudio.'''
        lyrics_strength: NotRequired[float]
        '''Used by AceStepAudio.'''
        width: NotRequired[int]
        '''Used by certain models (e.g. CLIPTextEncodeSDXL/Refiner, PixArtAlpha).'''
        height: NotRequired[int]
        '''Used by certain models (e.g. CLIPTextEncodeSDXL/Refiner, PixArtAlpha).'''
        aesthetic_score: NotRequired[float]
        '''Used by CLIPTextEncodeSDXL/Refiner.'''
        crop_w: NotRequired[int]
        '''Used by CLIPTextEncodeSDXL.'''
        crop_h: NotRequired[int]
        '''Used by CLIPTextEncodeSDXL.'''
        target_width: NotRequired[int]
        '''Used by CLIPTextEncodeSDXL.'''
        target_height: NotRequired[int]
        '''Used by CLIPTextEncodeSDXL.'''
        reference_latents: NotRequired[list[torch.Tensor]]
        '''Used by ReferenceLatent.'''
        guidance: NotRequired[float]
        '''Used by Flux-like models with guidance embed.'''
        guiding_frame_index: NotRequired[int]
        '''Used by Hunyuan ImageToVideo.'''
        ref_latent: NotRequired[torch.Tensor]
        '''Used by Hunyuan ImageToVideo.'''
        keyframe_idxs: NotRequired[list[int]]
        '''Used by LTXV.'''
        frame_rate: NotRequired[float]
        '''Used by LTXV.'''
        stable_cascade_prior: NotRequired[torch.Tensor]
        '''Used by StableCascade.'''
        elevation: NotRequired[list[float]]
        '''Used by SV3D.'''
        azimuth: NotRequired[list[float]]
        '''Used by SV3D.'''
        motion_bucket_id: NotRequired[int]
        '''Used by SVD-like models.'''
        fps: NotRequired[int]
        '''Used by SVD-like models.'''
        augmentation_level: NotRequired[float]
        '''Used by SVD-like models.'''
        clip_vision_output: NotRequired[ClipVisionOutput_]
        '''Used by WAN-like models.'''
        vace_frames: NotRequired[torch.Tensor]
        '''Used by WAN VACE.'''
        vace_mask: NotRequired[torch.Tensor]
        '''Used by WAN VACE.'''
        vace_strength: NotRequired[float]
        '''Used by WAN VACE.'''
        camera_conditions: NotRequired[Any] # TODO: assign proper type once defined
        '''Used by WAN Camera.'''
        time_dim_concat: NotRequired[torch.Tensor]
        '''Used by WAN Phantom Subject.'''

    CondList = list[tuple[torch.Tensor, PooledDict]]
    Type = CondList

@comfytype(io_type="SAMPLER")
class Sampler(ComfyTypeIO):
    Type = Sampler

@comfytype(io_type="SIGMAS")
class Sigmas(ComfyTypeIO):
    Type = torch.Tensor

@comfytype(io_type="NOISE")
class Noise(ComfyTypeIO):
    Type = torch.Tensor

@comfytype(io_type="GUIDER")
class Guider(ComfyTypeIO):
    Type = CFGGuider

@comfytype(io_type="CLIP")
class Clip(ComfyTypeIO):
    Type = CLIP

@comfytype(io_type="CONTROL_NET")
class ControlNet(ComfyTypeIO):
    Type = ControlNet

@comfytype(io_type="VAE")
class Vae(ComfyTypeIO):
    Type = VAE

@comfytype(io_type="MODEL")
class Model(ComfyTypeIO):
    Type = ModelPatcher

@comfytype(io_type="CLIP_VISION")
class ClipVision(ComfyTypeIO):
    Type = ClipVisionModel

@comfytype(io_type="CLIP_VISION_OUTPUT")
class ClipVisionOutput(ComfyTypeIO):
    Type = ClipVisionOutput_

@comfytype(io_type="STYLE_MODEL")
class StyleModel(ComfyTypeIO):
    Type = StyleModel_

@comfytype(io_type="GLIGEN")
class Gligen(ComfyTypeIO):
    '''ModelPatcher that wraps around a 'Gligen' model.'''
    Type = ModelPatcher

@comfytype(io_type="UPSCALE_MODEL")
class UpscaleModel(ComfyTypeIO):
    Type = ImageModelDescriptor

@comfytype(io_type="AUDIO")
class Audio(ComfyTypeIO):
    class AudioDict(TypedDict):
        waveform: torch.Tensor
        sampler_rate: int
    Type = AudioDict

@comfytype(io_type="VIDEO")
class Video(ComfyTypeIO):
    Type = VideoInput

@comfytype(io_type="SVG")
class SVG(ComfyTypeIO):
    Type = Any # TODO: SVG class is defined in comfy_extras/nodes_images.py, causing circular reference; should be moved to somewhere else before referenced directly in v3

@comfytype(io_type="LORA_MODEL")
class LoraModel(ComfyTypeIO):
    Type = dict[str, torch.Tensor]

@comfytype(io_type="LOSS_MAP")
class LossMap(ComfyTypeIO):
    class LossMapDict(TypedDict):
        loss: list[torch.Tensor]
    Type = LossMapDict

@comfytype(io_type="VOXEL")
class Voxel(ComfyTypeIO):
    Type = Any # TODO: VOXEL class is defined in comfy_extras/nodes_hunyuan3d.py; should be moved to somewhere else before referenced directly in v3

@comfytype(io_type="MESH")
class Mesh(ComfyTypeIO):
    Type = Any # TODO: MESH class is defined in comfy_extras/nodes_hunyuan3d.py; should be moved to somewhere else before referenced directly in v3

@comfytype(io_type="HOOKS")
class Hooks(ComfyTypeIO):
    Type = HookGroup

@comfytype(io_type="HOOK_KEYFRAMES")
class HookKeyframes(ComfyTypeIO):
    Type = HookKeyframeGroup

@comfytype(io_type="TIMESTEPS_RANGE")
class TimestepsRange(ComfyTypeIO):
    '''Range defined by start and endpoint, between 0.0 and 1.0.'''
    Type = tuple[int, int]

@comfytype(io_type="LATENT_OPERATION")
class LatentOperation(ComfyTypeIO):
    Type = Callable[[torch.Tensor], torch.Tensor]

@comfytype(io_type="FLOW_CONTROL")
class FlowControl(ComfyTypeIO):
    # NOTE: only used in testing_nodes right now
    Type = tuple[str, Any]

@comfytype(io_type="ACCUMULATION")
class Accumulation(ComfyTypeIO):
    # NOTE: only used in testing_nodes right now
    class AccumulationDict(TypedDict):
        accum: list[Any]
    Type = AccumulationDict

@comfytype(io_type="LOAD3D_CAMERA")
class Load3DCamera(ComfyTypeIO):
    Type = Any # TODO: figure out type for this; in code, only described as image['camera_info'], gotten from a LOAD_3D or LOAD_3D_ANIMATION type

@comfytype(io_type="POINT")
class Point(ComfyTypeIO):
    Type = Any # NOTE: I couldn't find any references in core code to POINT io_type. Does this exist?

@comfytype(io_type="FACE_ANALYSIS")
class FaceAnalysis(ComfyTypeIO):
    Type = Any # NOTE: I couldn't find any references in core code to POINT io_type. Does this exist?

@comfytype(io_type="BBOX")
class BBOX(ComfyTypeIO):
    Type = Any # NOTE: I couldn't find any references in core code to POINT io_type. Does this exist?

@comfytype(io_type="SEGS")
class SEGS(ComfyTypeIO):
    Type = Any # NOTE: I couldn't find any references in core code to POINT io_type. Does this exist?

@comfytype(io_type="COMFY_MULTITYPED_V3")
class MultiType:
    Type = Any
    class Input(InputV3):
        '''
        Input that permits more than one input type; if `id` is an instance of `ComfyType.Input`, then that input will be used to create a widget (if applicable) with overridden values.
        '''
        def __init__(self, id: str | InputV3, types: list[type[ComfyType] | ComfyType], display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None, extra_dict=None):
            # if id is an Input, then use that Input with overridden values
            self.input_override = None
            if isinstance(id, InputV3):
                self.input_override = id
                optional = id.optional if id.optional is True else optional
                tooltip = id.tooltip if id.tooltip is not None else tooltip
                display_name = id.display_name if id.display_name is not None else display_name
                lazy = id.lazy if id.lazy is not None else lazy
                id = id.id
            super().__init__(id, display_name, optional, tooltip, lazy, extra_dict)
            self._io_types = types
        
        @property
        def io_types(self) -> list[type[InputV3]]:
            '''
            Returns list of InputV3 class types permitted.
            '''
            io_types = []
            for x in self._io_types:
                if not is_class(x):
                    io_types.append(type(x))
                else:
                    io_types.append(x)
            return io_types
        
        def get_io_type_V1(self):
            # ensure types are unique and order is preserved
            str_types = [x.io_type for x in self.io_types]
            if self.input_override is not None:
                str_types.insert(0, self.input_override.get_io_type_V1())
            return ",".join(list(dict.fromkeys(str_types)))
        
        def as_dict_V1(self):
            if self.input_override is not None:
                return self.input_override.as_dict_V1() | super().as_dict_V1()
            else:
                return super().as_dict_V1()

class DynamicInput(InputV3):
    '''
    Abstract class for dynamic input registration.
    '''
    def __init__(self, io_type: str, id: str, display_name: str=None):
        super().__init__(io_type, id, display_name)

class DynamicOutput(OutputV3):
    '''
    Abstract class for dynamic output registration.
    '''
    def __init__(self, io_type: str, id: str, display_name: str=None):
        super().__init__(io_type, id, display_name)

# io_type="COMFY_MULTIGROW_V3"
class AutoGrowDynamicInput(DynamicInput):
    '''
    Dynamic Input that adds another template_input each time one is provided.

    Additional inputs are forced to have 'optional=True'.
    '''
    def __init__(self, id: str, template_input: InputV3, min: int=1, max: int=None):
        super().__init__("AutoGrowDynamicInput", id)
        self.template_input = template_input
        if min is not None:
            assert(min >= 1)
        if max is not None:
            assert(max >= 1)
        self.min = min
        self.max = max

# io_type="COMFY_COMBODYNAMIC_V3"
class ComboDynamicInput(DynamicInput):
    def __init__(self, id: str):
        pass

AutoGrowDynamicInput(id="dynamic", template_input=Image.Input(id="image"))


class HiddenHolder:
    def __init__(self, unique_id: str, prompt: Any,
                 extra_pnginfo: Any, dynprompt: Any,
                 auth_token_comfy_org: str, api_key_comfy_org: str, **kwargs):
        self.unique_id = unique_id
        """UNIQUE_ID is the unique identifier of the node, and matches the id property of the node on the client side. It is commonly used in client-server communications (see messages)."""
        self.prompt = prompt
        """PROMPT is the complete prompt sent by the client to the server. See the prompt object for a full description."""
        self.extra_pnginfo = extra_pnginfo
        """EXTRA_PNGINFO is a dictionary that will be copied into the metadata of any .png files saved. Custom nodes can store additional information in this dictionary for saving (or as a way to communicate with a downstream node)."""
        self.dynprompt = dynprompt
        """DYNPROMPT is an instance of comfy_execution.graph.DynamicPrompt. It differs from PROMPT in that it may mutate during the course of execution in response to Node Expansion."""
        self.auth_token_comfy_org = auth_token_comfy_org
        """AUTH_TOKEN_COMFY_ORG is a token acquired from signing into a ComfyOrg account on frontend."""
        self.api_key_comfy_org = api_key_comfy_org
        """API_KEY_COMFY_ORG is an API Key generated by ComfyOrg that allows skipping signing into a ComfyOrg account on frontend."""

    def __getattr__(self, key: str):
        '''If hidden variable not found, return None.'''
        return None
    
    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            unique_id=d.get(Hidden.unique_id, None),
            prompt=d.get(Hidden.prompt, None),
            extra_pnginfo=d.get(Hidden.extra_pnginfo, None),
            dynprompt=d.get(Hidden.dynprompt, None),
            auth_token_comfy_org=d.get(Hidden.auth_token_comfy_org, None),
            api_key_comfy_org=d.get(Hidden.api_key_comfy_org, None),
        )

class Hidden(str, Enum):
    '''
    Enumerator for requesting hidden variables in nodes.
    '''
    unique_id = "UNIQUE_ID"
    """UNIQUE_ID is the unique identifier of the node, and matches the id property of the node on the client side. It is commonly used in client-server communications (see messages)."""
    prompt = "PROMPT"
    """PROMPT is the complete prompt sent by the client to the server. See the prompt object for a full description."""
    extra_pnginfo = "EXTRA_PNGINFO"
    """EXTRA_PNGINFO is a dictionary that will be copied into the metadata of any .png files saved. Custom nodes can store additional information in this dictionary for saving (or as a way to communicate with a downstream node)."""
    dynprompt = "DYNPROMPT"
    """DYNPROMPT is an instance of comfy_execution.graph.DynamicPrompt. It differs from PROMPT in that it may mutate during the course of execution in response to Node Expansion."""
    auth_token_comfy_org = "AUTH_TOKEN_COMFY_ORG"
    """AUTH_TOKEN_COMFY_ORG is a token acquired from signing into a ComfyOrg account on frontend."""
    api_key_comfy_org = "API_KEY_COMFY_ORG"
    """API_KEY_COMFY_ORG is an API Key generated by ComfyOrg that allows skipping signing into a ComfyOrg account on frontend."""


@dataclass
class NodeInfoV1:
    input: dict=None
    input_order: dict[str, list[str]]=None
    output: list[str]=None
    output_is_list: list[bool]=None
    output_name: list[str]=None
    output_tooltips: list[str]=None
    name: str=None
    display_name: str=None
    description: str=None
    python_module: Any=None
    category: str=None
    output_node: bool=None
    deprecated: bool=None
    experimental: bool=None
    api_node: bool=None


def as_pruned_dict(dataclass_obj):
    '''Return dict of dataclass object with pruned None values.'''
    return prune_dict(asdict(dataclass_obj))

def prune_dict(d: dict):
    return {k: v for k,v in d.items() if v is not None}


@dataclass
class SchemaV3:
    """Definition of V3 node properties."""

    node_id: str
    """ID of node - should be globally unique. If this is a custom node, add a prefix or postfix to avoid name clashes."""
    display_name: str = None
    """Display name of node."""
    category: str = "sd"
    """The category of the node, as per the "Add Node" menu."""
    inputs: list[InputV3]=None
    outputs: list[OutputV3]=None
    hidden: list[Hidden]=None
    description: str=""
    """Node description, shown as a tooltip when hovering over the node."""
    is_input_list: bool = False
    """A flag indicating if this node implements the additional code necessary to deal with OUTPUT_IS_LIST nodes.

    All inputs of ``type`` will become ``list[type]``, regardless of how many items are passed in.  This also affects ``check_lazy_status``.

    From the docs:

    A node can also override the default input behaviour and receive the whole list in a single call. This is done by setting a class attribute `INPUT_IS_LIST` to ``True``.

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/lists#list-processing
    """
    is_output_node: bool=False
    """Flags this node as an output node, causing any inputs it requires to be executed.

    If a node is not connected to any output nodes, that node will not be executed.  Usage::

        OUTPUT_NODE = True

    From the docs:

    By default, a node is not considered an output. Set ``OUTPUT_NODE = True`` to specify that it is.

    Comfy Docs: https://docs.comfy.org/custom-nodes/backend/server_overview#output-node
    """
    is_deprecated: bool=False
    """Flags a node as deprecated, indicating to users that they should find alternatives to this node."""
    is_experimental: bool=False
    """Flags a node as experimental, informing users that it may change or not work as expected."""
    is_api_node: bool=False
    """Flags a node as an API node. See: https://docs.comfy.org/tutorials/api-nodes/overview."""
    not_idempotent: bool=False
    """Flags a node as not idempotent; when True, the node will run and not reuse the cached outputs when identical inputs are provided on a different node in the graph."""

    def validate(self):
        '''Validate the schema:
        - verify ids on inputs and outputs are unique - both internally and in relation to each other
        '''
        input_ids = [i.id for i in self.inputs] if self.inputs is not None else []
        output_ids = [o.id for o in self.outputs] if self.outputs is not None else []
        input_set = set(input_ids)
        output_set = set(output_ids)
        issues = []
        # verify ids are unique per list
        if len(input_set) != len(input_ids):
            issues.append(f"Input ids must be unique, but {[item for item, count in Counter(input_ids).items() if count > 1]} are not.")
        if len(output_set) != len(output_ids):
            issues.append(f"Output ids must be unique, but {[item for item, count in Counter(output_ids).items() if count > 1]} are not.")
        # verify ids are unique between lists
        intersection = input_set & output_set
        if len(intersection) > 0:
            issues.append(f"Ids must be unique between inputs and outputs, but {intersection} are not.")
        if len(issues) > 0:
            raise ValueError("\n".join(issues))

class Serializer:
    def __init_subclass__(cls, io_type: str, **kwargs):
        cls.io_type = io_type
        super().__init_subclass__(**kwargs)

    @classmethod
    def serialize(cls, o: Any) -> str:
        pass

    @classmethod
    def deserialize(cls, s: str) -> Any:
        pass


class classproperty(object):
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)


class ComfyNodeV3:
    """Common base class for all V3 nodes."""

    RELATIVE_PYTHON_MODULE = None
    SCHEMA = None
    
    # filled in during execution
    state: NodeState = None
    resources: Resources = None
    hidden: HiddenHolder = None

    @classmethod
    def GET_NODE_INFO_V3(cls) -> dict[str, Any]:
        schema = cls.GET_SCHEMA()
        # TODO: finish
        return None

    @classmethod
    @abstractmethod
    def DEFINE_SCHEMA(cls) -> SchemaV3:
        """
        Override this function with one that returns a SchemaV3 instance.
        """
        return None
    DEFINE_SCHEMA = None

    @classmethod
    @abstractmethod
    def execute(cls, **kwargs) -> NodeOutput:
        pass
    execute = None

    @classmethod
    def GET_SERIALIZERS(cls) -> list[Serializer]:
        return []

    def __init__(self):
        self.local_state: NodeStateLocal = None
        self.local_resources: ResourcesLocal = None
        self.__class__.VALIDATE_CLASS()

    @classmethod
    def VALIDATE_CLASS(cls):
        if not callable(cls.DEFINE_SCHEMA):
            raise Exception(f"No DEFINE_SCHEMA function was defined for node class {cls.__name__}.")
        if not callable(cls.execute):
            raise Exception(f"No execute function was defined for node class {cls.__name__}.")

    @classmethod
    def prepare_class_clone(cls, hidden_inputs: dict, *args, **kwargs) -> type[ComfyNodeV3]:
        """Creates clone of real node class to prevent monkey-patching."""
        c_type: type[ComfyNodeV3] = cls if is_class(cls) else type(cls)
        type_clone: type[ComfyNodeV3] = type(f"CLEAN_{c_type.__name__}", c_type.__bases__, {})
        # TODO: what parameters should be carried over?
        type_clone.SCHEMA = c_type.SCHEMA
        type_clone.hidden = HiddenHolder.from_dict(hidden_inputs) if hidden_inputs is not None else None
        # TODO: add anything we would want to expose inside node's execute function
        return type_clone

    #############################################
    # V1 Backwards Compatibility code
    #--------------------------------------------
    _DESCRIPTION = None
    @classproperty
    def DESCRIPTION(cls):
        if cls._DESCRIPTION is None:
            cls.GET_SCHEMA()
        return cls._DESCRIPTION

    _CATEGORY = None
    @classproperty
    def CATEGORY(cls):
        if cls._CATEGORY is None:
            cls.GET_SCHEMA()
        return cls._CATEGORY

    _EXPERIMENTAL = None
    @classproperty
    def EXPERIMENTAL(cls):
        if cls._EXPERIMENTAL is None:
            cls.GET_SCHEMA()
        return cls._EXPERIMENTAL

    _DEPRECATED = None
    @classproperty
    def DEPRECATED(cls):
        if cls._DEPRECATED is None:
            cls.GET_SCHEMA()
        return cls._DEPRECATED

    _API_NODE = None
    @classproperty
    def API_NODE(cls):
        if cls._API_NODE is None:
            cls.GET_SCHEMA()
        return cls._API_NODE

    _OUTPUT_NODE = None
    @classproperty
    def OUTPUT_NODE(cls):
        if cls._OUTPUT_NODE is None:
            cls.GET_SCHEMA()
        return cls._OUTPUT_NODE

    _INPUT_IS_LIST = None
    @classproperty
    def INPUT_IS_LIST(cls):
        if cls._INPUT_IS_LIST is None:
            cls.GET_SCHEMA()
        return cls._INPUT_IS_LIST
    _OUTPUT_IS_LIST = None

    @classproperty
    def OUTPUT_IS_LIST(cls):
        if cls._OUTPUT_IS_LIST is None:
            cls.GET_SCHEMA()
        return cls._OUTPUT_IS_LIST

    _RETURN_TYPES = None
    @classproperty
    def RETURN_TYPES(cls):
        if cls._RETURN_TYPES is None:
            cls.GET_SCHEMA()
        return cls._RETURN_TYPES

    _RETURN_NAMES = None
    @classproperty
    def RETURN_NAMES(cls):
        if cls._RETURN_NAMES is None:
            cls.GET_SCHEMA()
        return cls._RETURN_NAMES

    _OUTPUT_TOOLTIPS = None
    @classproperty
    def OUTPUT_TOOLTIPS(cls):
        if cls._OUTPUT_TOOLTIPS is None:
            cls.GET_SCHEMA()
        return cls._OUTPUT_TOOLTIPS

    _NOT_IDEMPOTENT = None
    @classproperty
    def NOT_IDEMPOTENT(cls):
        if cls._NOT_IDEMPOTENT is None:
            cls.GET_SCHEMA()
        return cls._NOT_IDEMPOTENT

    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls, include_hidden=True, return_schema=False) -> dict[str, dict] | tuple[dict[str, dict], SchemaV3]:
        schema = cls.DEFINE_SCHEMA()
        # for V1, make inputs be a dict with potential keys {required, optional, hidden}
        input = {
            "required": {}
        }
        if schema.inputs:
            for i in schema.inputs:
                key = "optional" if i.optional else "required"
                input.setdefault(key, {})[i.id] = (i.get_io_type_V1(), i.as_dict_V1())
        if schema.hidden and include_hidden:
            for hidden in schema.hidden:
                input.setdefault("hidden", {})[hidden.name] = (hidden.value,)
        if return_schema:
            return input, schema
        return input

    @classmethod
    def GET_SCHEMA(cls) -> SchemaV3:
        cls.VALIDATE_CLASS()
        schema = cls.DEFINE_SCHEMA()
        schema.validate()
        if cls._DESCRIPTION is None:
            cls._DESCRIPTION = schema.description
        if cls._CATEGORY is None:
            cls._CATEGORY = schema.category
        if cls._EXPERIMENTAL is None:
            cls._EXPERIMENTAL = schema.is_experimental
        if cls._DEPRECATED is None:
            cls._DEPRECATED = schema.is_deprecated
        if cls._API_NODE is None:
            cls._API_NODE = schema.is_api_node
        if cls._OUTPUT_NODE is None:
            cls._OUTPUT_NODE = schema.is_output_node
        if cls._INPUT_IS_LIST is None:
            cls._INPUT_IS_LIST = schema.is_input_list
        if cls._NOT_IDEMPOTENT is None:
            cls._NOT_IDEMPOTENT = schema.not_idempotent

        if cls._RETURN_TYPES is None:
            output = []
            output_name = []
            output_is_list = []
            output_tooltips = []
            if schema.outputs:
                for o in schema.outputs:
                    output.append(o.io_type)
                    output_name.append(o.display_name if o.display_name else o.io_type)
                    output_is_list.append(o.is_output_list)
                    output_tooltips.append(o.tooltip if o.tooltip else None)

            cls._RETURN_TYPES = output
            cls._RETURN_NAMES = output_name
            cls._OUTPUT_IS_LIST = output_is_list
            cls._OUTPUT_TOOLTIPS = output_tooltips
        cls.SCHEMA = schema
        return schema

    @classmethod
    def GET_NODE_INFO_V1(cls) -> dict[str, Any]:
        schema = cls.GET_SCHEMA()
        # get V1 inputs
        input = cls.INPUT_TYPES()

        # create separate lists from output fields
        output = []
        output_is_list = []
        output_name = []
        output_tooltips = []
        if schema.outputs:
            for o in schema.outputs:
                output.append(o.io_type)
                output_is_list.append(o.is_output_list)
                output_name.append(o.display_name if o.display_name else o.io_type)
                output_tooltips.append(o.tooltip if o.tooltip else None)

        info = NodeInfoV1(
            input=input,
            input_order={key: list(value.keys()) for (key, value) in input.items()},
            output=output,
            output_is_list=output_is_list,
            output_name=output_name,
            output_tooltips=output_tooltips,
            name=schema.node_id,
            display_name=schema.display_name,
            category=schema.category,
            description=schema.description,
            output_node=schema.is_output_node,
            deprecated=schema.is_deprecated,
            experimental=schema.is_experimental,
            api_node=schema.is_api_node,
            python_module=getattr(cls, "RELATIVE_PYTHON_MODULE", "nodes")
        )
        return asdict(info)
    #--------------------------------------------
    #############################################


class NodeOutput:
    '''
    Standardized output of a node; can pass in any number of args and/or a UIOutput into 'ui' kwarg.
    '''
    def __init__(self, *args: Any, ui: _UIOutput | dict=None, expand: dict=None, block_execution: str=None, **kwargs):
        self.args = args
        self.ui = ui
        self.expand = expand
        self.block_execution = block_execution
        # self.kwargs = kwargs

    @property
    def result(self):
        # TODO: use kwargs to refer to outputs by id + organize in proper order
        return self.args if len(self.args) > 0 else None

class _UIOutput(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def as_dict(self) -> dict:
        ... # TODO: finish

class TestNode(ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return SchemaV3(
        node_id="TestNode_v3",
        display_name="Test Node (V3)",
        category="v3_test",
        inputs=[Int.Input("my_int"),
                #AutoGrowDynamicInput("growing", Image.Input),
                Mask.Input("thing"),
                ],
        outputs=[Image.Output("image_output")],
        hidden=[Hidden.api_key_comfy_org, Hidden.auth_token_comfy_org, Hidden.unique_id]
    )

    @classmethod
    def execute(cls, **kwargs):
        pass

if __name__ == "__main__":
    print("hello there")
    inputs: list[InputV3] = [
        Int.Input("tessfes", widgetType=String.io_type),
        Int.Input("my_int"),
        Custom("XYZ").Input("xyz"),
        Custom("MODEL_M").Input("model1"),
        Image.Input("my_image"),
        Float.Input("my_float"),
        MultiType.Input("my_inputs", [String, Custom("MODEL_M"), Custom("XYZ")]),
    ]
    Custom("XYZ").Input()
    outputs: list[OutputV3] = [
        Image.Output("image"),
        Custom("XYZ").Output("xyz"),
    ]

    for c in inputs:
        if isinstance(c, MultiType):
            print(f"{c}, {type(c)}, {type(c).io_type}, {c.id}, {[x.io_type for x in c.io_types]}")
            print(c.get_io_type_V1())
        else:
            print(f"{c}, {type(c)}, {type(c).io_type}, {c.id}")

    for c in outputs:
        print(f"{c}, {type(c)}, {type(c).io_type}, {c.id}")

    zz = TestNode()
    print(zz.GET_NODE_INFO_V1())

    # aa = NodeInfoV1()
    # print(asdict(aa))
    # print(as_pruned_dict(aa))
