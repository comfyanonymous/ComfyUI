from __future__ import annotations

import copy
import inspect
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Literal, TypedDict, TypeVar, TYPE_CHECKING
from typing_extensions import NotRequired, final

# used for type hinting
import torch

if TYPE_CHECKING:
    from spandrel import ImageModelDescriptor
    from comfy.clip_vision import ClipVisionModel
    from comfy.clip_vision import Output as ClipVisionOutput_
    from comfy.controlnet import ControlNet
    from comfy.hooks import HookGroup, HookKeyframeGroup
    from comfy.model_patcher import ModelPatcher
    from comfy.samplers import CFGGuider, Sampler
    from comfy.sd import CLIP, VAE
    from comfy.sd import StyleModel as StyleModel_
    from comfy_api.input import VideoInput
from comfy_api.internal import (_ComfyNodeInternal, _NodeOutputInternal, classproperty, copy_class, first_real_override, is_class,
    prune_dict, shallow_clone_class)
from comfy_api.latest._resources import Resources, ResourcesLocal
from comfy_execution.graph_utils import ExecutionBlocker

# from comfy_extras.nodes_images import SVG as SVG_ # NOTE: needs to be moved before can be imported due to circular reference

class FolderType(str, Enum):
    input = "input"
    output = "output"
    temp = "temp"


class UploadType(str, Enum):
    image = "image_upload"
    audio = "audio_upload"
    video = "video_upload"
    model = "file_upload"


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


class NumberDisplay(str, Enum):
    number = "number"
    slider = "slider"


class _StringIOType(str):
    def __ne__(self, value: object) -> bool:
        if self == "*" or value == "*":
            return False
        if not isinstance(value, str):
            return True
        a = frozenset(self.split(","))
        b = frozenset(value.split(","))
        return not (b.issubset(a) or a.issubset(b))

class _ComfyType(ABC):
    Type = Any
    io_type: str = None

# NOTE: this is a workaround to make the decorator return the correct type
T = TypeVar("T", bound=type)
def comfytype(io_type: str, **kwargs):
    '''
    Decorator to mark nested classes as ComfyType; io_type will be bound to the class.

    A ComfyType may have the following attributes:
    - Type = <type hint here>
    - class Input(Input): ...
    - class Output(Output): ...
    '''
    def decorator(cls: T) -> T:
        if isinstance(cls, _ComfyType) or issubclass(cls, _ComfyType):
            # clone Input and Output classes to avoid modifying the original class
            new_cls = cls
            if hasattr(new_cls, "Input"):
                new_cls.Input = copy_class(new_cls.Input)
            if hasattr(new_cls, "Output"):
                new_cls.Output = copy_class(new_cls.Output)
        else:
            # copy class attributes except for special ones that shouldn't be in type()
            cls_dict = {
                k: v for k, v in cls.__dict__.items()
                if k not in ('__dict__', '__weakref__', '__module__', '__doc__')
            }
            # new class
            new_cls: ComfyTypeIO = type(
                cls.__name__,
                (cls, ComfyTypeIO),
                cls_dict
            )
            # metadata preservation
            new_cls.__module__ = cls.__module__
            new_cls.__doc__ = cls.__doc__
            # assign ComfyType attributes, if needed
        # NOTE: use __ne__ trick for io_type (see node_typing.IO.__ne__ for details)
        new_cls.io_type = _StringIOType(io_type)
        if hasattr(new_cls, "Input") and new_cls.Input is not None:
            new_cls.Input.Parent = new_cls
        if hasattr(new_cls, "Output") and new_cls.Output is not None:
            new_cls.Output.Parent = new_cls
        return new_cls
    return decorator

def Custom(io_type: str) -> type[ComfyTypeIO]:
    '''Create a ComfyType for a custom io_type.'''
    @comfytype(io_type=io_type)
    class CustomComfyType(ComfyTypeIO):
        ...
    return CustomComfyType

class _IO_V3:
    '''
    Base class for V3 Inputs and Outputs.
    '''
    Parent: _ComfyType = None

    def __init__(self):
        pass

    @property
    def io_type(self):
        return self.Parent.io_type

    @property
    def Type(self):
        return self.Parent.Type

class Input(_IO_V3):
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

    def as_dict(self):
        return prune_dict({
            "display_name": self.display_name,
            "optional": self.optional,
            "tooltip": self.tooltip,
            "lazy": self.lazy,
        }) | prune_dict(self.extra_dict)

    def get_io_type(self):
        return _StringIOType(self.io_type)

class WidgetInput(Input):
    '''
    Base class for a V3 Input with widget.
    '''
    def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                 default: Any=None,
                 socketless: bool=None, widget_type: str=None, force_input: bool=None, extra_dict=None):
        super().__init__(id, display_name, optional, tooltip, lazy, extra_dict)
        self.default = default
        self.socketless = socketless
        self.widget_type = widget_type
        self.force_input = force_input

    def as_dict(self):
        return super().as_dict() | prune_dict({
            "default": self.default,
            "socketless": self.socketless,
            "widgetType": self.widget_type,
            "forceInput": self.force_input,
        })

    def get_io_type(self):
        return self.widget_type if self.widget_type is not None else super().get_io_type()


class Output(_IO_V3):
    def __init__(self, id: str=None, display_name: str=None, tooltip: str=None,
                 is_output_list=False):
        self.id = id
        self.display_name = display_name
        self.tooltip = tooltip
        self.is_output_list = is_output_list

    def as_dict(self):
        return prune_dict({
            "display_name": self.display_name,
            "tooltip": self.tooltip,
            "is_output_list": self.is_output_list,
        })

    def get_io_type(self):
        return self.io_type


class ComfyTypeI(_ComfyType):
    '''ComfyType subclass that only has a default Input class - intended for types that only have Inputs.'''
    class Input(Input):
        ...

class ComfyTypeIO(ComfyTypeI):
    '''ComfyType subclass that has default Input and Output classes; useful for types with both Inputs and Outputs.'''
    class Output(Output):
        ...


@comfytype(io_type="BOOLEAN")
class Boolean(ComfyTypeIO):
    Type = bool

    class Input(WidgetInput):
        '''Boolean input.'''
        def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                    default: bool=None, label_on: str=None, label_off: str=None,
                    socketless: bool=None, force_input: bool=None):
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, None, force_input)
            self.label_on = label_on
            self.label_off = label_off
            self.default: bool

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "label_on": self.label_on,
                "label_off": self.label_off,
            })

@comfytype(io_type="INT")
class Int(ComfyTypeIO):
    Type = int

    class Input(WidgetInput):
        '''Integer input.'''
        def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                    default: int=None, min: int=None, max: int=None, step: int=None, control_after_generate: bool=None,
                    display_mode: NumberDisplay=None, socketless: bool=None, force_input: bool=None):
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, None, force_input)
            self.min = min
            self.max = max
            self.step = step
            self.control_after_generate = control_after_generate
            self.display_mode = display_mode
            self.default: int

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "min": self.min,
                "max": self.max,
                "step": self.step,
                "control_after_generate": self.control_after_generate,
                "display": self.display_mode.value if self.display_mode else None,
            })

@comfytype(io_type="FLOAT")
class Float(ComfyTypeIO):
    Type = float

    class Input(WidgetInput):
        '''Float input.'''
        def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                    default: float=None, min: float=None, max: float=None, step: float=None, round: float=None,
                    display_mode: NumberDisplay=None, socketless: bool=None, force_input: bool=None):
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, None, force_input)
            self.min = min
            self.max = max
            self.step = step
            self.round = round
            self.display_mode = display_mode
            self.default: float

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "min": self.min,
                "max": self.max,
                "step": self.step,
                "round": self.round,
                "display": self.display_mode,
            })

@comfytype(io_type="STRING")
class String(ComfyTypeIO):
    Type = str

    class Input(WidgetInput):
        '''String input.'''
        def __init__(self, id: str, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                    multiline=False, placeholder: str=None, default: str=None, dynamic_prompts: bool=None,
                    socketless: bool=None, force_input: bool=None):
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless, None, force_input)
            self.multiline = multiline
            self.placeholder = placeholder
            self.dynamic_prompts = dynamic_prompts
            self.default: str

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "multiline": self.multiline,
                "placeholder": self.placeholder,
                "dynamicPrompts": self.dynamic_prompts,
            })

@comfytype(io_type="COMBO")
class Combo(ComfyTypeI):
    Type = str
    class Input(WidgetInput):
        """Combo input (dropdown)."""
        Type = str
        def __init__(self, id: str, options: list[str]=None, display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                    default: str=None, control_after_generate: bool=None,
                    upload: UploadType=None, image_folder: FolderType=None,
                    remote: RemoteOptions=None,
                    socketless: bool=None):
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless)
            self.multiselect = False
            self.options = options
            self.control_after_generate = control_after_generate
            self.upload = upload
            self.image_folder = image_folder
            self.remote = remote
            self.default: str

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "multiselect": self.multiselect,
                "options": self.options,
                "control_after_generate": self.control_after_generate,
                **({self.upload.value: True} if self.upload is not None else {}),
                "image_folder": self.image_folder.value if self.image_folder else None,
                "remote": self.remote.as_dict() if self.remote else None,
            })


@comfytype(io_type="COMBO")
class MultiCombo(ComfyTypeI):
    '''Multiselect Combo input (dropdown for selecting potentially more than one value).'''
    # TODO: something is wrong with the serialization, frontend does not recognize it as multiselect
    Type = list[str]
    class Input(Combo.Input):
        def __init__(self, id: str, options: list[str], display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None,
                    default: list[str]=None, placeholder: str=None, chip: bool=None, control_after_generate: bool=None,
                    socketless: bool=None):
            super().__init__(id, options, display_name, optional, tooltip, lazy, default, control_after_generate, socketless=socketless)
            self.multiselect = True
            self.placeholder = placeholder
            self.chip = chip
            self.default: list[str]

        def as_dict(self):
            to_return = super().as_dict() | prune_dict({
                "multi_select": self.multiselect,
                "placeholder": self.placeholder,
                "chip": self.chip,
            })
            return to_return

@comfytype(io_type="IMAGE")
class Image(ComfyTypeIO):
    Type = torch.Tensor


@comfytype(io_type="WAN_CAMERA_EMBEDDING")
class WanCameraEmbedding(ComfyTypeIO):
    Type = torch.Tensor


@comfytype(io_type="WEBCAM")
class Webcam(ComfyTypeIO):
    Type = str

    class Input(WidgetInput):
        """Webcam input."""
        Type = str
        def __init__(
                self, id: str, display_name: str=None, optional=False,
                tooltip: str=None, lazy: bool=None, default: str=None, socketless: bool=None
        ):
            super().__init__(id, display_name, optional, tooltip, lazy, default, socketless)


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
    if TYPE_CHECKING:
        Type = Sampler

@comfytype(io_type="SIGMAS")
class Sigmas(ComfyTypeIO):
    Type = torch.Tensor

@comfytype(io_type="NOISE")
class Noise(ComfyTypeIO):
    Type = torch.Tensor

@comfytype(io_type="GUIDER")
class Guider(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = CFGGuider

@comfytype(io_type="CLIP")
class Clip(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = CLIP

@comfytype(io_type="CONTROL_NET")
class ControlNet(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = ControlNet

@comfytype(io_type="VAE")
class Vae(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = VAE

@comfytype(io_type="MODEL")
class Model(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = ModelPatcher

@comfytype(io_type="CLIP_VISION")
class ClipVision(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = ClipVisionModel

@comfytype(io_type="CLIP_VISION_OUTPUT")
class ClipVisionOutput(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = ClipVisionOutput_

@comfytype(io_type="STYLE_MODEL")
class StyleModel(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = StyleModel_

@comfytype(io_type="GLIGEN")
class Gligen(ComfyTypeIO):
    '''ModelPatcher that wraps around a 'Gligen' model.'''
    if TYPE_CHECKING:
        Type = ModelPatcher

@comfytype(io_type="UPSCALE_MODEL")
class UpscaleModel(ComfyTypeIO):
    if TYPE_CHECKING:
        Type = ImageModelDescriptor

@comfytype(io_type="AUDIO")
class Audio(ComfyTypeIO):
    class AudioDict(TypedDict):
        waveform: torch.Tensor
        sampler_rate: int
    Type = AudioDict

@comfytype(io_type="VIDEO")
class Video(ComfyTypeIO):
    if TYPE_CHECKING:
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
    if TYPE_CHECKING:
        Type = HookGroup

@comfytype(io_type="HOOK_KEYFRAMES")
class HookKeyframes(ComfyTypeIO):
    if TYPE_CHECKING:
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
    class CameraInfo(TypedDict):
        position: dict[str, float | int]
        target: dict[str, float | int]
        zoom: int
        cameraType: str

    Type = CameraInfo


@comfytype(io_type="LOAD_3D")
class Load3D(ComfyTypeIO):
    """3D models are stored as a dictionary."""
    class Model3DDict(TypedDict):
        image: str
        mask: str
        normal: str
        camera_info: Load3DCamera.CameraInfo
        recording: NotRequired[str]

    Type = Model3DDict


@comfytype(io_type="LOAD_3D_ANIMATION")
class Load3DAnimation(Load3D):
    ...


@comfytype(io_type="PHOTOMAKER")
class Photomaker(ComfyTypeIO):
    Type = Any


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

@comfytype(io_type="*")
class AnyType(ComfyTypeIO):
    Type = Any

@comfytype(io_type="MODEL_PATCH")
class MODEL_PATCH(ComfyTypeIO):
    Type = Any

@comfytype(io_type="COMFY_MULTITYPED_V3")
class MultiType:
    Type = Any
    class Input(Input):
        '''
        Input that permits more than one input type; if `id` is an instance of `ComfyType.Input`, then that input will be used to create a widget (if applicable) with overridden values.
        '''
        def __init__(self, id: str | Input, types: list[type[_ComfyType] | _ComfyType], display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None, extra_dict=None):
            # if id is an Input, then use that Input with overridden values
            self.input_override = None
            if isinstance(id, Input):
                self.input_override = copy.copy(id)
                optional = id.optional if id.optional is True else optional
                tooltip = id.tooltip if id.tooltip is not None else tooltip
                display_name = id.display_name if id.display_name is not None else display_name
                lazy = id.lazy if id.lazy is not None else lazy
                id = id.id
                # if is a widget input, make sure widget_type is set appropriately
                if isinstance(self.input_override, WidgetInput):
                    self.input_override.widget_type = self.input_override.get_io_type()
            super().__init__(id, display_name, optional, tooltip, lazy, extra_dict)
            self._io_types = types

        @property
        def io_types(self) -> list[type[Input]]:
            '''
            Returns list of Input class types permitted.
            '''
            io_types = []
            for x in self._io_types:
                if not is_class(x):
                    io_types.append(type(x))
                else:
                    io_types.append(x)
            return io_types

        def get_io_type(self):
            # ensure types are unique and order is preserved
            str_types = [x.io_type for x in self.io_types]
            if self.input_override is not None:
                str_types.insert(0, self.input_override.get_io_type())
            return ",".join(list(dict.fromkeys(str_types)))

        def as_dict(self):
            if self.input_override is not None:
                return self.input_override.as_dict() | super().as_dict()
            else:
                return super().as_dict()

class DynamicInput(Input, ABC):
    '''
    Abstract class for dynamic input registration.
    '''
    @abstractmethod
    def get_dynamic(self) -> list[Input]:
        ...

class DynamicOutput(Output, ABC):
    '''
    Abstract class for dynamic output registration.
    '''
    def __init__(self, id: str=None, display_name: str=None, tooltip: str=None,
                 is_output_list=False):
        super().__init__(id, display_name, tooltip, is_output_list)

    @abstractmethod
    def get_dynamic(self) -> list[Output]:
        ...


@comfytype(io_type="COMFY_AUTOGROW_V3")
class AutogrowDynamic(ComfyTypeI):
    Type = list[Any]
    class Input(DynamicInput):
        def __init__(self, id: str, template_input: Input, min: int=1, max: int=None,
                     display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None, extra_dict=None):
            super().__init__(id, display_name, optional, tooltip, lazy, extra_dict)
            self.template_input = template_input
            if min is not None:
                assert(min >= 1)
            if max is not None:
                assert(max >= 1)
            self.min = min
            self.max = max

        def get_dynamic(self) -> list[Input]:
            curr_count = 1
            new_inputs = []
            for i in range(self.min):
                new_input = copy.copy(self.template_input)
                new_input.id = f"{new_input.id}{curr_count}_${self.id}_ag$"
                if new_input.display_name is not None:
                    new_input.display_name = f"{new_input.display_name}{curr_count}"
                new_input.optional = self.optional or new_input.optional
                if isinstance(self.template_input, WidgetInput):
                    new_input.force_input = True
                new_inputs.append(new_input)
                curr_count += 1
            # pretend to expand up to max
            for i in range(curr_count-1, self.max):
                new_input = copy.copy(self.template_input)
                new_input.id = f"{new_input.id}{curr_count}_${self.id}_ag$"
                if new_input.display_name is not None:
                    new_input.display_name = f"{new_input.display_name}{curr_count}"
                new_input.optional = True
                if isinstance(self.template_input, WidgetInput):
                    new_input.force_input = True
                new_inputs.append(new_input)
                curr_count += 1
            return new_inputs

@comfytype(io_type="COMFY_COMBODYNAMIC_V3")
class ComboDynamic(ComfyTypeI):
    class Input(DynamicInput):
        def __init__(self, id: str):
            pass

@comfytype(io_type="COMFY_MATCHTYPE_V3")
class MatchType(ComfyTypeIO):
    class Template:
        def __init__(self, template_id: str, allowed_types: _ComfyType | list[_ComfyType]):
            self.template_id = template_id
            self.allowed_types = [allowed_types] if isinstance(allowed_types, _ComfyType) else allowed_types

        def as_dict(self):
            return {
                "template_id": self.template_id,
                "allowed_types": "".join(t.io_type for t in self.allowed_types),
            }

    class Input(DynamicInput):
        def __init__(self, id: str, template: MatchType.Template,
                    display_name: str=None, optional=False, tooltip: str=None, lazy: bool=None, extra_dict=None):
            super().__init__(id, display_name, optional, tooltip, lazy, extra_dict)
            self.template = template

        def get_dynamic(self) -> list[Input]:
            return [self]

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "template": self.template.as_dict(),
            })

    class Output(DynamicOutput):
        def __init__(self, id: str, template: MatchType.Template, display_name: str=None, tooltip: str=None,
                     is_output_list=False):
            super().__init__(id, display_name, tooltip, is_output_list)
            self.template = template

        def get_dynamic(self) -> list[Output]:
            return [self]

        def as_dict(self):
            return super().as_dict() | prune_dict({
                "template": self.template.as_dict(),
            })


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
    def from_dict(cls, d: dict | None):
        if d is None:
            d = {}
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

@dataclass
class NodeInfoV3:
    input: dict=None
    output: dict=None
    hidden: list[str]=None
    name: str=None
    display_name: str=None
    description: str=None
    category: str=None
    output_node: bool=None
    deprecated: bool=None
    experimental: bool=None
    api_node: bool=None


@dataclass
class Schema:
    """Definition of V3 node properties."""

    node_id: str
    """ID of node - should be globally unique. If this is a custom node, add a prefix or postfix to avoid name clashes."""
    display_name: str = None
    """Display name of node."""
    category: str = "sd"
    """The category of the node, as per the "Add Node" menu."""
    inputs: list[Input]=None
    outputs: list[Output]=None
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
    enable_expand: bool=False
    """Flags a node as expandable, allowing NodeOutput to include 'expand' property."""

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

    def finalize(self):
        """Add hidden based on selected schema options, and give outputs without ids default ids."""
        # if is an api_node, will need key-related hidden
        if self.is_api_node:
            if self.hidden is None:
                self.hidden = []
            if Hidden.auth_token_comfy_org not in self.hidden:
                self.hidden.append(Hidden.auth_token_comfy_org)
            if Hidden.api_key_comfy_org not in self.hidden:
                self.hidden.append(Hidden.api_key_comfy_org)
        # if is an output_node, will need prompt and extra_pnginfo
        if self.is_output_node:
            if self.hidden is None:
                self.hidden = []
            if Hidden.prompt not in self.hidden:
                self.hidden.append(Hidden.prompt)
            if Hidden.extra_pnginfo not in self.hidden:
                self.hidden.append(Hidden.extra_pnginfo)
        # give outputs without ids default ids
        if self.outputs is not None:
            for i, output in enumerate(self.outputs):
                if output.id is None:
                    output.id = f"_{i}_{output.io_type}_"

    def get_v1_info(self, cls) -> NodeInfoV1:
        # get V1 inputs
        input = {
            "required": {}
        }
        if self.inputs:
            for i in self.inputs:
                if isinstance(i, DynamicInput):
                    dynamic_inputs = i.get_dynamic()
                    for d in dynamic_inputs:
                        add_to_dict_v1(d, input)
                else:
                    add_to_dict_v1(i, input)
        if self.hidden:
            for hidden in self.hidden:
                input.setdefault("hidden", {})[hidden.name] = (hidden.value,)
        # create separate lists from output fields
        output = []
        output_is_list = []
        output_name = []
        output_tooltips = []
        if self.outputs:
            for o in self.outputs:
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
            name=self.node_id,
            display_name=self.display_name,
            category=self.category,
            description=self.description,
            output_node=self.is_output_node,
            deprecated=self.is_deprecated,
            experimental=self.is_experimental,
            api_node=self.is_api_node,
            python_module=getattr(cls, "RELATIVE_PYTHON_MODULE", "nodes")
        )
        return info


    def get_v3_info(self, cls) -> NodeInfoV3:
        input_dict = {}
        output_dict = {}
        hidden_list = []
        # TODO: make sure dynamic types will be handled correctly
        if self.inputs:
            for input in self.inputs:
                add_to_dict_v3(input, input_dict)
        if self.outputs:
            for output in self.outputs:
                add_to_dict_v3(output, output_dict)
        if self.hidden:
            for hidden in self.hidden:
                hidden_list.append(hidden.value)

        info = NodeInfoV3(
            input=input_dict,
            output=output_dict,
            hidden=hidden_list,
            name=self.node_id,
            display_name=self.display_name,
            description=self.description,
            category=self.category,
            output_node=self.is_output_node,
            deprecated=self.is_deprecated,
            experimental=self.is_experimental,
            api_node=self.is_api_node,
            python_module=getattr(cls, "RELATIVE_PYTHON_MODULE", "nodes")
        )
        return info


def add_to_dict_v1(i: Input, input: dict):
    key = "optional" if i.optional else "required"
    as_dict = i.as_dict()
    # for v1, we don't want to include the optional key
    as_dict.pop("optional", None)
    input.setdefault(key, {})[i.id] = (i.get_io_type(), as_dict)

def add_to_dict_v3(io: Input | Output, d: dict):
    d[io.id] = (io.get_io_type(), io.as_dict())



class _ComfyNodeBaseInternal(_ComfyNodeInternal):
    """Common base class for storing internal methods and properties; DO NOT USE for defining nodes."""

    RELATIVE_PYTHON_MODULE = None
    SCHEMA = None

    # filled in during execution
    resources: Resources = None
    hidden: HiddenHolder = None

    @classmethod
    @abstractmethod
    def define_schema(cls) -> Schema:
        """Override this function with one that returns a Schema instance."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def execute(cls, **kwargs) -> NodeOutput:
        """Override this function with one that performs node's actions."""
        raise NotImplementedError

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool:
        """Optionally, define this function to validate inputs; equivalent to V1's VALIDATE_INPUTS."""
        raise NotImplementedError

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> Any:
        """Optionally, define this function to fingerprint inputs; equivalent to V1's IS_CHANGED."""
        raise NotImplementedError

    @classmethod
    def check_lazy_status(cls, **kwargs) -> list[str]:
        """Optionally, define this function to return a list of input names that should be evaluated.

        This basic mixin impl. requires all inputs.

        :kwargs: All node inputs will be included here.  If the input is ``None``, it should be assumed that it has not yet been evaluated.  \
            When using ``INPUT_IS_LIST = True``, unevaluated will instead be ``(None,)``.

        Params should match the nodes execution ``FUNCTION`` (self, and all inputs by name).
        Will be executed repeatedly until it returns an empty list, or all requested items were already evaluated (and sent as params).

        Comfy Docs: https://docs.comfy.org/custom-nodes/backend/lazy_evaluation#defining-check-lazy-status
        """
        return [name for name in kwargs if kwargs[name] is None]

    def __init__(self):
        self.local_resources: ResourcesLocal = None
        self.__class__.VALIDATE_CLASS()

    @classmethod
    def GET_BASE_CLASS(cls):
        return _ComfyNodeBaseInternal

    @final
    @classmethod
    def VALIDATE_CLASS(cls):
        if first_real_override(cls, "define_schema") is None:
            raise Exception(f"No define_schema function was defined for node class {cls.__name__}.")
        if first_real_override(cls, "execute") is None:
            raise Exception(f"No execute function was defined for node class {cls.__name__}.")

    @classproperty
    def FUNCTION(cls):  # noqa
        if inspect.iscoroutinefunction(cls.execute):
            return "EXECUTE_NORMALIZED_ASYNC"
        return "EXECUTE_NORMALIZED"

    @final
    @classmethod
    def EXECUTE_NORMALIZED(cls, *args, **kwargs) -> NodeOutput:
        to_return = cls.execute(*args, **kwargs)
        if to_return is None:
            to_return = NodeOutput()
        elif isinstance(to_return, NodeOutput):
            pass
        elif isinstance(to_return, tuple):
            to_return = NodeOutput(*to_return)
        elif isinstance(to_return, dict):
            to_return = NodeOutput.from_dict(to_return)
        elif isinstance(to_return, ExecutionBlocker):
            to_return = NodeOutput(block_execution=to_return.message)
        else:
            raise Exception(f"Invalid return type from node: {type(to_return)}")
        if to_return.expand is not None and not cls.SCHEMA.enable_expand:
            raise Exception(f"Node {cls.__name__} is not expandable, but expand included in NodeOutput; developer should set enable_expand=True on node's Schema to allow this.")
        return to_return

    @final
    @classmethod
    async def EXECUTE_NORMALIZED_ASYNC(cls, *args, **kwargs) -> NodeOutput:
        to_return = await cls.execute(*args, **kwargs)
        if to_return is None:
            to_return = NodeOutput()
        elif isinstance(to_return, NodeOutput):
            pass
        elif isinstance(to_return, tuple):
            to_return = NodeOutput(*to_return)
        elif isinstance(to_return, dict):
            to_return = NodeOutput.from_dict(to_return)
        elif isinstance(to_return, ExecutionBlocker):
            to_return = NodeOutput(block_execution=to_return.message)
        else:
            raise Exception(f"Invalid return type from node: {type(to_return)}")
        if to_return.expand is not None and not cls.SCHEMA.enable_expand:
            raise Exception(f"Node {cls.__name__} is not expandable, but expand included in NodeOutput; developer should set enable_expand=True on node's Schema to allow this.")
        return to_return

    @final
    @classmethod
    def PREPARE_CLASS_CLONE(cls, hidden_inputs: dict) -> type[ComfyNode]:
        """Creates clone of real node class to prevent monkey-patching."""
        c_type: type[ComfyNode] = cls if is_class(cls) else type(cls)
        type_clone: type[ComfyNode] = shallow_clone_class(c_type)
        # set hidden
        type_clone.hidden = HiddenHolder.from_dict(hidden_inputs)
        return type_clone

    @final
    @classmethod
    def GET_NODE_INFO_V3(cls) -> dict[str, Any]:
        schema = cls.GET_SCHEMA()
        info = schema.get_v3_info(cls)
        return asdict(info)
    #############################################
    # V1 Backwards Compatibility code
    #--------------------------------------------
    @final
    @classmethod
    def GET_NODE_INFO_V1(cls) -> dict[str, Any]:
        schema = cls.GET_SCHEMA()
        info = schema.get_v1_info(cls)
        return asdict(info)

    _DESCRIPTION = None
    @final
    @classproperty
    def DESCRIPTION(cls):  # noqa
        if cls._DESCRIPTION is None:
            cls.GET_SCHEMA()
        return cls._DESCRIPTION

    _CATEGORY = None
    @final
    @classproperty
    def CATEGORY(cls):  # noqa
        if cls._CATEGORY is None:
            cls.GET_SCHEMA()
        return cls._CATEGORY

    _EXPERIMENTAL = None
    @final
    @classproperty
    def EXPERIMENTAL(cls):  # noqa
        if cls._EXPERIMENTAL is None:
            cls.GET_SCHEMA()
        return cls._EXPERIMENTAL

    _DEPRECATED = None
    @final
    @classproperty
    def DEPRECATED(cls):  # noqa
        if cls._DEPRECATED is None:
            cls.GET_SCHEMA()
        return cls._DEPRECATED

    _API_NODE = None
    @final
    @classproperty
    def API_NODE(cls):  # noqa
        if cls._API_NODE is None:
            cls.GET_SCHEMA()
        return cls._API_NODE

    _OUTPUT_NODE = None
    @final
    @classproperty
    def OUTPUT_NODE(cls):  # noqa
        if cls._OUTPUT_NODE is None:
            cls.GET_SCHEMA()
        return cls._OUTPUT_NODE

    _INPUT_IS_LIST = None
    @final
    @classproperty
    def INPUT_IS_LIST(cls):  # noqa
        if cls._INPUT_IS_LIST is None:
            cls.GET_SCHEMA()
        return cls._INPUT_IS_LIST
    _OUTPUT_IS_LIST = None

    @final
    @classproperty
    def OUTPUT_IS_LIST(cls):  # noqa
        if cls._OUTPUT_IS_LIST is None:
            cls.GET_SCHEMA()
        return cls._OUTPUT_IS_LIST

    _RETURN_TYPES = None
    @final
    @classproperty
    def RETURN_TYPES(cls):  # noqa
        if cls._RETURN_TYPES is None:
            cls.GET_SCHEMA()
        return cls._RETURN_TYPES

    _RETURN_NAMES = None
    @final
    @classproperty
    def RETURN_NAMES(cls):  # noqa
        if cls._RETURN_NAMES is None:
            cls.GET_SCHEMA()
        return cls._RETURN_NAMES

    _OUTPUT_TOOLTIPS = None
    @final
    @classproperty
    def OUTPUT_TOOLTIPS(cls):  # noqa
        if cls._OUTPUT_TOOLTIPS is None:
            cls.GET_SCHEMA()
        return cls._OUTPUT_TOOLTIPS

    _NOT_IDEMPOTENT = None
    @final
    @classproperty
    def NOT_IDEMPOTENT(cls):  # noqa
        if cls._NOT_IDEMPOTENT is None:
            cls.GET_SCHEMA()
        return cls._NOT_IDEMPOTENT

    @final
    @classmethod
    def INPUT_TYPES(cls, include_hidden=True, return_schema=False) -> dict[str, dict] | tuple[dict[str, dict], Schema]:
        schema = cls.FINALIZE_SCHEMA()
        info = schema.get_v1_info(cls)
        input = info.input
        if not include_hidden:
            input.pop("hidden", None)
        if return_schema:
            return input, schema
        return input

    @final
    @classmethod
    def FINALIZE_SCHEMA(cls):
        """Call define_schema and finalize it."""
        schema = cls.define_schema()
        schema.finalize()
        return schema

    @final
    @classmethod
    def GET_SCHEMA(cls) -> Schema:
        """Validate node class, finalize schema, validate schema, and set expected class properties."""
        cls.VALIDATE_CLASS()
        schema = cls.FINALIZE_SCHEMA()
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
    #--------------------------------------------
    #############################################


class ComfyNode(_ComfyNodeBaseInternal):
    """Common base class for all V3 nodes."""

    @classmethod
    @abstractmethod
    def define_schema(cls) -> Schema:
        """Override this function with one that returns a Schema instance."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def execute(cls, **kwargs) -> NodeOutput:
        """Override this function with one that performs node's actions."""
        raise NotImplementedError

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool:
        """Optionally, define this function to validate inputs; equivalent to V1's VALIDATE_INPUTS."""
        raise NotImplementedError

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> Any:
        """Optionally, define this function to fingerprint inputs; equivalent to V1's IS_CHANGED."""
        raise NotImplementedError

    @classmethod
    def check_lazy_status(cls, **kwargs) -> list[str]:
        """Optionally, define this function to return a list of input names that should be evaluated.

        This basic mixin impl. requires all inputs.

        :kwargs: All node inputs will be included here.  If the input is ``None``, it should be assumed that it has not yet been evaluated.  \
            When using ``INPUT_IS_LIST = True``, unevaluated will instead be ``(None,)``.

        Params should match the nodes execution ``FUNCTION`` (self, and all inputs by name).
        Will be executed repeatedly until it returns an empty list, or all requested items were already evaluated (and sent as params).

        Comfy Docs: https://docs.comfy.org/custom-nodes/backend/lazy_evaluation#defining-check-lazy-status
        """
        return [name for name in kwargs if kwargs[name] is None]

    @final
    @classmethod
    def GET_BASE_CLASS(cls):
        """DO NOT override this class. Will break things in execution.py."""
        return ComfyNode


class NodeOutput(_NodeOutputInternal):
    '''
    Standardized output of a node; can pass in any number of args and/or a UIOutput into 'ui' kwarg.
    '''
    def __init__(self, *args: Any, ui: _UIOutput | dict=None, expand: dict=None, block_execution: str=None):
        self.args = args
        self.ui = ui
        self.expand = expand
        self.block_execution = block_execution

    @property
    def result(self):
        return self.args if len(self.args) > 0 else None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeOutput":
        args = ()
        ui = None
        expand = None
        if "result" in data:
            result = data["result"]
            if isinstance(result, ExecutionBlocker):
                return cls(block_execution=result.message)
            args = result
        if "ui" in data:
            ui = data["ui"]
        if "expand" in data:
            expand = data["expand"]
        return cls(args=args, ui=ui, expand=expand)

    def __getitem__(self, index) -> Any:
        return self.args[index]

class _UIOutput(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def as_dict(self) -> dict:
        ...


class _IO:
    FolderType = FolderType
    UploadType = UploadType
    RemoteOptions = RemoteOptions
    NumberDisplay = NumberDisplay

    comfytype = staticmethod(comfytype)
    Custom = staticmethod(Custom)
    Input = Input
    WidgetInput = WidgetInput
    Output = Output
    ComfyTypeI = ComfyTypeI
    ComfyTypeIO = ComfyTypeIO
    #---------------------------------
    # Supported Types
    Boolean = Boolean
    Int = Int
    Float = Float
    String = String
    Combo = Combo
    MultiCombo = MultiCombo
    Image = Image
    WanCameraEmbedding = WanCameraEmbedding
    Webcam = Webcam
    Mask = Mask
    Latent = Latent
    Conditioning = Conditioning
    Sampler = Sampler
    Sigmas = Sigmas
    Noise = Noise
    Guider = Guider
    Clip = Clip
    ControlNet = ControlNet
    Vae = Vae
    Model = Model
    ClipVision = ClipVision
    ClipVisionOutput = ClipVisionOutput
    StyleModel = StyleModel
    Gligen = Gligen
    UpscaleModel = UpscaleModel
    Audio = Audio
    Video = Video
    SVG = SVG
    LoraModel = LoraModel
    LossMap = LossMap
    Voxel = Voxel
    Mesh = Mesh
    Hooks = Hooks
    HookKeyframes = HookKeyframes
    TimestepsRange = TimestepsRange
    LatentOperation = LatentOperation
    FlowControl = FlowControl
    Accumulation = Accumulation
    Load3DCamera = Load3DCamera
    Load3D = Load3D
    Load3DAnimation = Load3DAnimation
    Photomaker = Photomaker
    Point = Point
    FaceAnalysis = FaceAnalysis
    BBOX = BBOX
    SEGS = SEGS
    AnyType = AnyType
    MultiType = MultiType
    #---------------------------------
    HiddenHolder = HiddenHolder
    Hidden = Hidden
    NodeInfoV1 = NodeInfoV1
    NodeInfoV3 = NodeInfoV3
    Schema = Schema
    ComfyNode = ComfyNode
    NodeOutput = NodeOutput
    add_to_dict_v1 = staticmethod(add_to_dict_v1)
    add_to_dict_v3 = staticmethod(add_to_dict_v3)
