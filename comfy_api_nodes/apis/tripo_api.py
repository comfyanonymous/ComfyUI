from __future__ import annotations
from enum import Enum
from typing import Optional, List, Dict, Any, Union

from pydantic import BaseModel, Field, RootModel

class TripoModelVersion(str, Enum):
    v2_5_20250123 = 'v2.5-20250123'
    v2_0_20240919 = 'v2.0-20240919'
    v1_4_20240625 = 'v1.4-20240625'


class TripoTextureQuality(str, Enum):
    standard = 'standard'
    detailed = 'detailed'


class TripoStyle(str, Enum):
    PERSON_TO_CARTOON = "person:person2cartoon"
    ANIMAL_VENOM = "animal:venom"
    OBJECT_CLAY = "object:clay"
    OBJECT_STEAMPUNK = "object:steampunk"
    OBJECT_CHRISTMAS = "object:christmas"
    OBJECT_BARBIE = "object:barbie"
    GOLD = "gold"
    ANCIENT_BRONZE = "ancient_bronze"
    NONE = "None"

class TripoTaskType(str, Enum):
    TEXT_TO_MODEL = "text_to_model"
    IMAGE_TO_MODEL = "image_to_model"
    MULTIVIEW_TO_MODEL = "multiview_to_model"
    TEXTURE_MODEL = "texture_model"
    REFINE_MODEL = "refine_model"
    ANIMATE_PRERIGCHECK = "animate_prerigcheck"
    ANIMATE_RIG = "animate_rig"
    ANIMATE_RETARGET = "animate_retarget"
    STYLIZE_MODEL = "stylize_model"
    CONVERT_MODEL = "convert_model"

class TripoTextureAlignment(str, Enum):
    ORIGINAL_IMAGE = "original_image"
    GEOMETRY = "geometry"

class TripoOrientation(str, Enum):
    ALIGN_IMAGE = "align_image"
    DEFAULT = "default"

class TripoOutFormat(str, Enum):
    GLB = "glb"
    FBX = "fbx"

class TripoTopology(str, Enum):
    BIP = "bip"
    QUAD = "quad"

class TripoSpec(str, Enum):
    MIXAMO = "mixamo"
    TRIPO = "tripo"

class TripoAnimation(str, Enum):
    IDLE = "preset:idle"
    WALK = "preset:walk"
    CLIMB = "preset:climb"
    JUMP = "preset:jump"
    RUN = "preset:run"
    SLASH = "preset:slash"
    SHOOT = "preset:shoot"
    HURT = "preset:hurt"
    FALL = "preset:fall"
    TURN = "preset:turn"

class TripoStylizeStyle(str, Enum):
    LEGO = "lego"
    VOXEL = "voxel"
    VORONOI = "voronoi"
    MINECRAFT = "minecraft"

class TripoConvertFormat(str, Enum):
    GLTF = "GLTF"
    USDZ = "USDZ"
    FBX = "FBX"
    OBJ = "OBJ"
    STL = "STL"
    _3MF = "3MF"

class TripoTextureFormat(str, Enum):
    BMP = "BMP"
    DPX = "DPX"
    HDR = "HDR"
    JPEG = "JPEG"
    OPEN_EXR = "OPEN_EXR"
    PNG = "PNG"
    TARGA = "TARGA"
    TIFF = "TIFF"
    WEBP = "WEBP"

class TripoTaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"
    BANNED = "banned"
    EXPIRED = "expired"

class TripoFileTokenReference(BaseModel):
    type: Optional[str] = Field(None, description='The type of the reference')
    file_token: str

class TripoUrlReference(BaseModel):
    type: Optional[str] = Field(None, description='The type of the reference')
    url: str

class TripoObjectStorage(BaseModel):
    bucket: str
    key: str

class TripoObjectReference(BaseModel):
    type: str
    object: TripoObjectStorage

class TripoFileEmptyReference(BaseModel):
    pass

class TripoFileReference(RootModel):
    root: Union[TripoFileTokenReference, TripoUrlReference, TripoObjectReference, TripoFileEmptyReference]

class TripoGetStsTokenRequest(BaseModel):
    format: str = Field(..., description='The format of the image')

class TripoTextToModelRequest(BaseModel):
    type: TripoTaskType = Field(TripoTaskType.TEXT_TO_MODEL, description='Type of task')
    prompt: str = Field(..., description='The text prompt describing the model to generate', max_length=1024)
    negative_prompt: Optional[str] = Field(None, description='The negative text prompt', max_length=1024)
    model_version: Optional[TripoModelVersion] = TripoModelVersion.v2_5_20250123
    face_limit: Optional[int] = Field(None, description='The number of faces to limit the generation to')
    texture: Optional[bool] = Field(True, description='Whether to apply texture to the generated model')
    pbr: Optional[bool] = Field(True, description='Whether to apply PBR to the generated model')
    image_seed: Optional[int] = Field(None, description='The seed for the text')
    model_seed: Optional[int] = Field(None, description='The seed for the model')
    texture_seed: Optional[int] = Field(None, description='The seed for the texture')
    texture_quality: Optional[TripoTextureQuality] = TripoTextureQuality.standard
    style: Optional[TripoStyle] = None
    auto_size: Optional[bool] = Field(False, description='Whether to auto-size the model')
    quad: Optional[bool] = Field(False, description='Whether to apply quad to the generated model')

class TripoImageToModelRequest(BaseModel):
    type: TripoTaskType = Field(TripoTaskType.IMAGE_TO_MODEL, description='Type of task')
    file: TripoFileReference = Field(..., description='The file reference to convert to a model')
    model_version: Optional[TripoModelVersion] = Field(None, description='The model version to use for generation')
    face_limit: Optional[int] = Field(None, description='The number of faces to limit the generation to')
    texture: Optional[bool] = Field(True, description='Whether to apply texture to the generated model')
    pbr: Optional[bool] = Field(True, description='Whether to apply PBR to the generated model')
    model_seed: Optional[int] = Field(None, description='The seed for the model')
    texture_seed: Optional[int] = Field(None, description='The seed for the texture')
    texture_quality: Optional[TripoTextureQuality] = TripoTextureQuality.standard
    texture_alignment: Optional[TripoTextureAlignment] = Field(TripoTextureAlignment.ORIGINAL_IMAGE, description='The texture alignment method')
    style: Optional[TripoStyle] = Field(None, description='The style to apply to the generated model')
    auto_size: Optional[bool] = Field(False, description='Whether to auto-size the model')
    orientation: Optional[TripoOrientation] = TripoOrientation.DEFAULT
    quad: Optional[bool] = Field(False, description='Whether to apply quad to the generated model')

class TripoMultiviewToModelRequest(BaseModel):
    type: TripoTaskType = TripoTaskType.MULTIVIEW_TO_MODEL
    files: List[TripoFileReference] = Field(..., description='The file references to convert to a model')
    model_version: Optional[TripoModelVersion] = Field(None, description='The model version to use for generation')
    orthographic_projection: Optional[bool] = Field(False, description='Whether to use orthographic projection')
    face_limit: Optional[int] = Field(None, description='The number of faces to limit the generation to')
    texture: Optional[bool] = Field(True, description='Whether to apply texture to the generated model')
    pbr: Optional[bool] = Field(True, description='Whether to apply PBR to the generated model')
    model_seed: Optional[int] = Field(None, description='The seed for the model')
    texture_seed: Optional[int] = Field(None, description='The seed for the texture')
    texture_quality: Optional[TripoTextureQuality] = TripoTextureQuality.standard
    texture_alignment: Optional[TripoTextureAlignment] = TripoTextureAlignment.ORIGINAL_IMAGE
    auto_size: Optional[bool] = Field(False, description='Whether to auto-size the model')
    orientation: Optional[TripoOrientation] = Field(TripoOrientation.DEFAULT, description='The orientation for the model')
    quad: Optional[bool] = Field(False, description='Whether to apply quad to the generated model')

class TripoTextureModelRequest(BaseModel):
    type: TripoTaskType = Field(TripoTaskType.TEXTURE_MODEL, description='Type of task')
    original_model_task_id: str = Field(..., description='The task ID of the original model')
    texture: Optional[bool] = Field(True, description='Whether to apply texture to the model')
    pbr: Optional[bool] = Field(True, description='Whether to apply PBR to the model')
    model_seed: Optional[int] = Field(None, description='The seed for the model')
    texture_seed: Optional[int] = Field(None, description='The seed for the texture')
    texture_quality: Optional[TripoTextureQuality] = Field(None, description='The quality of the texture')
    texture_alignment: Optional[TripoTextureAlignment] = Field(TripoTextureAlignment.ORIGINAL_IMAGE, description='The texture alignment method')

class TripoRefineModelRequest(BaseModel):
    type: TripoTaskType = Field(TripoTaskType.REFINE_MODEL, description='Type of task')
    draft_model_task_id: str = Field(..., description='The task ID of the draft model')

class TripoAnimatePrerigcheckRequest(BaseModel):
    type: TripoTaskType = Field(TripoTaskType.ANIMATE_PRERIGCHECK, description='Type of task')
    original_model_task_id: str = Field(..., description='The task ID of the original model')

class TripoAnimateRigRequest(BaseModel):
    type: TripoTaskType = Field(TripoTaskType.ANIMATE_RIG, description='Type of task')
    original_model_task_id: str = Field(..., description='The task ID of the original model')
    out_format: Optional[TripoOutFormat] = Field(TripoOutFormat.GLB, description='The output format')
    spec: Optional[TripoSpec] = Field(TripoSpec.TRIPO, description='The specification for rigging')

class TripoAnimateRetargetRequest(BaseModel):
    type: TripoTaskType = Field(TripoTaskType.ANIMATE_RETARGET, description='Type of task')
    original_model_task_id: str = Field(..., description='The task ID of the original model')
    animation: TripoAnimation = Field(..., description='The animation to apply')
    out_format: Optional[TripoOutFormat] = Field(TripoOutFormat.GLB, description='The output format')
    bake_animation: Optional[bool] = Field(True, description='Whether to bake the animation')

class TripoStylizeModelRequest(BaseModel):
    type: TripoTaskType = Field(TripoTaskType.STYLIZE_MODEL, description='Type of task')
    style: TripoStylizeStyle = Field(..., description='The style to apply to the model')
    original_model_task_id: str = Field(..., description='The task ID of the original model')
    block_size: Optional[int] = Field(80, description='The block size for stylization')

class TripoConvertModelRequest(BaseModel):
    type: TripoTaskType = Field(TripoTaskType.CONVERT_MODEL, description='Type of task')
    format: TripoConvertFormat = Field(..., description='The format to convert to')
    original_model_task_id: str = Field(..., description='The task ID of the original model')
    quad: Optional[bool] = Field(False, description='Whether to apply quad to the model')
    force_symmetry: Optional[bool] = Field(False, description='Whether to force symmetry')
    face_limit: Optional[int] = Field(10000, description='The number of faces to limit the conversion to')
    flatten_bottom: Optional[bool] = Field(False, description='Whether to flatten the bottom of the model')
    flatten_bottom_threshold: Optional[float] = Field(0.01, description='The threshold for flattening the bottom')
    texture_size: Optional[int] = Field(4096, description='The size of the texture')
    texture_format: Optional[TripoTextureFormat] = Field(TripoTextureFormat.JPEG, description='The format of the texture')
    pivot_to_center_bottom: Optional[bool] = Field(False, description='Whether to pivot to the center bottom')

class TripoTaskRequest(RootModel):
    root: Union[
        TripoTextToModelRequest,
        TripoImageToModelRequest,
        TripoMultiviewToModelRequest,
        TripoTextureModelRequest,
        TripoRefineModelRequest,
        TripoAnimatePrerigcheckRequest,
        TripoAnimateRigRequest,
        TripoAnimateRetargetRequest,
        TripoStylizeModelRequest,
        TripoConvertModelRequest
    ]

class TripoTaskOutput(BaseModel):
    model: Optional[str] = Field(None, description='URL to the model')
    base_model: Optional[str] = Field(None, description='URL to the base model')
    pbr_model: Optional[str] = Field(None, description='URL to the PBR model')
    rendered_image: Optional[str] = Field(None, description='URL to the rendered image')
    riggable: Optional[bool] = Field(None, description='Whether the model is riggable')

class TripoTask(BaseModel):
    task_id: str = Field(..., description='The task ID')
    type: Optional[str] = Field(None, description='The type of task')
    status: Optional[TripoTaskStatus] = Field(None, description='The status of the task')
    input: Optional[Dict[str, Any]] = Field(None, description='The input parameters for the task')
    output: Optional[TripoTaskOutput] = Field(None, description='The output of the task')
    progress: Optional[int] = Field(None, description='The progress of the task', ge=0, le=100)
    create_time: Optional[int] = Field(None, description='The creation time of the task')
    running_left_time: Optional[int] = Field(None, description='The estimated time left for the task')
    queue_position: Optional[int] = Field(None, description='The position in the queue')

class TripoTaskResponse(BaseModel):
    code: int = Field(0, description='The response code')
    data: TripoTask = Field(..., description='The task data')

class TripoGeneralResponse(BaseModel):
    code: int = Field(0, description='The response code')
    data: Dict[str, str] = Field(..., description='The task ID data')

class TripoBalanceData(BaseModel):
    balance: float = Field(..., description='The account balance')
    frozen: float = Field(..., description='The frozen balance')

class TripoBalanceResponse(BaseModel):
    code: int = Field(0, description='The response code')
    data: TripoBalanceData = Field(..., description='The balance data')

class TripoErrorResponse(BaseModel):
    code: int = Field(..., description='The error code')
    message: str = Field(..., description='The error message')
    suggestion: str = Field(..., description='The suggestion for fixing the error')
