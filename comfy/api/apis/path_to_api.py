import typing
import typing_extensions

from comfy.api.apis.paths.solidus import Solidus
from comfy.api.apis.paths.api_v1_images_digest import ApiV1ImagesDigest
from comfy.api.apis.paths.api_v1_prompts import ApiV1Prompts
from comfy.api.apis.paths.embeddings import Embeddings
from comfy.api.apis.paths.extensions import Extensions
from comfy.api.apis.paths.history import History
from comfy.api.apis.paths.interrupt import Interrupt
from comfy.api.apis.paths.object_info import ObjectInfo
from comfy.api.apis.paths.prompt import Prompt
from comfy.api.apis.paths.queue import Queue
from comfy.api.apis.paths.upload_image import UploadImage
from comfy.api.apis.paths.view import View

PathToApi = typing.TypedDict(
    'PathToApi',
    {
    "/": typing.Type[Solidus],
    "/api/v1/images/{digest}": typing.Type[ApiV1ImagesDigest],
    "/api/v1/prompts": typing.Type[ApiV1Prompts],
    "/embeddings": typing.Type[Embeddings],
    "/extensions": typing.Type[Extensions],
    "/history": typing.Type[History],
    "/interrupt": typing.Type[Interrupt],
    "/object_info": typing.Type[ObjectInfo],
    "/prompt": typing.Type[Prompt],
    "/queue": typing.Type[Queue],
    "/upload/image": typing.Type[UploadImage],
    "/view": typing.Type[View],
    }
)

path_to_api = PathToApi(
    {
    "/": Solidus,
    "/api/v1/images/{digest}": ApiV1ImagesDigest,
    "/api/v1/prompts": ApiV1Prompts,
    "/embeddings": Embeddings,
    "/extensions": Extensions,
    "/history": History,
    "/interrupt": Interrupt,
    "/object_info": ObjectInfo,
    "/prompt": Prompt,
    "/queue": Queue,
    "/upload/image": UploadImage,
    "/view": View,
    }
)
