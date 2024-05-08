from __future__ import annotations

import dataclasses
import json
import logging
import os
import posixpath
import re
import sys
import uuid
from datetime import datetime
from typing import Sequence, Optional, TypedDict, Dict, List, Literal, Callable, Tuple

import PIL
import fsspec
import numpy as np
import torch
from PIL import Image, ImageSequence, ImageOps
from PIL.ImageFile import ImageFile
from PIL.PngImagePlugin import PngInfo
from fsspec.core import OpenFile
from fsspec.generic import GenericFileSystem
from fsspec.implementations.local import LocalFileSystem
from joblib import Parallel, delayed
from natsort import natsorted
from torch import Tensor

from comfy.cmd import folder_paths
from comfy.digest import digest
from comfy.nodes.package_typing import CustomNode, InputTypes, FunctionReturnsUIVariables, SaveNodeResult, \
    InputTypeSpec, ValidatedNodeResult

_open_api_common_schema: Dict[str, InputTypeSpec] = {
    "name": ("STRING", {}),
    "title": ("STRING", {"default": ""}),
    "description": ("STRING", {"default": "", "multiline": True}),
    "__required": ("BOOLEAN", {"default": True})
}

_common_image_metadatas = {
    "CreationDate": ("STRING", {"default": ""}),
    "Title": ("STRING", {"default": ""}),
    "Description": ("STRING", {"default": ""}),
    "Artist": ("STRING", {"default": ""}),
    "ImageNumber": ("STRING", {"default": ""}),
    "Rating": ("STRING", {"default": ""}),
    "UserComment": ("STRING", {"default": "", "multiline": True}),
}

_null_uri = "/dev/null"


def is_null_uri(local_uri):
    return local_uri == _null_uri or local_uri == "NUL"


class FsSpecComfyMetadata(TypedDict, total=True):
    prompt_json_str: str
    batch_number_str: str


class SaveNodeResultWithName(SaveNodeResult):
    name: str


@dataclasses.dataclass
class ExifContainer:
    exif: dict = dataclasses.field(default_factory=dict)

    def __getitem__(self, item: str):
        return self.exif[item]


class IntRequestParameter(CustomNode):

    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize})
            },
            "optional": {
                **_open_api_common_schema,
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = "api/openapi"

    def execute(self, value=0, *args, **kwargs) -> ValidatedNodeResult:
        return (value,)


class FloatRequestParameter(CustomNode):

    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value": ("FLOAT", {"default": 0})
            },
            "optional": {
                **_open_api_common_schema,
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = "api/openapi"

    def execute(self, value=0.0, *args, **kwargs) -> ValidatedNodeResult:
        return (value,)


class StringRequestParameter(CustomNode):

    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value": ("STRING", {"multiline": True})
            },
            "optional": {
                **_open_api_common_schema,
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/openapi"

    def execute(self, value="", *args, **kwargs) -> ValidatedNodeResult:
        return (value,)


class BooleanRequestParameter(CustomNode):

    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value": ("BOOLEAN", {"default": True})
            },
            "optional": {
                **_open_api_common_schema,
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/openapi"

    def execute(self, value: bool = True, *args, **kwargs) -> ValidatedNodeResult:
        return (value,)


class StringEnumRequestParameter(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return StringRequestParameter.INPUT_TYPES()

    RETURN_TYPES = ([],)
    FUNCTION = "execute"
    CATEGORY = "api/openapi"

    def execute(self, value: str, *args, **kwargs) -> ValidatedNodeResult:
        return (value,)


class HashImage(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE_HASHES",)
    FUNCTION = "execute"
    CATEGORY = "api/openapi"

    def execute(self, images: Sequence[Tensor]) -> ValidatedNodeResult:
        def process_image(image: Tensor) -> str:
            image_as_numpy_array: np.ndarray = 255. * image.cpu().numpy()
            image_as_numpy_array = np.ascontiguousarray(np.clip(image_as_numpy_array, 0, 255).astype(np.uint8))
            data = image_as_numpy_array.data
            try:
                image_bytes_digest = digest(data)
            finally:
                data.release()
            return image_bytes_digest

        hashes = Parallel(n_jobs=-1)(delayed(process_image)(image) for image in images)
        return (hashes,)


class StringPosixPathJoin(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                f"value{i}": ("STRING", {"default": "", "multiline": False}) for i in range(5)
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "api/openapi"

    def execute(self, *args: str, **kwargs) -> ValidatedNodeResult:
        sorted_keys = natsorted(kwargs.keys())
        return (posixpath.join(*[kwargs[key] for key in sorted_keys if kwargs[key] != ""]),)


class LegacyOutputURIs(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "images": ("IMAGE",),
                "prefix": ("STRING", {"default": "ComfyUI_"}),
                "suffix": ("STRING", {"default": "_.png"}),
            }
        }

    RETURN_TYPES = ("URIS",)
    FUNCTION = "execute"
    CATEGORY = "api/openapi"

    def execute(self, images: Sequence[Tensor], prefix: str = "ComfyUI_", suffix: str = "_.png") -> ValidatedNodeResult:
        output_directory = folder_paths.get_output_directory()
        pattern = rf'^{prefix}([\d]+){suffix}$'
        compiled_pattern = re.compile(pattern)
        matched_values = ["0"]

        # todo: use fcntl to lock a pattern while executing a job
        with os.scandir(output_directory) as entries:
            for entry in entries:
                match = compiled_pattern.match(entry.name)
                if entry.is_file() and match is not None:
                    matched_values.append(match.group(1))

        # find the highest value in the matched files
        highest_value = max(int(v, 10) for v in matched_values)
        # substitute batch number string
        # this is not going to produce exactly the same path names as SaveImage, but there's no reason to for %batch_num%
        uris = [os.path.join(output_directory, f'{prefix.replace("%batch_num%", str(i))}{highest_value + i + 1:05d}{suffix}') for i in range(len(images))]
        return (uris,)


class DevNullUris(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("URIS",)
    FUNCTION = "execute"
    CATEGORY = "api/openapi"

    def execute(self, images: Sequence[Tensor]) -> ValidatedNodeResult:
        return ([_null_uri] * len(images),)


class StringJoin(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        required = {f"value{i}": ("STRING", {"default": "", "multiline": True}) for i in range(5)}
        required["separator"] = ("STRING", {"default": "_"})
        return {
            "required": required
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "api/openapi"
    FUNCTION = "execute"

    def execute(self, separator: str = "_", *args: str, **kwargs) -> ValidatedNodeResult:
        sorted_keys = natsorted(kwargs.keys())
        return (separator.join([kwargs[key] for key in sorted_keys if kwargs[key] != ""]),)


class StringToUri(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value": ("STRING", {"default": "", "multiline": True}),
                "batch": ("INT", {"default": 1})
            }
        }

    RETURN_TYPES = ("URIS",)
    FUNCTION = "execute"
    CATEGORY = "api/openapi"

    def execute(self, value: str = "", batch: int = 1) -> ValidatedNodeResult:
        return ([value] * batch,)


class UriFormat(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "uri_template": ("STRING", {"default": "{output}/{uuid4}_{batch_index:05d}.png"}),
                "metadata_uri_extension": ("STRING", {"default": ".json"}),
                "image_hash_format_name": ("STRING", {"default": "image_hash"}),
                "uuid_format_name": ("STRING", {"default": "uuid4"}),
                "batch_index_format_name": ("STRING", {"default": "batch_index"}),
                "output_dir_format_name": ("STRING", {"default": "output"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "image_hashes": ("IMAGE_HASHES",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("URIS", "URIS")
    RETURN_NAMES = ("URIS (FILES)", "URIS (META)")
    FUNCTION = "execute"
    CATEGORY = "api/openapi"

    def execute(self,
                uri_template: str = "{output}/{uuid}_{batch_index:05d}.png",
                metadata_uri_extension: str = ".json",
                images: Optional[Sequence[Tensor]] | List[Literal[None]] = None,
                image_hashes: Optional[Sequence[str]] = None,
                output_dir_format_name: str = "output",
                image_hash_format_name: str = "image_hash",
                batch_index_format_name: str = "batch_index",
                uuid_format_name: str = "uuid",
                *args, **kwargs) -> Tuple[Sequence[str], Sequence[str]]:
        batch_indices = [0]
        if images is not None:
            batch_indices = list(range(len(images)))
        if image_hashes is None:
            image_hashes = [""] * len(batch_indices)
        if len(image_hashes) > len(batch_indices):
            batch_indices = list(range(len(image_hashes)))

        # trusted but not verified
        output_directory = folder_paths.get_output_directory()

        uris = []
        metadata_uris = []
        without_ext, ext = os.path.splitext(uri_template)
        metadata_uri_template = f"{without_ext}{metadata_uri_extension}"
        for batch_index, image_hash in zip(batch_indices, image_hashes):
            uuid_val = str(uuid.uuid4())
            format_vars = {
                image_hash_format_name: image_hash,
                uuid_format_name: uuid_val,
                batch_index_format_name: batch_index,
                output_dir_format_name: output_directory
            }
            uri = uri_template.format(**format_vars)
            metadata_uri = metadata_uri_template.format(**format_vars)

            uris.append(uri)
            metadata_uris.append(metadata_uri)

        return uris, metadata_uris


class ImageExifMerge(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                f"value{i}": ("EXIF",) for i in range(5)
            }
        }

    RETURN_TYPES = ("EXIF",)
    FUNCTION = "execute"
    CATEGORY = "api/openapi"

    def execute(self, **kwargs) -> ValidatedNodeResult:
        merges = [kwargs[key] for key in natsorted(kwargs.keys())]
        exifs_per_image = [list(group) for group in zip(*[pair for pair in merges])]
        result = []
        for exifs in exifs_per_image:
            new_exif = ExifContainer()
            exif: ExifContainer
            for exif in exifs:
                new_exif.exif.update({k: v for k, v in exif.exif.items() if v != ""})

            result.append(new_exif)
        return (result,)


class ImageExifCreationDateAndBatchNumber(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("EXIF",)
    FUNCTION = "execute"
    CATEGORY = "api/openapi"

    def execute(self, images: Sequence[Tensor]) -> ValidatedNodeResult:
        exifs = [ExifContainer({"ImageNumber": str(i), "CreationDate": datetime.now().strftime("%Y:%m:%d %H:%M:%S%z")}) for i in range(len(images))]
        return (exifs,)


class ImageExifBase:
    def execute(self, images: Sequence[Tensor] = (), *args, **metadata) -> ValidatedNodeResult:
        metadata = {k: v for k, v in metadata.items() if v != ""}
        exifs = [ExifContainer({**metadata}) for _ in images]
        return (exifs,)


class ImageExif(ImageExifBase, CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "images": ("IMAGE",),
                **_common_image_metadatas
            }
        }

    RETURN_TYPES = ("EXIF",)
    FUNCTION = "execute"
    CATEGORY = "api/openapi"


class ImageExifUncommon(ImageExifBase, CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "images": ("IMAGE",),
                **_common_image_metadatas,
                "Make": ("STRING", {"default": ""}),
                "Model": ("STRING", {"default": ""}),
                "ExposureTime": ("STRING", {"default": ""}),
                "FNumber": ("STRING", {"default": ""}),
                "ISO": ("STRING", {"default": ""}),
                "DateTimeOriginal": ("STRING", {"default": ""}),
                "ShutterSpeedValue": ("STRING", {"default": ""}),
                "ApertureValue": ("STRING", {"default": ""}),
                "BrightnessValue": ("STRING", {"default": ""}),
                "FocalLength": ("STRING", {"default": ""}),
                "MeteringMode": ("STRING", {"default": ""}),
                "Flash": ("STRING", {"default": ""}),
                "WhiteBalance": ("STRING", {"default": ""}),
                "ExposureMode": ("STRING", {"default": ""}),
                "DigitalZoomRatio": ("STRING", {"default": ""}),
                "FocalLengthIn35mmFilm": ("STRING", {"default": ""}),
                "SceneCaptureType": ("STRING", {"default": ""}),
                "GPSLatitude": ("STRING", {"default": ""}),
                "GPSLongitude": ("STRING", {"default": ""}),
                "GPSTimeStamp": ("STRING", {"default": ""}),
                "GPSAltitude": ("STRING", {"default": ""}),
                "LensMake": ("STRING", {"default": ""}),
                "LensModel": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("EXIF",)
    FUNCTION = "execute"
    CATEGORY = "api/openapi"


class SaveImagesResponse(CustomNode):

    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "images": ("IMAGE",),
                "uris": ("URIS",),
                "pil_save_format": ("STRING", {"default": "png"}),
            },
            "optional": {
                "exif": ("EXIF",),
                "metadata_uris": ("URIS",),
                "local_uris": ("URIS",),
                **_open_api_common_schema,
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE_RESULT",)
    CATEGORY = "api/openapi"

    def execute(self,
                name: str = "",
                images: Sequence[Tensor] = tuple(),
                uris: Sequence[str] = ("",),
                exif: Sequence[ExifContainer] = None,
                metadata_uris: Optional[Sequence[str | None]] = None,
                local_uris: Optional[Sequence[Optional[str]]] = None,
                pil_save_format="png",
                # from comfyui
                prompt: Optional[dict] = None,
                extra_pnginfo=None,
                *args,
                **kwargs,
                ) -> FunctionReturnsUIVariables:

        ui_images_result: ValidatedNodeResult = {"ui": {
            "images": []
        }}

        if metadata_uris is None:
            metadata_uris = [None] * len(images)
        if local_uris is None:
            local_uris = [None] * len(images)
        if exif is None:
            exif = [ExifContainer() for _ in range(len(images))]

        assert len(uris) == len(images) == len(metadata_uris) == len(local_uris) == len(exif), f"len(uris)={len(uris)} == len(images)={len(images)} == len(metadata_uris)={len(metadata_uris)} == len(local_uris)={len(local_uris)} == len(exif)={len(exif)}"

        image: Tensor
        uri: str
        metadata_uri: str | None
        local_uri: str | Callable[[bytearray | memoryview], str]

        images_ = ui_images_result["ui"]["images"]

        exif_inst: ExifContainer
        for batch_number, (image, uri, metadata_uri, local_uri, exif_inst) in enumerate(zip(images, uris, metadata_uris, local_uris, exif)):
            image_as_numpy_array: np.ndarray = 255. * image.cpu().numpy()
            image_as_numpy_array = np.ascontiguousarray(np.clip(image_as_numpy_array, 0, 255).astype(np.uint8))
            image_as_pil: PIL.Image = Image.fromarray(image_as_numpy_array)

            if prompt is not None and "prompt" not in exif_inst.exif:
                exif_inst.exif["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    exif_inst.exif[x] = json.dumps(extra_pnginfo[x])

            png_metadata = PngInfo()
            for tag, value in exif_inst.exif.items():
                png_metadata.add_text(tag, value)

            fsspec_metadata: FsSpecComfyMetadata = {
                "prompt_json_str": json.dumps(prompt, separators=(',', ':')),
                "batch_number_str": str(batch_number),
            }

            _, file_ext = os.path.splitext(uri)

            additional_args = {}
            if pil_save_format.lower() == "png":
                additional_args = {"pnginfo": png_metadata, "compress_level": 9}

            # save it to the local directory when None is passed with a random name
            output_directory = folder_paths.get_output_directory()
            test_open: OpenFile = fsspec.open(uri)
            fs: GenericFileSystem = test_open.fs
            uri_is_remote = not isinstance(fs, LocalFileSystem)

            local_uri: str
            if uri_is_remote and local_uri is None:
                filename_for_ui = f"{uuid.uuid4()}.png"
                local_uri = os.path.join(output_directory, filename_for_ui)
                subfolder = ""
            elif uri_is_remote and local_uri is not None:
                filename_for_ui = os.path.basename(local_uri)
                subfolder = self.subfolder_of(local_uri, output_directory)
            else:
                filename_for_ui = os.path.basename(uri)
                subfolder = self.subfolder_of(uri, output_directory) if os.path.isabs(uri) else os.path.dirname(uri)

            if not uri_is_remote and not os.path.isabs(uri):
                uri = os.path.join(output_directory, uri)
            abs_path = uri
            try:
                with fsspec.open(uri, mode="wb", auto_mkdir=True) as f:
                    image_as_pil.save(f, format=pil_save_format, **additional_args)
                if metadata_uri is not None:
                    # all values are stringified for the metadata
                    # in case these are going to be used as S3, google blob storage key-value tags
                    fsspec_metadata_img = {k: v for k, v in fsspec_metadata.items()}
                    fsspec_metadata_img.update(exif)

                    with fsspec.open(metadata_uri, mode="wt") as f:
                        json.dump(fsspec_metadata, f)
            except Exception as e:
                logging.error(f"Error while trying to save file with fsspec_url {uri}", exc_info=e)
                abs_path = os.path.abspath(local_uri)

            if is_null_uri(local_uri):
                filename_for_ui = ""
                subfolder = ""
            elif uri_is_remote:
                logging.debug(f"saving this uri locally: {local_uri}")
                os.makedirs(os.path.dirname(local_uri), exist_ok=True)
                image_as_pil.save(local_uri, format=pil_save_format, **additional_args)

            img_item: SaveNodeResultWithName = {
                "abs_path": str(abs_path),
                "filename": filename_for_ui,
                "subfolder": subfolder,
                "type": "output",
                "name": name
            }

            images_.append(img_item)
        if "ui" in ui_images_result and "images" in ui_images_result["ui"]:
            ui_images_result["result"] = ui_images_result["ui"]["images"]

        return ui_images_result

    def subfolder_of(self, local_uri, output_directory):
        return os.path.dirname(os.path.relpath(os.path.abspath(local_uri), os.path.abspath(output_directory)))


class ImageRequestParameter(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "uri": ("STRING", {"default": ""})
            },
            "optional": {
                **_open_api_common_schema,
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "api/openapi"

    def execute(self, uri: str = "", *args, **kwargs) -> ValidatedNodeResult:
        output_images = []

        with fsspec.open(uri, mode="rb") as f:
            # from LoadImage
            img = Image.open(f)
            for i in ImageSequence.Iterator(img):
                prev_value = None
                try:
                    i = ImageOps.exif_transpose(i)
                except OSError:
                    prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
                    ImageFile.LOAD_TRUNCATED_IMAGES = True
                    i = ImageOps.exif_transpose(i)
                finally:
                    if prev_value is not None:
                        ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
                    if i.mode == 'I':
                        i = i.point(lambda i: i * (1 / 255))
                    image = i.convert("RGB")
                    image = np.array(image).astype(np.float32) / 255.0
                    image = torch.from_numpy(image)[None,]
                    output_images.append(image)

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
        else:
            output_image = output_images[0]

        return (output_image,)


NODE_CLASS_MAPPINGS = {}
for cls in (
        IntRequestParameter,
        FloatRequestParameter,
        StringRequestParameter,
        StringEnumRequestParameter,
        BooleanRequestParameter,
        HashImage,
        StringPosixPathJoin,
        LegacyOutputURIs,
        DevNullUris,
        StringJoin,
        StringToUri,
        UriFormat,
        ImageExif,
        ImageExifMerge,
        ImageExifUncommon,
        ImageExifCreationDateAndBatchNumber,
        SaveImagesResponse,
        ImageRequestParameter
):
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
