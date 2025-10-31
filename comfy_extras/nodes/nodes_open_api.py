from __future__ import annotations

import dataclasses
import io
import json
import logging
import os
import posixpath
import re
import ssl
import sys
import uuid
from datetime import datetime
from fractions import Fraction
from typing import Sequence, Optional, TypedDict, List, Literal, Tuple, Any, Dict

import PIL
import aiohttp
import certifi
import cv2
import fsspec
import numpy as np
import torch
from PIL import Image, ImageSequence, ImageOps, ExifTags
from PIL.Image import Exif
from PIL.ImageFile import ImageFile
from PIL.PngImagePlugin import PngInfo
from fsspec.core import OpenFile
from fsspec.generic import GenericFileSystem
from fsspec.implementations.local import LocalFileSystem
from joblib import Parallel, delayed
from natsort import natsorted
from torch import Tensor

from comfy.cmd import folder_paths
from comfy.comfy_types import IO
from comfy.component_model.images_types import ImageMaskTuple
from comfy.component_model.tensor_types import RGBAImageBatch, RGBImageBatch, MaskBatch, ImageBatch
from comfy.digest import digest
from comfy.node_helpers import export_custom_nodes
from comfy.nodes.package_typing import CustomNode, InputTypes, FunctionReturnsUIVariables, SaveNodeResult, \
    InputTypeSpec, ValidatedNodeResult
from comfy.open_exr import mut_srgb_to_linear

logger = logging.getLogger(__name__)

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


async def get_client(**kwargs):
    """
    workaround for issues with fsspec on Windows
    :param kwargs:
    :return:
    """
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    conn = aiohttp.TCPConnector(ssl=ssl_context)
    return aiohttp.ClientSession(connector=conn, **kwargs)


class FsSpecComfyMetadata(TypedDict, total=True):
    prompt_json_str: str
    batch_number_str: str


# for keys that are missing
_PNGINFO_TO_EXIF_KEY_MAP = {
    "CreationDate": "DateTimeOriginal",
    "Title": "DocumentName",
    "Description": "ImageDescription",
}


class SaveNodeResultWithName(SaveNodeResult):
    name: str


def create_exif_from_pnginfo(metadata: Dict[str, Any]) -> Exif:
    """Convert PNG metadata dictionary to PIL Exif object"""
    exif = Exif()

    gps_data = {}
    for key, value in metadata.items():
        if key.startswith('GPS'):
            tag_name = key[3:]
            try:
                tag = getattr(ExifTags.GPS, tag_name)
                if tag_name in ('Latitude', 'Longitude', 'Altitude'):
                    decimal = float(value)
                    fraction = Fraction(decimal).limit_denominator(1000000)
                    gps_data[tag] = ((fraction.numerator, fraction.denominator),)
                else:
                    gps_data[tag] = value
            except (AttributeError, ValueError):
                continue

    if gps_data:
        gps_data[ExifTags.GPS.GPSVersionID] = (2, 2, 0, 0)
        if 'Latitude' in metadata:
            gps_data[ExifTags.GPS.GPSLatitudeRef] = 'N' if float(metadata['Latitude']) >= 0 else 'S'
        if 'Longitude' in metadata:
            gps_data[ExifTags.GPS.GPSLongitudeRef] = 'E' if float(metadata['Longitude']) >= 0 else 'W'
        if 'Altitude' in metadata:
            gps_data[ExifTags.GPS.GPSAltitudeRef] = 0  # Above sea level

        exif[ExifTags.Base.GPSInfo] = gps_data

    for key, value in metadata.items():
        if key.startswith('GPS'):
            continue

        exif_key = _PNGINFO_TO_EXIF_KEY_MAP.get(key, key)

        try:
            tag = getattr(ExifTags.Base, exif_key)
            exif[tag] = value
        except (AttributeError, ValueError):
            continue

    return exif


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
                "value": ("FLOAT", {"default": 0, "step": 0.00001, "round": 0.00001})
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

    RETURN_TYPES = ("BOOLEAN",)
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
                "images": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("IMAGE_HASHES",)
    FUNCTION = "execute"
    CATEGORY = "api/openapi"

    def execute(self, images: Sequence[Tensor]) -> ValidatedNodeResult:
        def process_image(image: Tensor) -> str:
            image_as_numpy_array: np.ndarray = 255. * image.float().cpu().numpy()
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
            "required": {},
            "optional": {
                f"value{i}": ("STRING", {"default": "", "multiline": False, "forceInput": True}) for i in range(5)
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
                "images": ("IMAGE", {}),
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
        optional = {f"value{i}": ("STRING", {"default": "", "multiline": True, "forceInput": True}) for i in range(5)}
        optional["separator"] = ("STRING", {"default": "_"})
        return {
            "required": {},
            "optional": optional
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "api/openapi"
    FUNCTION = "execute"

    def execute(self, separator: str = "_", *args: str, **kwargs) -> ValidatedNodeResult:
        sorted_keys = natsorted(kwargs.keys())
        return (separator.join([kwargs[key] for key in sorted_keys if kwargs[key] != ""]),)


class StringJoin1(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        optional = {f"value{i}": (IO.ANY, {}) for i in range(5)}
        optional["separator"] = (IO.STRING, {"default": "_"})
        return {
            "required": {},
            "optional": optional
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "api/openapi"
    FUNCTION = "execute"

    def execute(self, separator: str = "_", *args: str, **kwargs) -> ValidatedNodeResult:
        sorted_keys = natsorted(kwargs.keys())
        return (separator.join([str(kwargs[key]) for key in sorted_keys if kwargs[key] is not None]),)


class StringToUri(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
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
                "images": ("IMAGE", {}),
                "image_hashes": ("IMAGE_HASHES", {}),
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
            "required": {},
            "optional": {
                f"value{i}": ("EXIF", {}) for i in range(5)
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
                "images": ("IMAGE", {}),
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
                "images": ("IMAGE", {}),
            },
            "optional": {
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
                "images": ("IMAGE", {}),
            },
            "optional": {
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
                "exif": ("EXIF", {}),
                "metadata_uris": ("URIS", {}),
                "local_uris": ("URIS", {}),
                "bits": ([8, 16], {}),
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
                images: RGBImageBatch | RGBAImageBatch = tuple(),
                uris: Sequence[str] = ("",),
                exif: Sequence[ExifContainer] = None,
                metadata_uris: Optional[Sequence[str | None]] = None,
                local_uris: Optional[Sequence[Optional[str]]] = None,
                bits: int = 8,
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

        assert len(uris) == len(images) == len(metadata_uris) == len(local_uris) == len(exif), \
            f"len(uris)={len(uris)} == len(images)={len(images)} == len(metadata_uris)={len(metadata_uris)} == len(local_uris)={len(local_uris)} == len(exif)={len(exif)}"

        images_ = ui_images_result["ui"]["images"]

        for batch_number, (image, uri, metadata_uri, local_path, exif_inst) in enumerate(zip(images, uris, metadata_uris, local_uris, exif)):
            image_as_numpy_array: np.ndarray = image.float().cpu().numpy()

            cv_save_options = []
            image_as_pil: PIL.Image = None
            additional_args = {}
            if bits == 8:
                image_scaled = np.ascontiguousarray(np.clip(image_as_numpy_array * 255, 0, 255).astype(np.uint8))

                channels = image_scaled.shape[-1]
                if channels == 1:
                    mode = "L"
                elif channels == 3:
                    mode = "RGB"
                elif channels == 4:
                    mode = "RGBA"
                else:
                    raise ValueError(f"invalid channels {channels}")

                image_as_pil: PIL.Image = Image.fromarray(image_scaled, mode=mode)

                if prompt is not None and "prompt" not in exif_inst.exif:
                    exif_inst.exif["prompt"] = json.dumps(prompt)
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        exif_inst.exif[x] = json.dumps(extra_pnginfo[x])

                save_method = 'pil'
                save_format = pil_save_format
                if pil_save_format == 'png':
                    png_metadata = PngInfo()
                    for tag, value in exif_inst.exif.items():
                        png_metadata.add_text(tag, value)
                    additional_args = {"pnginfo": png_metadata, "compress_level": 9}
                else:
                    exif_obj = create_exif_from_pnginfo(exif_inst.exif)
                    additional_args = {"exif": exif_obj.tobytes()}

            elif bits >= 16:
                if 'exr' in pil_save_format:
                    image_as_numpy_array = image_as_numpy_array.copy()
                    mut_srgb_to_linear(image_as_numpy_array[:, :, :3])
                    image_scaled = image_as_numpy_array.astype(np.float32)
                    if bits == 16:
                        cv_save_options = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]
                else:
                    image_scaled = np.clip(image_as_numpy_array * 65535, 0, 65535).astype(np.uint16)

                # Ensure BGR color order for OpenCV
                if image_scaled.shape[-1] == 3:
                    image_scaled = image_scaled[..., ::-1]

                save_method = 'opencv'
                save_format = pil_save_format

            else:
                raise ValueError(f"invalid bits {bits}")

            # Prepare metadata
            fsspec_metadata: FsSpecComfyMetadata = {
                "prompt_json_str": json.dumps(prompt, separators=(',', ':')),
                "batch_number_str": str(batch_number),
            }

            output_directory = folder_paths.get_output_directory()
            test_open: OpenFile = fsspec.open(uri)
            fs: GenericFileSystem = test_open.fs
            uri_is_remote = not isinstance(fs, LocalFileSystem)

            if uri_is_remote and local_path is None:
                filename_for_ui = f"{uuid.uuid4()}.{save_format}"
                local_path = os.path.join(output_directory, filename_for_ui)
                subfolder = ""
            elif uri_is_remote and local_path is not None:
                filename_for_ui = os.path.basename(local_path)
                subfolder = self.subfolder_of(local_path, output_directory)
            else:
                filename_for_ui = os.path.basename(uri)
                subfolder = self.subfolder_of(uri, output_directory) if os.path.isabs(uri) else os.path.dirname(uri)

            if not uri_is_remote and not os.path.isabs(uri):
                uri = os.path.join(output_directory, uri)
            abs_path = uri

            fsspec_kwargs = {}
            if not uri_is_remote:
                fsspec_kwargs["auto_mkdir"] = True
            # todo: this might need special handling for s3 URLs too
            if uri.startswith("http"):
                fsspec_kwargs['get_client'] = get_client

            try:
                if save_method == 'pil':
                    with fsspec.open(uri, mode="wb", **fsspec_kwargs) as f:
                        image_as_pil.save(f, format=save_format, **additional_args)
                elif save_method == 'opencv':
                    _, img_encode = cv2.imencode(f'.{save_format}', image_scaled, cv_save_options)
                    img_bytes = img_encode.tobytes()

                    if exif_inst.exif and save_format == 'png':
                        import zlib
                        import struct
                        exif_obj = create_exif_from_pnginfo(exif_inst.exif)
                        # The eXIf chunk should contain the raw TIFF data, but Pillow's `tobytes()`
                        # includes the "Exif\x00\x00" prefix for JPEG APP1 markers. We must strip it.
                        exif_bytes = exif_obj.tobytes()[6:]
                        # PNG signature (8 bytes) + IHDR chunk (25 bytes) = 33 bytes.
                        insertion_point = 33
                        # Create eXIf chunk
                        exif_chunk = struct.pack('>I', len(exif_bytes)) + b'eXIf' + exif_bytes + struct.pack('>I', zlib.crc32(b'eXIf' + exif_bytes))
                        img_bytes = img_bytes[:insertion_point] + exif_chunk + img_bytes[insertion_point:]

                    with fsspec.open(uri, mode="wb", **fsspec_kwargs) as f:
                        f.write(img_bytes)

                if metadata_uri is not None:
                    # all values are stringified for the metadata
                    # in case these are going to be used as S3, google blob storage key-value tags
                    fsspec_metadata_img = {k: v for k, v in fsspec_metadata.items()}
                    fsspec_metadata_img.update(exif_inst.exif)

                    with fsspec.open(metadata_uri, mode="wt") as f:
                        json.dump(fsspec_metadata, f)

            except Exception as e:
                logging.error(f"Error while trying to save file with fsspec_url {uri}", exc_info=e)
                abs_path = "" if local_path is None else os.path.abspath(local_path)

            if is_null_uri(local_path):
                filename_for_ui = ""
                subfolder = ""
            # this results in a second file being saved - when a local path
            elif uri_is_remote:
                logging.debug(f"saving this uri locally : {local_path}")
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                if save_method == 'pil':
                    image_as_pil.save(local_path, format=save_format, **additional_args)
                else:
                    cv2.imwrite(local_path, image_scaled)

            img_item: SaveNodeResultWithName = {
                "abs_path": str(abs_path),
                "filename": filename_for_ui,
                "subfolder": subfolder,
                "type": "output",
                "name": name
            }

            images_.append(img_item)

        if "ui" in ui_images_result and "images" in ui_images_result["ui"]:
            ui_images_result["result"] = (ui_images_result["ui"]["images"],)

        return ui_images_result

    def subfolder_of(self, local_uri, output_directory):
        return os.path.dirname(os.path.relpath(os.path.abspath(local_uri), os.path.abspath(output_directory)))


class ImageRequestParameter(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value": ("STRING", {"default": ""})
            },
            "optional": {
                **_open_api_common_schema,
                "default_if_empty": ("IMAGE",),
                "alpha_is_transparency": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "execute"
    CATEGORY = "api/openapi"

    def execute(self, value: str = "", default_if_empty=None, alpha_is_transparency=False, *args, **kwargs) -> ImageMaskTuple:
        if value.strip() == "":
            return (default_if_empty,)
        output_images = []
        output_masks = []
        f: OpenFile
        fsspec_kwargs = {}
        if value.startswith('http'):
            fsspec_kwargs.update({
                "headers": {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.5672.64 Safari/537.36'
                },
                'get_client': get_client
            })
        # todo: additional security is needed here to prevent users from accessing local paths
        # however this generally needs to be done with user accounts on all OSes
        with fsspec.open_files(value, mode="rb", **fsspec_kwargs) as files:
            for f in files:
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
                        image = i.convert("RGBA" if alpha_is_transparency else "RGB")
                        image = np.array(image).astype(np.float32) / 255.0
                        image = torch.from_numpy(image)[None,]
                        if 'A' in i.getbands():
                            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                            mask = 1. - torch.from_numpy(mask)
                        elif i.mode == 'P' and 'transparency' in i.info:
                            mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                            mask = 1. - torch.from_numpy(mask)
                        else:
                            mask = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32, device="cpu")
                        output_images.append(image)
                        output_masks.append(mask.unsqueeze(0))

        output_images_batched: ImageBatch = torch.cat(output_images, dim=0)
        output_masks_batched: MaskBatch = torch.cat(output_masks, dim=0)

        return ImageMaskTuple(output_images_batched, output_masks_batched)


export_custom_nodes()
