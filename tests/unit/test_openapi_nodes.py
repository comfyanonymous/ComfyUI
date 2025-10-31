from __future__ import annotations

import platform

# noqa: E402
from comfy.cmd.main_pre import args
import os
import re
import uuid
from datetime import datetime

import cv2
import numpy as np
import pytest
import torch
from PIL import Image, ExifTags
from freezegun import freeze_time

from comfy.cmd import folder_paths
from comfy.component_model.executor_types import ValidateInputsTuple
from comfy_extras.nodes.nodes_open_api import SaveImagesResponse, IntRequestParameter, FloatRequestParameter, \
    StringRequestParameter, HashImage, StringPosixPathJoin, LegacyOutputURIs, DevNullUris, StringJoin, StringToUri, \
    UriFormat, ImageExifMerge, ImageExifCreationDateAndBatchNumber, ImageExif, ImageExifUncommon, \
    StringEnumRequestParameter, ExifContainer, BooleanRequestParameter, ImageRequestParameter

_image_1x1 = torch.zeros((1, 1, 1, 3), dtype=torch.float32, device="cpu")


def test_save_image_response(use_temporary_output_directory):
    assert SaveImagesResponse.INPUT_TYPES() is not None
    n = SaveImagesResponse()
    ui_node_ret_dict = n.execute(images=_image_1x1, uris=["with_prefix/1.png"], name="test")
    assert os.path.isfile(os.path.join(folder_paths.get_output_directory(), "with_prefix/1.png"))
    assert len(ui_node_ret_dict["result"]) == 1
    assert len(ui_node_ret_dict["ui"]["images"]) == 1
    image_result, = ui_node_ret_dict["result"]
    assert image_result[0]["filename"] == "1.png"
    assert image_result[0]["subfolder"] == "with_prefix"
    assert image_result[0]["name"] == "test"


def test_save_image_response_abs_local_uris(use_temporary_output_directory):
    assert SaveImagesResponse.INPUT_TYPES() is not None
    n = SaveImagesResponse()
    ui_node_ret_dict = n.execute(images=_image_1x1, uris=[os.path.join(folder_paths.get_output_directory(), "with_prefix/1.png")], name="test")
    assert os.path.isfile(os.path.join(folder_paths.get_output_directory(), "with_prefix/1.png"))
    assert len(ui_node_ret_dict["result"]) == 1
    assert len(ui_node_ret_dict["ui"]["images"]) == 1
    image_result, = ui_node_ret_dict["result"]
    assert image_result[0]["filename"] == "1.png"
    assert image_result[0]["subfolder"] == "with_prefix"
    assert image_result[0]["name"] == "test"


def test_save_image_response_remote_uris(use_temporary_output_directory):
    n = SaveImagesResponse()
    uri = "memory://some_folder/1.png"
    ui_node_ret_dict = n.execute(images=_image_1x1, uris=[uri])
    assert len(ui_node_ret_dict["result"]) == 1
    assert len(ui_node_ret_dict["ui"]["images"]) == 1
    image_result, = ui_node_ret_dict["result"]
    filename_ = image_result[0]["filename"]
    assert filename_ != "1.png"
    assert filename_ != ""
    assert uuid.UUID(filename_.replace(".png", "")) is not None
    assert os.path.isfile(os.path.join(folder_paths.get_output_directory(), filename_))
    assert image_result[0]["abs_path"] == uri
    assert image_result[0]["subfolder"] == ""


def test_save_exif(use_temporary_output_directory):
    n = SaveImagesResponse()
    filename = "with_prefix/2.png"
    n.execute(images=_image_1x1, uris=[filename], name="test", exif=[ExifContainer({
        "Title": "test title"
    })])
    filepath = os.path.join(folder_paths.get_output_directory(), filename)
    assert os.path.isfile(filepath)
    with Image.open(filepath) as img:
        assert img.info['Title'] == "test title"


def test_no_local_file():
    n = SaveImagesResponse()
    uri = "memory://some_folder/2.png"
    ui_node_ret_dict = n.execute(images=_image_1x1, uris=[uri], local_uris=["/dev/null"])
    assert len(ui_node_ret_dict["result"]) == 1
    assert len(ui_node_ret_dict["ui"]["images"]) == 1
    image_result, = ui_node_ret_dict["result"]
    assert image_result[0]["filename"] == ""
    assert not os.path.isfile(os.path.join(folder_paths.get_output_directory(), image_result[0]["filename"]))
    assert image_result[0]["abs_path"] == uri
    assert image_result[0]["subfolder"] == ""


def test_int_request_parameter():
    nt = IntRequestParameter.INPUT_TYPES()
    assert nt is not None
    n = IntRequestParameter()
    v, = n.execute(value=1, name="test")
    assert v == 1


def test_float_request_parameter():
    nt = FloatRequestParameter.INPUT_TYPES()
    assert nt is not None
    n = FloatRequestParameter()
    v, = n.execute(value=3.5, name="test", description="")
    assert v == 3.5


def test_string_request_parameter():
    nt = StringRequestParameter.INPUT_TYPES()
    assert nt is not None
    n = StringRequestParameter()
    v, = n.execute(value="test", name="test")
    assert v == "test"


def test_bool_request_parameter():
    nt = BooleanRequestParameter.INPUT_TYPES()
    assert nt is not None
    n = BooleanRequestParameter()
    v, = n.execute(value=True, name="test")
    assert v == True


async def test_string_enum_request_parameter():
    nt = StringEnumRequestParameter.INPUT_TYPES()
    assert nt is not None
    n = StringEnumRequestParameter()
    v, = n.execute(value="test", name="test")
    assert v == "test"
    prompt = {
        "1": {
            "inputs": {
                "value": "euler",
                "name": "sampler_name",
                "title": "KSampler Node Sampler",
                "description":
                    "This allows users to select a sampler for generating images with Latent Diffusion Models, including Stable Diffusion, ComfyUI, and SDXL. \n\nChange this only if explicitly requested by the user.\n\nList of sampler choice (this parameter): valid choices for scheduler (value for scheduler parameter).\n\n- euler: normal, karras, exponential, sgm_uniform, simple, ddim_uniform\n- euler_ancestral: normal, karras\n- heun: normal, karras\n- heunpp2: normal, karras\n- dpm_2: normal, karras\n- dpm_2_ancestral: normal, karras\n- lms: normal, karras\n- dpm_fast: normal, exponential\n- dpm_adaptive: normal, exponential\n- dpmpp_2s_ancestral: karras, exponential\n- dpmpp_sde: karras, exponential\n- dpmpp_sde_gpu: karras, exponential\n- dpmpp_2m: karras, sgm_uniform\n- dpmpp_2m_sde: karras, sgm_uniform\n- dpmpp_2m_sde_gpu: karras, sgm_uniform\n- dpmpp_3m_sde: karras, sgm_uniform\n- dpmpp_3m_sde_gpu: karras, sgm_uniform\n- ddpm: normal, simple\n- lcm: normal, exponential\n- ddim: normal, ddim_uniform\n- uni_pc: normal, karras, exponential\n- uni_pc_bh2: normal, karras, exponential",
                "__required": True,
            },
            "class_type": "StringEnumRequestParameter",
            "_meta": {
                "title": "StringEnumRequestParameter",
            },
        },
        "2": {
            "inputs": {
                "sampler_name": ["1", 0],
            },
            "class_type": "KSamplerSelect",
            "_meta": {
                "title": "KSamplerSelect",
            },
        },
    }
    from comfy.cmd.execution import validate_inputs
    validated: dict[str, ValidateInputsTuple] = {}
    prompt_id = str(uuid.uuid4())
    validated["1"] = await validate_inputs(prompt_id, prompt, "1", validated)
    validated["2"] = await validate_inputs(prompt_id, prompt, "2", validated)
    assert validated["2"].valid


@pytest.mark.skip("issues")
def test_hash_images():
    nt = HashImage.INPUT_TYPES()
    assert nt is not None
    n = HashImage()
    hashes, = n.execute(images=torch.cat([_image_1x1.clone(), _image_1x1.clone()]))
    # same image, same hash
    assert hashes[0] == hashes[1]
    # hash should be a valid sha256 hash
    p = re.compile(r'^[0-9a-fA-F]{64}$')
    for hash in hashes:
        assert p.match(hash)


def test_string_posix_path_join():
    nt = StringPosixPathJoin.INPUT_TYPES()
    assert nt is not None
    n = StringPosixPathJoin()
    joined_path, = n.execute(value2="c", value0="a", value1="b")
    assert joined_path == "a/b/c"


def test_legacy_output_uris(use_temporary_output_directory):
    nt = LegacyOutputURIs.INPUT_TYPES()
    assert nt is not None
    n = LegacyOutputURIs()
    images_ = torch.cat([_image_1x1.clone(), _image_1x1.clone()])
    output_paths, = n.execute(images=images_)
    # from SaveImage node
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path("ComfyUI", str(use_temporary_output_directory), images_[0].shape[1], images_[0].shape[0])
    file1 = f"{filename}_{counter:05}_.png"
    file2 = f"{filename}_{counter + 1:05}_.png"
    files = [file1, file2]
    assert os.path.basename(output_paths[0]) == files[0]
    assert os.path.basename(output_paths[1]) == files[1]


def test_null_uris():
    nt = DevNullUris.INPUT_TYPES()
    assert nt is not None
    n = DevNullUris()
    res, = n.execute(torch.cat([_image_1x1.clone(), _image_1x1.clone()]))
    assert all(x == "/dev/null" for x in res)


def test_string_join():
    assert StringJoin.INPUT_TYPES() is not None
    n = StringJoin()
    res, = n.execute(separator="*", value1="b", value3="c", value0="a")
    assert res == "a*b*c"


def test_string_to_uri():
    assert StringToUri.INPUT_TYPES() is not None
    n = StringToUri()
    res, = n.execute("x", batch=3)
    assert res == ["x"] * 3


def test_uri_format(use_temporary_output_directory):
    assert UriFormat.INPUT_TYPES() is not None
    n = UriFormat()
    images = torch.cat([_image_1x1.clone(), _image_1x1.clone()])
    # with defaults
    uris, metadata_uris = n.execute(images=images, uri_template="{output}/{uuid}_{batch_index:05d}.png")
    for uri in uris:
        assert os.path.isabs(uri), "uri format returns absolute URIs when output appears"
        assert os.path.commonpath([uri, use_temporary_output_directory]) == str(use_temporary_output_directory), "should be under output dir"
    uris, metadata_uris = n.execute(images=images, uri_template="{output}/{uuid}.png")
    for uri in uris:
        assert os.path.isabs(uri)
        assert os.path.commonpath([uri, use_temporary_output_directory]) == str(use_temporary_output_directory), "should be under output dir"

    with pytest.raises(KeyError):
        n.execute(images=images, uri_template="{xyz}.png")


def test_image_exif_merge():
    assert ImageExifMerge.INPUT_TYPES() is not None
    n = ImageExifMerge()
    res, = n.execute(value0=[ExifContainer({"a": "1"}), ExifContainer({"a": "1"})], value1=[ExifContainer({"b": "2"}), ExifContainer({"a": "1"})], value2=[ExifContainer({"a": 3}), ExifContainer({})], value4=[ExifContainer({"a": ""}), ExifContainer({})])
    assert res[0].exif["a"] == 3
    assert res[0].exif["b"] == "2"
    assert res[1].exif["a"] == "1"


@freeze_time("2024-01-14 03:21:34", tz_offset=-4)
@pytest.mark.skipif(True, reason="Time freezing not reliable on many platforms and interacts incorrectly with transformers")
def test_image_exif_creation_date_and_batch_number():
    assert ImageExifCreationDateAndBatchNumber.INPUT_TYPES() is not None
    n = ImageExifCreationDateAndBatchNumber()
    res, = n.execute(images=torch.cat([_image_1x1.clone(), _image_1x1.clone()]))
    mock_now = datetime(2024, 1, 13, 23, 21, 34)

    now_formatted = mock_now.strftime("%Y:%m:%d %H:%M:%S%z")
    assert res[0].exif["ImageNumber"] == "0"
    assert res[1].exif["ImageNumber"] == "1"
    assert res[0].exif["CreationDate"] == res[1].exif["CreationDate"] == now_formatted


def test_image_exif():
    assert ImageExif.INPUT_TYPES() is not None
    n = ImageExif()
    res, = n.execute(images=_image_1x1, Title="test", Artist="test2")
    assert res[0].exif["Title"] == "test"
    assert res[0].exif["Artist"] == "test2"


def test_image_exif_uncommon():
    assert "DigitalZoomRatio" in ImageExifUncommon.INPUT_TYPES()["optional"]
    ImageExifUncommon().execute(images=_image_1x1)


def test_posix_join_curly_brackets():
    n = StringPosixPathJoin()
    joined_path, = n.execute(value2="c", value0="a_{test}", value1="b")
    assert joined_path == "a_{test}/b/c"


def test_file_request_parameter(use_temporary_input_directory):
    _image_1x1_px = np.array([[[255, 0, 0]]], dtype=np.uint8)
    image_path = os.path.join(use_temporary_input_directory, "test_image.png")
    image = Image.fromarray(_image_1x1_px)
    image.save(image_path)

    n = ImageRequestParameter()
    loaded_image, *_ = n.execute(value=image_path)
    assert loaded_image.shape == (1, 1, 1, 3)
    from comfy.nodes.base_nodes import LoadImage

    load_image_node = LoadImage()
    load_image_node_rgb, _ = load_image_node.load_image(image=os.path.basename(image_path))

    assert loaded_image.shape == load_image_node_rgb.shape
    assert torch.allclose(loaded_image, load_image_node_rgb)


def test_file_request_parameter2(use_temporary_input_directory):
    n = ImageRequestParameter()

    # Test 1: Load a single RGB image
    _image_1x1_px_rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)  # 1x1 RGB
    image_path_rgb = os.path.join(use_temporary_input_directory, "test_image_rgb.png")
    image_rgb = Image.fromarray(_image_1x1_px_rgb, 'RGB')
    image_rgb.save(image_path_rgb)

    loaded_image_rgb, loaded_mask_rgb = n.execute(value=image_path_rgb, alpha_is_transparency=True)

    # Node converts RGB to RGBA
    assert loaded_image_rgb.shape == (1, 1, 1, 4)  # B, H, W, C
    # Check RGB values
    assert torch.allclose(loaded_image_rgb[0, 0, 0, :3], torch.tensor([1.0, 0.0, 0.0]))
    # Check added Alpha channel
    assert torch.allclose(loaded_image_rgb[0, 0, 0, 3], torch.tensor(1.0))
    # Check mask (should be all 0s for 1.0 alpha)
    assert loaded_mask_rgb.shape == (1, 1, 1)  # B, H, W
    assert torch.all(loaded_mask_rgb == 0.0)

    # Test 2: Load a single RGBA image with transparency
    _image_1x1_px_rgba = np.array([[[255, 0, 0, 128]]], dtype=np.uint8)  # 1x1 RGBA
    image_path_rgba = os.path.join(use_temporary_input_directory, "test_image_rgba.png")
    image_rgba = Image.fromarray(_image_1x1_px_rgba, 'RGBA')
    image_rgba.save(image_path_rgba)

    loaded_image_rgba, loaded_mask_rgba = n.execute(value=image_path_rgba, alpha_is_transparency=True)

    # Node should load RGBA as is
    assert loaded_image_rgba.shape == (1, 1, 1, 4)  # B, H, W, C
    # Check RGBA values
    assert torch.allclose(loaded_image_rgba[0, 0, 0, :], torch.tensor([1.0, 0.0, 0.0, 128 / 255.0]))
    # Check mask (should be 1.0 - alpha)
    assert loaded_mask_rgba.shape == (1, 1, 1)  # B, H, W
    assert torch.allclose(loaded_mask_rgba[0, 0, 0], torch.tensor(1.0 - 128 / 255.0))

    # Test 3: Load a single RGB image with alpha_is_transparency=False
    loaded_image_rgb_no_alpha, loaded_mask_rgb_no_alpha = n.execute(value=image_path_rgb, alpha_is_transparency=False)

    # Node converts to RGB
    assert loaded_image_rgb_no_alpha.shape == (1, 1, 1, 3)  # B, H, W, C
    # Check RGB values
    assert torch.allclose(loaded_image_rgb_no_alpha[0, 0, 0, :], torch.tensor([1.0, 0.0, 0.0]))
    # Check mask
    assert loaded_mask_rgb_no_alpha.shape == (1, 1, 1)  # B, H, W

    # Test 4: Load a single RGBA image with alpha_is_transparency=False
    loaded_image_rgba_no_alpha, loaded_mask_rgba_no_alpha = n.execute(value=image_path_rgba, alpha_is_transparency=False)

    # Node should load RGBA as RGB (dropping alpha)
    assert loaded_image_rgba_no_alpha.shape == (1, 1, 1, 3)  # B, H, W, C
    # Check RGB values (straight, not pre-multiplied)
    assert torch.allclose(loaded_image_rgba_no_alpha[0, 0, 0, :], torch.tensor([1.0, 0.0, 0.0]))
    assert loaded_mask_rgba_no_alpha.shape == (1, 1, 1)  # B, H, W


def test_file_request_parameter_glob(use_temporary_input_directory):
    # 1. Create dummy images (2x2)
    # Image 1 (RGB)
    img_rgb_data = np.array([
        [[255, 0, 0], [0, 255, 0]],
        [[0, 0, 255], [255, 255, 255]]
    ], dtype=np.uint8)
    img_rgb = Image.fromarray(img_rgb_data, 'RGB')
    path_rgb = os.path.join(use_temporary_input_directory, "img_rgb.png")
    img_rgb.save(path_rgb)

    # Image 2 (RGBA with transparency)
    img_rgba_data_rgb = np.array([
        [[10, 20, 30], [40, 50, 60]],
        [[70, 80, 90], [100, 110, 120]]
    ], dtype=np.uint8)
    img_rgba_data_a = np.array([
        [255, 128], # alpha 1.0, 0.5
        [0, 255]   # alpha 0.0, 1.0
    ], dtype=np.uint8)
    img_rgba_data = np.dstack((img_rgba_data_rgb, img_rgba_data_a))
    img_rgba = Image.fromarray(img_rgba_data, 'RGBA')
    path_rgba = os.path.join(use_temporary_input_directory, "img_rgba.png")
    img_rgba.save(path_rgba)

    # 2. Construct fsspec URL
    prefix = 'file:///' if platform.system() == "Windows" else 'file://'
    # Add glob pattern
    url = f"{prefix}{use_temporary_input_directory}/*.png"

    n = ImageRequestParameter()

    # 3. Test with alpha_is_transparency=True (default)
    loaded_images, loaded_masks = n.execute(value=url, alpha_is_transparency=True)

    # 4. Verify results (True) - Order independent
    assert loaded_images.shape == (2, 2, 2, 4)  # B=2, H=2, W=2, C=RGBA
    assert loaded_masks.shape == (2, 2, 2)  # B=2, H=2, W=2

    # Find which image is which by checking the mask sum
    mask_sums = torch.sum(loaded_masks, dim=(1, 2))
    expected_rgba_mask_sum = (1.0 - 128/255.0) + 1.0 # From alpha 128 and 0

    # Argmin should find the all-zero mask (from the RGB image)
    rgb_img_index = torch.argmin(mask_sums)
    # Argmax should find the mask with transparency
    rgba_img_index = torch.argmax(mask_sums)

    assert rgb_img_index != rgba_img_index
    assert torch.allclose(mask_sums[rgb_img_index], torch.tensor(0.0))
    assert torch.allclose(mask_sums[rgba_img_index], torch.tensor(expected_rgba_mask_sum))

    # Check RGB image tensor (which was converted to RGBA)
    rgb_image_tensor = loaded_images[rgb_img_index]
    assert torch.allclose(rgb_image_tensor[0, 0, :3], torch.tensor([1.0, 0.0, 0.0])) # Red pixel
    assert torch.allclose(rgb_image_tensor[0, 0, 3], torch.tensor(1.0)) # Added alpha
    assert torch.allclose(rgb_image_tensor[1, 1, :3], torch.tensor([1.0, 1.0, 1.0])) # White pixel
    assert torch.allclose(rgb_image_tensor[1, 1, 3], torch.tensor(1.0)) # Added alpha

    # Check RGBA image tensor
    rgba_image_tensor = loaded_images[rgba_img_index]
    # Pixel [0, 1] (alpha 128)
    assert torch.allclose(rgba_image_tensor[0, 1, :3], torch.tensor([40/255.0, 50/255.0, 60/255.0]))
    assert torch.allclose(rgba_image_tensor[0, 1, 3], torch.tensor(128/255.0)) # Original alpha
    # Pixel [1, 0] (alpha 0)
    assert torch.allclose(rgba_image_tensor[1, 0, :3], torch.tensor([70/255.0, 80/255.0, 90/255.0]))
    assert torch.allclose(rgba_image_tensor[1, 0, 3], torch.tensor(0.0)) # Original alpha
    assert torch.allclose(rgba_image_tensor[1, 0, 3], torch.tensor(0.0))  # Original alpha

    # 5. Test with alpha_is_transparency=False
    loaded_images_no_alpha, loaded_masks_no_alpha = n.execute(value=url, alpha_is_transparency=False)

    # 6. Verify results (False)
    assert loaded_images_no_alpha.shape == (2, 2, 2, 3)  # B=2, H=2, W=2, C=RGB
    assert loaded_masks_no_alpha.shape == (2, 2, 2)  # B=2, H, W (empty)

    # Find which image is which by checking pixel sum (RGB image has brighter pixels)
    img_sums = torch.sum(loaded_images_no_alpha, dim=(1, 2, 3))
    rgb_img_index = torch.argmax(img_sums)
    rgba_img_index = torch.argmin(img_sums)

    assert rgb_img_index != rgba_img_index

    # Check RGB image tensor
    rgb_image_tensor_no_alpha = loaded_images_no_alpha[rgb_img_index]
    assert torch.allclose(rgb_image_tensor_no_alpha[0, 0, :], torch.tensor([1.0, 0.0, 0.0]))  # Red
    assert torch.allclose(rgb_image_tensor_no_alpha[1, 1, :], torch.tensor([1.0, 1.0, 1.0]))  # White

    # Check RGBA image tensor (which was converted to RGB, dropping alpha - straight matte)
    rgba_image_tensor_no_alpha = loaded_images_no_alpha[rgba_img_index]
    # Pixel [0, 1] (RGB [40, 50, 60], alpha 128)
    # RGB channels are passed through unaltered
    expected_rgb_0_1 = torch.tensor([40 / 255.0, 50 / 255.0, 60 / 255.0])
    assert torch.allclose(rgba_image_tensor_no_alpha[0, 1, :], expected_rgb_0_1)
    # Pixel [1, 0] (RGB [70, 80, 90], alpha 0)
    # RGB channels are passed through unaltered
    expected_rgb_1_0 = torch.tensor([70 / 255.0, 80 / 255.0, 90 / 255.0])
    assert torch.allclose(rgba_image_tensor_no_alpha[1, 0, :], expected_rgb_1_0)

def test_file_request_to_http_url_no_exceptions():
    n = ImageRequestParameter()
    loaded_image, loaded_mask = n.execute(value="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/A_rainbow_at_sunset_after_rain_in_Gaziantep%2C_Turkey.IMG_2448.jpg/484px-A_rainbow_at_sunset_after_rain_in_Gaziantep%2C_Turkey.IMG_2448.jpg")
    # This is an RGB jpg, so it will be converted to RGBA
    b, height, width, channels = loaded_image.shape
    assert b == 1
    assert width == 484
    assert height == 480
    assert channels == 3
    # Mask should be all zeros
    assert loaded_mask.shape == (1, 480, 484)
    assert torch.all(loaded_mask == 0.0)


@pytest.mark.parametrize("format,bits,supports_16bit", [
    ("png", 8, True),
    ("png", 16, True),
    ("tiff", 8, True),
    ("tiff", 16, True),
    ("exr", 16, True),
    ("jpeg", 8, False),  # JPEG doesn't support 16-bit
    ("webp", 8, False),  # WebP doesn't support 16-bit
])
def test_save_image_bit_depth(format, bits, supports_16bit, use_temporary_output_directory):
    # Create a test image with known values
    test_tensor = torch.full((1, 8, 8, 3), 0.5, dtype=torch.float32)

    # Save the image
    node = SaveImagesResponse()
    filename = f"test_image.{format}"
    result = node.execute(
        images=test_tensor,
        uris=[filename],
        bits=bits,
        pil_save_format=format
    )

    # Construct full filepath
    filepath = os.path.join(folder_paths.get_output_directory(), filename)

    # Read image with OpenCV (supports 16-bit by default)
    if bits == 16 and supports_16bit:
        # Force 16-bit color depth for formats that support it
        read_flag = cv2.IMREAD_UNCHANGED
    else:
        # Use default 8-bit reading for 8-bit images or unsupported formats
        read_flag = cv2.IMREAD_COLOR

    saved_data = cv2.imread(filepath, read_flag)
    assert saved_data is not None, f"Failed to read image at {filepath}"

    # Special handling for EXR files (floating-point)
    if format == 'exr':
        # For EXR, expect direct comparison with original 0.2140 value, which is srgb to linear
        np.testing.assert_allclose(saved_data, 0.2140, rtol=1e-5, atol=1e-5)
        return

    # Calculate expected value based on bit depth
    if bits == 8 or not supports_16bit:
        expected_value = int(0.5 * 255)
        # Convert saved data to 8-bit if needed
        if saved_data.dtype == np.uint16:
            saved_data = (saved_data / 256).astype(np.uint8)
    else:  # 16-bit
        expected_value = int(0.5 * 65535)
        # Convert 8-bit data to 16-bit if needed
        if saved_data.dtype == np.uint8:
            saved_data = (saved_data.astype(np.uint16) * 256)

    # Check that all pixels are close to expected value
    # Allow small deviation due to compression
    if format in ['jpeg', 'webp']:
        # These formats use lossy compression, so be more lenient
        mean_diff = abs(float(saved_data.mean()) - float(expected_value))
        assert mean_diff < 5
    else:
        # For lossless formats, expect exact values
        pixel_diffs = np.abs(saved_data.astype(np.int32) - expected_value)
        assert np.all(pixel_diffs <= 1), f"Max difference was {pixel_diffs.max()}, expected at most 1"

    # Verify bit depth
    if supports_16bit and bits == 16:
        assert saved_data.dtype == np.uint16
    else:
        assert saved_data.dtype == np.uint8


@pytest.mark.parametrize("value", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_color_value_preservation(value, use_temporary_output_directory):
    """Test that floating point values are correctly scaled to integer color values"""
    test_tensor = torch.full((1, 64, 64, 3), value, dtype=torch.float32)

    node = SaveImagesResponse()

    # Test with PNG format (lossless)
    filename = "test_color.png"
    node.execute(
        images=test_tensor,
        uris=[filename],
        bits=8,
        pil_save_format="png"
    )

    # Load and verify
    filepath = f"{folder_paths.get_output_directory()}/{filename}"
    with Image.open(filepath) as img:
        saved_data = np.array(img)
        expected_value = int(value * 255)
        assert np.all(np.abs(saved_data - expected_value) <= 1)


def test_high_precision_tiff(use_temporary_output_directory):
    """Test that TIFF format preserves high precision values"""
    # Create a gradient image to test precision
    x = torch.linspace(0, 1, 256)
    y = torch.linspace(0, 1, 256)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    test_tensor = X.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)

    node = SaveImagesResponse()
    filename = "test_gradient.tiff"
    node.execute(
        images=test_tensor,
        uris=[filename],
        bits=16,
        pil_save_format="tiff"
    )

    # Load and verify
    filepath = os.path.join(folder_paths.get_output_directory(), filename)
    saved_data = cv2.imread(filepath, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    original_data = test_tensor[0].numpy()

    # Check that the gradient is preserved with high precision
    assert np.allclose(saved_data, original_data, atol=1.0 / 65535.0)


def test_alpha_channel_preservation(use_temporary_output_directory):
    """Test that alpha channel is preserved in formats that support it"""
    # Create RGBA test image
    test_tensor = torch.ones((1, 64, 64, 4), dtype=torch.float32) * 0.5

    node = SaveImagesResponse()

    # Test PNG with alpha
    filename = "test_alpha.png"
    node.execute(
        images=test_tensor,
        uris=[filename],
        bits=16,
        pil_save_format="png"
    )

    filepath = os.path.join(folder_paths.get_output_directory(), filename)
    saved_data = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    # Check alpha channel preservation
    assert saved_data.shape[-1] == 4  # Should have alpha channel
    expected_value = int(0.5 * 65535)
    assert np.all(np.abs(saved_data - expected_value) <= 1)


@pytest.mark.parametrize("format, bits, supports_16bit", [
    ("png", 8, True),
    ("png", 16, True),
    ("tiff", 8, True),
    # todo: we will worry about tiff 16 bit another time
    # ("tiff", 16, True),
    ("jpeg", 8, False),
    ("webp", 8, False),
])
def test_basic_exif(format, bits, supports_16bit, use_temporary_output_directory):
    """Test basic EXIF tags are correctly saved and loaded, including for 16-bit PNGs."""
    node = SaveImagesResponse()
    filename = f"test_exif_{bits}bit.{format}"

    # Create EXIF data with common tags, including Title and Description to test mapping
    exif = ExifContainer({
        "Artist": "Test Artist",
        "Copyright": "Test Copyright",
        "Title": "Test Title",
        "Description": "Test Description",
        "Make": "Test Camera",
        "Model": "Test Model",
        "Software": "Test Software",
    })

    # Save image with EXIF data
    node.execute(
        images=_image_1x1,
        uris=[filename],
        exif=[exif],
        pil_save_format=format,
        bits=bits
    )

    filepath = os.path.join(folder_paths.get_output_directory(), filename)

    # First, verify bit depth using OpenCV
    saved_data = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    assert saved_data is not None, f"Failed to read image at {filepath}"
    if supports_16bit and bits == 16:
        assert saved_data.dtype == np.uint16, f"Image should be 16-bit, but dtype is {saved_data.dtype}"
    else:
        assert saved_data.dtype == np.uint8, f"Image should be 8-bit, but dtype is {saved_data.dtype}"

    # Second, verify EXIF data using Pillow
    with Image.open(filepath) as img:
        # For 8-bit PNG, we use PIL's native text chunk saving.
        # For 16-bit PNG, we use a custom OpenCV path that injects a raw eXIf chunk.
        # For other formats, we use PIL's or a custom EXIF saving method.
        if format == "png" and bits == 8:
            # 8-bit PNG stores metadata in the 'info' dictionary as text chunks.
            info = img.info
            assert info.get("Artist") == "Test Artist"
            assert info.get("Copyright") == "Test Copyright"
            assert info.get("Title") == "Test Title"
            assert info.get("Description") == "Test Description"
            assert info.get("Make") == "Test Camera"
            assert info.get("Model") == "Test Model"
            assert info.get("Software") == "Test Software"
        else:
            # 16-bit PNGs (with eXIf), TIFFs, and other formats use the standard EXIF structure.
            exif_data = img.getexif()
            assert exif_data is not None, "EXIF data is missing."

            checked_tags = {
                "Artist": "Test Artist",
                "Copyright": "Test Copyright",
                # For formats that use full EXIF (like 16-bit PNG, JPEG, TIFF),
                # check that semantic names are mapped to standard EXIF tags.
                "DocumentName": "Test Title",  # Mapped from "Title"
                "ImageDescription": "Test Description",  # Mapped from "Description"
                "Make": "Test Camera",
                "Model": "Test Model",
                "Software": "Test Software",
            }

            # Reverse lookup for tag IDs
            tag_map = {name: key for key, name in ExifTags.TAGS.items()}

            for tag_name, expected_value in checked_tags.items():
                tag_id = tag_map.get(tag_name)
                assert tag_id is not None, f"Tag name '{tag_name}' is not a valid EXIF tag."
                assert tag_id in exif_data, f"Tag '{tag_name}' (ID: {tag_id}) not found in image EXIF data."
                assert exif_data[tag_id] == expected_value, f"Mismatch for tag '{tag_name}'."


@pytest.mark.parametrize("format", ["tiff", "jpeg", "webp"])
def test_gps_exif(format, use_temporary_output_directory):
    """Test GPS EXIF tags are correctly saved and loaded"""
    node = SaveImagesResponse()
    filename = f"test_gps.{format}"

    # Create EXIF data with GPS tags
    exif = ExifContainer({
        "GPSLatitude": "35.628611",
        "GPSLongitude": "139.738333",
        "GPSAltitude": "43.2",
        "GPSTimeStamp": "12:00:00",
    })

    # Save image with GPS EXIF data
    node.execute(
        images=_image_1x1,
        uris=[filename],
        exif=[exif],
        pil_save_format=format
    )

    # Load and verify GPS EXIF data
    filepath = os.path.join(folder_paths.get_output_directory(), filename)
    with Image.open(filepath) as img:
        exif_data = img.getexif()

        # Get GPS IFD
        if ExifTags.Base.GPSInfo in exif_data:
            gps_info = exif_data.get_ifd(ExifTags.Base.GPSInfo)

            # Verify GPS data
            # Note: GPS data might be stored in different formats depending on the image format
            assert gps_info.get(ExifTags.GPS.GPSLatitude) is not None
            assert gps_info.get(ExifTags.GPS.GPSLongitude) is not None
            if format == "tiff":  # TIFF tends to preserve exact values
                assert float(gps_info.get(ExifTags.GPS.GPSAltitude, "0")) == pytest.approx(43.2, rel=0.1)


@pytest.mark.parametrize("format,bits", [
    ("png", 8),
    ("png", 16),
    ("tiff", 8),
    ("jpeg", 8),
    ("webp", 8),
])
def test_datetime_exif(format, bits, use_temporary_output_directory):
    """Test DateTime EXIF tags are correctly saved and loaded"""
    node = SaveImagesResponse()
    filename = f"test_datetime_{bits}bit.{format}"

    # Fixed datetime string in EXIF format
    now = "2024:01:14 12:34:56"

    # Create EXIF data with datetime tags
    exif = ExifContainer({
        "DateTime": now,
        "CreationDate": now,  # This should be mapped to DateTimeOriginal for EXIF formats
        "DateTimeDigitized": now,
    })

    # Save image with datetime EXIF data
    node.execute(
        images=_image_1x1,
        uris=[filename],
        exif=[exif],
        pil_save_format=format,
        bits=bits
    )

    # Load and verify datetime EXIF data
    filepath = os.path.join(folder_paths.get_output_directory(), filename)
    with Image.open(filepath) as img:
        if format == "png" and bits == 8:
            # For 8-bit PNG, keys are saved as-is in text chunks
            assert img.info["DateTime"] == now
            assert img.info["CreationDate"] == now
            assert img.info["DateTimeDigitized"] == now
        else:
            exif_data = img.getexif()
            assert exif_data is not None, f"EXIF data is missing for {format} {bits}-bit."
            # For EXIF formats (including 16-bit PNG), CreationDate is mapped to DateTimeOriginal
            for tag_name in ["DateTime", "DateTimeOriginal", "DateTimeDigitized"]:
                tag_id = None
                for key, name in ExifTags.TAGS.items():
                    if name == tag_name:
                        tag_id = key
                        break
                assert tag_id is not None, f"Tag name '{tag_name}' is not a valid EXIF tag."
                assert tag_id in exif_data, f"Tag '{tag_name}' not found in EXIF for {format} {bits}-bit"
                assert exif_data[tag_id] == now


@pytest.mark.parametrize("format", ["tiff", "jpeg", "webp"])
def test_numeric_exif(format, use_temporary_output_directory):
    """Test numeric EXIF tags are correctly saved and loaded"""
    node = SaveImagesResponse()
    filename = f"test_numeric.{format}"

    # Create EXIF data with numeric tags
    exif = ExifContainer({
        "FNumber": "5.6",
        "ExposureTime": "1/125",
        "ISOSpeedRatings": "400",
        "FocalLength": "50",
    })

    # Save image with numeric EXIF data
    node.execute(
        images=_image_1x1,
        uris=[filename],
        exif=[exif],
        pil_save_format=format
    )

    # Load and verify numeric EXIF data
    filepath = os.path.join(folder_paths.get_output_directory(), filename)
    with Image.open(filepath) as img:
        exif_data = img.getexif()

        for tag_name, expected_value in [
            ("FNumber", "5.6"),
            ("ExposureTime", "1/125"),
            ("ISOSpeedRatings", "400"),
            ("FocalLength", "50"),
        ]:
            tag_id = None
            for key, name in ExifTags.TAGS.items():
                if name == tag_name:
                    tag_id = key
                    break
            assert tag_id is not None
            if tag_id in exif_data:
                # Convert both to strings for comparison since formats might store numbers differently
                assert str(exif_data[tag_id]) == expected_value
