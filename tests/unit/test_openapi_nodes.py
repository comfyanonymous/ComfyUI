import os
import re
import uuid
from datetime import datetime

import numpy as np
import pytest
import torch
from PIL import Image
from freezegun import freeze_time

from comfy.cmd import folder_paths
from comfy_extras.nodes.nodes_open_api import SaveImagesResponse, IntRequestParameter, FloatRequestParameter, \
    StringRequestParameter, HashImage, StringPosixPathJoin, LegacyOutputURIs, DevNullUris, StringJoin, StringToUri, \
    UriFormat, ImageExifMerge, ImageExifCreationDateAndBatchNumber, ImageExif, ImageExifUncommon, \
    StringEnumRequestParameter, ExifContainer, BooleanRequestParameter, ImageRequestParameter

_image_1x1 = torch.zeros((1, 1, 3), dtype=torch.float32, device="cpu")


def test_save_image_response():
    assert SaveImagesResponse.INPUT_TYPES() is not None
    n = SaveImagesResponse()
    result = n.execute(images=[_image_1x1], uris=["with_prefix/1.png"], name="test")
    assert os.path.isfile(os.path.join(folder_paths.get_output_directory(), "with_prefix/1.png"))
    assert len(result["result"]) == 1
    assert len(result["ui"]["images"]) == 1
    assert result["result"][0]["filename"] == "1.png"
    assert result["result"][0]["subfolder"] == "with_prefix"
    assert result["result"][0]["name"] == "test"


def test_save_image_response_abs_local_uris():
    assert SaveImagesResponse.INPUT_TYPES() is not None
    n = SaveImagesResponse()
    result = n.execute(images=[_image_1x1], uris=[os.path.join(folder_paths.get_output_directory(), "with_prefix/1.png")], name="test")
    assert os.path.isfile(os.path.join(folder_paths.get_output_directory(), "with_prefix/1.png"))
    assert len(result["result"]) == 1
    assert len(result["ui"]["images"]) == 1
    assert result["result"][0]["filename"] == "1.png"
    assert result["result"][0]["subfolder"] == "with_prefix"
    assert result["result"][0]["name"] == "test"


def test_save_image_response_remote_uris():
    n = SaveImagesResponse()
    uri = "memory://some_folder/1.png"
    result = n.execute(images=[_image_1x1], uris=[uri])
    assert len(result["result"]) == 1
    assert len(result["ui"]["images"]) == 1
    filename_ = result["result"][0]["filename"]
    assert filename_ != "1.png"
    assert filename_ != ""
    assert uuid.UUID(filename_.replace(".png", "")) is not None
    assert os.path.isfile(os.path.join(folder_paths.get_output_directory(), filename_))
    assert result["result"][0]["abs_path"] == uri
    assert result["result"][0]["subfolder"] == ""


def test_save_exif():
    n = SaveImagesResponse()
    filename = "with_prefix/2.png"
    n.execute(images=[_image_1x1], uris=[filename], name="test", exif=[ExifContainer({
        "Title": "test title"
    })])
    filepath = os.path.join(folder_paths.get_output_directory(), filename)
    assert os.path.isfile(filepath)
    with Image.open(filepath) as img:
        assert img.info['Title'] == "test title"


def test_no_local_file():
    n = SaveImagesResponse()
    uri = "memory://some_folder/2.png"
    result = n.execute(images=[_image_1x1], uris=[uri], local_uris=["/dev/null"])
    assert len(result["result"]) == 1
    assert len(result["ui"]["images"]) == 1
    assert result["result"][0]["filename"] == ""
    assert not os.path.isfile(os.path.join(folder_paths.get_output_directory(), result["result"][0]["filename"]))
    assert result["result"][0]["abs_path"] == uri
    assert result["result"][0]["subfolder"] == ""


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


def test_string_enum_request_parameter():
    nt = StringEnumRequestParameter.INPUT_TYPES()
    assert nt is not None
    n = StringEnumRequestParameter()
    v, = n.execute(value="test", name="test")
    assert v == "test"
    # todo: check that a graph that uses this in a checkpoint is valid


@pytest.mark.skip("issues")
def test_hash_images():
    nt = HashImage.INPUT_TYPES()
    assert nt is not None
    n = HashImage()
    hashes, = n.execute(images=[_image_1x1.clone(), _image_1x1.clone()])
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
    images_ = [_image_1x1, _image_1x1]
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
    res, = n.execute([_image_1x1, _image_1x1])
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
    images = [_image_1x1, _image_1x1]
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
    res, = n.execute(images=[_image_1x1, _image_1x1])
    mock_now = datetime(2024, 1, 13, 23, 21, 34)

    now_formatted = mock_now.strftime("%Y:%m:%d %H:%M:%S%z")
    assert res[0].exif["ImageNumber"] == "0"
    assert res[1].exif["ImageNumber"] == "1"
    assert res[0].exif["CreationDate"] == res[1].exif["CreationDate"] == now_formatted


def test_image_exif():
    assert ImageExif.INPUT_TYPES() is not None
    n = ImageExif()
    res, = n.execute(images=[_image_1x1], Title="test", Artist="test2")
    assert res[0].exif["Title"] == "test"
    assert res[0].exif["Artist"] == "test2"


def test_image_exif_uncommon():
    assert "DigitalZoomRatio" in ImageExifUncommon.INPUT_TYPES()["required"]
    ImageExifUncommon().execute(images=[_image_1x1])


def test_posix_join_curly_brackets():
    n = StringPosixPathJoin()
    joined_path, = n.execute(value2="c", value0="a_{test}", value1="b")
    assert joined_path == "a_{test}/b/c"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_file_request_parameter(use_temporary_input_directory):
    _image_1x1_px = np.array([[[255, 0, 0]]], dtype=np.uint8)
    image_path = os.path.join(use_temporary_input_directory, "test_image.png")
    image = Image.fromarray(_image_1x1_px)
    image.save(image_path)

    n = ImageRequestParameter()
    loaded_image, = n.execute(value=image_path)
    assert loaded_image.shape == (1, 1, 1, 3)
    from comfy.nodes.base_nodes import LoadImage

    load_image_node = LoadImage()
    load_image_node_rgb, _ = load_image_node.load_image(image=os.path.basename(image_path))

    assert loaded_image.shape == load_image_node_rgb.shape
    assert torch.allclose(loaded_image, load_image_node_rgb)


def test_file_request_to_http_url_no_exceptions():
    n = ImageRequestParameter()
    loaded_image, = n.execute(value="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/A_rainbow_at_sunset_after_rain_in_Gaziantep%2C_Turkey.IMG_2448.jpg/484px-A_rainbow_at_sunset_after_rain_in_Gaziantep%2C_Turkey.IMG_2448.jpg")
    _, height, width, channels = loaded_image.shape
    assert width == 484
    assert height == 480
    assert channels == 3
