import importlib
import json
import sys
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock
from xml.etree import ElementTree

import comfy_extras
import numpy as np
import pytest
import torch
from PIL import Image as PILImage


MISSING = object()

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

original_nodes = sys.modules.get("nodes", MISSING)
original_server = sys.modules.get("server", MISSING)
original_nodes_images = sys.modules.get("comfy_extras.nodes_images", MISSING)
original_nodes_images_attr = getattr(comfy_extras, "nodes_images", MISSING)
sys.modules["nodes"] = mock_nodes
sys.modules["server"] = mock_server
sys.modules.pop("comfy_extras.nodes_images", None)
if original_nodes_images_attr is not MISSING:
    delattr(comfy_extras, "nodes_images")
try:
    nodes_images = importlib.import_module("comfy_extras.nodes_images")
finally:
    if original_nodes is MISSING:
        sys.modules.pop("nodes", None)
    else:
        sys.modules["nodes"] = original_nodes
    if original_server is MISSING:
        sys.modules.pop("server", None)
    else:
        sys.modules["server"] = original_server
    if original_nodes_images is MISSING:
        sys.modules.pop("comfy_extras.nodes_images", None)
    else:
        sys.modules["comfy_extras.nodes_images"] = original_nodes_images
    if original_nodes_images_attr is MISSING:
        if hasattr(comfy_extras, "nodes_images"):
            delattr(comfy_extras, "nodes_images")
    else:
        setattr(comfy_extras, "nodes_images", original_nodes_images_attr)


COMFYUI_XMP_NAMESPACE = "https://github.com/Comfy-Org/ComfyUI"
RDF_NAMESPACE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def test_save_image_advanced_schema_exposes_avif_quality():
    schema = nodes_images.SaveImageAdvanced.define_schema()
    format_input = next(
        input_item for input_item in schema.inputs if input_item.id == "format"
    )
    format_options = {option.key: option for option in format_input.options}

    assert list(format_options) == ["png", "exr", "avif"]

    avif_inputs = format_options["avif"].inputs
    assert [input_item.id for input_item in avif_inputs] == ["quality"]
    assert avif_inputs[0].default == 75
    assert avif_inputs[0].min == 0
    assert avif_inputs[0].max == 100

    serialized_options = {
        option["key"]: option["inputs"] for option in format_input.as_dict()["options"]
    }
    assert serialized_options["avif"]["required"]["quality"][:1] == ("INT",)


def test_build_avif_xmp_escapes_and_preserves_comfyui_metadata():
    prompt = {"1": {"class_type": "KSampler", "inputs": {"text": "<tag> & value"}}}
    workflow = {"nodes": [{"id": 1, "title": 'Save "AVIF"'}], "version": 1}
    extra_pnginfo = {"workflow": workflow, "custom metadata": {"value": "<one> & two"}}

    xmp = nodes_images.build_avif_xmp(prompt, extra_pnginfo)

    assert xmp is not None
    root = ElementTree.fromstring(xmp)
    description = root.find(f".//{{{RDF_NAMESPACE}}}Description")
    workflow_element = root.find(f".//{{{COMFYUI_XMP_NAMESPACE}}}workflow")
    extra_pnginfo_element = root.find(
        f".//{{{COMFYUI_XMP_NAMESPACE}}}extra_pnginfo"
    )

    assert description is not None
    assert workflow_element is not None
    assert extra_pnginfo_element is not None
    assert description.attrib[f"{{{COMFYUI_XMP_NAMESPACE}}}prompt"] == json.dumps(
        prompt
    )
    assert workflow_element.text == json.dumps(workflow)
    assert json.loads(extra_pnginfo_element.text) == extra_pnginfo


def test_build_avif_xmp_preserves_extra_pnginfo_without_workflow():
    extra_pnginfo = {"custom": {"value": 1}}

    xmp = nodes_images.build_avif_xmp(None, extra_pnginfo)

    assert xmp is not None
    root = ElementTree.fromstring(xmp)
    extra_pnginfo_element = root.find(
        f".//{{{COMFYUI_XMP_NAMESPACE}}}extra_pnginfo"
    )
    assert extra_pnginfo_element is not None
    assert json.loads(extra_pnginfo_element.text) == extra_pnginfo


def test_encode_avif_image_passes_fixed_encoding_options(monkeypatch):
    saved_options = {}
    pillow_initialized = []

    def fake_save(image, destination, **options):
        saved_options.update(options)
        destination.write(b"encoded-avif")

    monkeypatch.setattr(
        nodes_images.PILImage, "init", lambda: pillow_initialized.append(True)
    )
    monkeypatch.setitem(nodes_images.PILImage.SAVE, "AVIF", object())
    monkeypatch.setattr(
        nodes_images.features, "check", lambda feature: feature == "avif"
    )
    monkeypatch.setattr(PILImage.Image, "save", fake_save)

    image = torch.zeros((8, 12, 3), dtype=torch.float32)
    encoded = nodes_images.encode_avif_image(image, quality=63, xmp=b"xmp-packet")

    assert encoded == b"encoded-avif"
    assert pillow_initialized == [True]
    assert saved_options == {
        "format": "AVIF",
        "quality": 63,
        "subsampling": "4:2:0",
        "range": "full",
        "xmp": b"xmp-packet",
    }


def test_encode_avif_image_round_trips_rgba_and_xmp():
    prompt = {"1": {"class_type": "KSampler"}}
    workflow = {"nodes": [], "version": 1}
    xmp = nodes_images.build_avif_xmp(prompt, {"workflow": workflow})
    image = torch.rand((12, 16, 4), dtype=torch.float32)

    encoded = nodes_images.encode_avif_image(image, quality=85, xmp=xmp)

    with PILImage.open(BytesIO(encoded)) as saved_image:
        saved_image.load()
        assert saved_image.format == "AVIF"
        assert saved_image.size == (16, 12)
        assert saved_image.mode == "RGBA"
        assert saved_image.info["xmp"] == xmp


@pytest.mark.parametrize("image_shape", [(12, 16), (12, 16, 1)])
def test_encode_avif_image_round_trips_grayscale(image_shape):
    image = torch.rand(image_shape, dtype=torch.float32)

    encoded = nodes_images.encode_avif_image(image, quality=85, xmp=None)

    with PILImage.open(BytesIO(encoded)) as saved_image:
        saved_image.load()
        assert saved_image.format == "AVIF"
        assert saved_image.size == (16, 12)
        assert saved_image.mode == "RGB"


def test_encode_avif_image_reports_missing_pillow_support(monkeypatch):
    monkeypatch.setattr(nodes_images.features, "check", lambda feature: False)

    with pytest.raises(RuntimeError, match="Pillow build with AVIF support"):
        nodes_images.encode_avif_image(torch.zeros((8, 8, 3)), quality=75, xmp=None)


@pytest.mark.parametrize("disable_metadata", [False, True])
def test_save_image_advanced_writes_avif_batch_and_respects_metadata_setting(
    monkeypatch, tmp_path, disable_metadata
):
    prompt = {"1": {"class_type": "KSampler"}}
    workflow = {"nodes": [], "version": 1}
    extra_pnginfo = {"workflow": workflow, "custom": {"value": 1}}
    images = torch.from_numpy(
        np.random.default_rng(7).random((2, 12, 16, 3), dtype=np.float32)
    )

    monkeypatch.setattr(nodes_images.args, "disable_metadata", disable_metadata)
    monkeypatch.setattr(
        nodes_images.folder_paths, "get_output_directory", lambda: str(tmp_path)
    )
    monkeypatch.setattr(
        nodes_images.folder_paths,
        "get_save_image_path",
        lambda *args: (str(tmp_path), "ComfyUI_%batch_num%", 7, "", "ComfyUI"),
    )
    monkeypatch.setattr(
        nodes_images.SaveImageAdvanced,
        "hidden",
        SimpleNamespace(prompt=prompt, extra_pnginfo=extra_pnginfo),
    )

    output = nodes_images.SaveImageAdvanced.execute(
        images,
        filename_prefix="ComfyUI",
        format={"format": "avif", "quality": 82},
    )

    expected_filenames = ["ComfyUI_0_00007.avif", "ComfyUI_1_00008.avif"]
    assert output[0] is images
    assert [result["filename"] for result in output.ui["images"]] == expected_filenames

    for filename in expected_filenames:
        with PILImage.open(tmp_path / filename) as saved_image:
            saved_image.load()
            if disable_metadata:
                assert "xmp" not in saved_image.info
            else:
                assert saved_image.info["xmp"] == nodes_images.build_avif_xmp(
                    prompt, extra_pnginfo
                )
