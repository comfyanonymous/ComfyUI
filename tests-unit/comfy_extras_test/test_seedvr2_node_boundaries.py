import ast
import inspect
import textwrap

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras import nodes_seedvr  # noqa: E402


def _schema_ids(items):
    return [item.id for item in items]


def test_resize_schemas_are_preprocess_only():
    simple = nodes_seedvr.SeedVR2Resize.define_schema()
    advanced = nodes_seedvr.SeedVR2ResizeAdvanced.define_schema()

    assert _schema_ids(simple.inputs) == ["images", "multiplier"]
    assert _schema_ids(simple.outputs) == ["input_pixels", "original_image", "upscaled_shorter_edge"]
    assert simple.outputs[0].get_io_type() == "IMAGE"

    assert _schema_ids(advanced.inputs) == ["images", "shorter_edge"]
    assert _schema_ids(advanced.outputs) == ["input_pixels", "original_image", "upscaled_shorter_edge"]
    assert advanced.outputs[0].get_io_type() == "IMAGE"


def test_resize_nodes_do_not_call_encode_decode_or_color_transfer():
    source = "\n".join(
        [
            inspect.getsource(nodes_seedvr.SeedVR2Resize.execute),
            inspect.getsource(nodes_seedvr.SeedVR2ResizeAdvanced.execute),
        ]
    )
    tree = ast.parse(textwrap.dedent(source))
    forbidden_names = {
        "encode",
        "encode_tiled",
        "decode",
        "decode_tiled",
        "tiled_vae",
        "lab_color_transfer",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            else:
                continue
            assert name not in forbidden_names
