import importlib.util
from pathlib import Path

import torch
import pytest

from comfy.cli_args import args

args.cpu = True

import comfy.model_management
import comfy.ops
from comfy.ldm.llada_image.model import LLaDAImage


UPSTREAM_MODEL = (
    Path(__file__).resolve().parents[3]
    / "LLaDA-Image"
    / "src"
    / "models"
    / "transformer_llada_image.py"
)
pytestmark = pytest.mark.skipif(
    not UPSTREAM_MODEL.is_file(),
    reason="optional parity test requires a sibling LLaDA-Image checkout",
)


def load_reference_class():
    spec = importlib.util.spec_from_file_location(
        "llada_image_reference", UPSTREAM_MODEL
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LLaDAImageTransformer2DModel


def make_models():
    config = {
        "all_patch_size": (1,),
        "all_f_patch_size": (1,),
        "in_channels": 4,
        "dim": 32,
        "n_layers": 2,
        "n_refiner_layers": 1,
        "n_heads": 2,
        "norm_eps": 1e-5,
        "qk_norm": True,
        "cap_feat_dim": 8,
        "semantic_feat_dim": 10,
        "rope_theta": 256.0,
        "t_scale": 1000.0,
        "axes_dims": (4, 6, 6),
    }
    torch.manual_seed(1)
    reference = load_reference_class()(**config, axes_lens=(512, 32, 32)).eval()
    model = LLaDAImage(**config, operations=comfy.ops.disable_weight_init).eval()
    incompatible = model.load_state_dict(reference.state_dict(), strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    return reference, model


def pad_features(features):
    length = max(len(value) for value in features)
    output = features[0].new_zeros((len(features), length, features[0].shape[-1]))
    mask = torch.zeros((len(features), length), dtype=torch.bool)
    for index, value in enumerate(features):
        output[index, : len(value)] = value
        mask[index, : len(value)] = True
    return output, mask


def test_text_to_image_matches_reference():
    reference, model = make_models()
    x = torch.randn(2, 4, 4, 4)
    timestep = torch.tensor([0.8, 0.35])
    captions = [torch.randn(3, 8), torch.randn(5, 8)]
    context, attention_mask = pad_features(captions)

    expected = reference(
        x=[value.unsqueeze(1) for value in x], t=timestep, cap_feats=captions
    ).sample
    expected = -torch.stack([value.squeeze(1) for value in expected])
    actual = model(x, timestep, context=context, attention_mask=attention_mask)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_editing_matches_reference():
    reference, model = make_models()
    x = torch.randn(2, 4, 4, 4)
    source = torch.randn(2, 4, 4, 4)
    timestep = torch.tensor([0.7, 0.2])
    captions = [torch.randn(3, 8), torch.randn(5, 8)]
    semantics = [torch.randn(2, 10), torch.randn(4, 10)]
    context, attention_mask = pad_features(captions)
    semantic_features, semantic_mask = pad_features(semantics)

    expected = reference(
        x=[value.unsqueeze(1) for value in x],
        t=timestep,
        cap_feats=captions,
        glm_cap_feats=semantics,
        source_latents=[value.unsqueeze(1) for value in source],
    ).sample
    expected = -torch.stack([value.squeeze(1) for value in expected])
    actual = model(
        x,
        timestep,
        context=context,
        attention_mask=attention_mask,
        semantic_features=semantic_features,
        semantic_mask=semantic_mask,
        source_latents=source,
    )

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
