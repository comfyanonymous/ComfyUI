import importlib.util
from pathlib import Path

import torch
import pytest

from comfy.cli_args import args

args.cpu = True

import comfy.ops
from comfy.ldm.llada_image.conditioning import QueryFormer, SigVQ, TextProjection


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


def load_reference_module():
    spec = importlib.util.spec_from_file_location(
        "llada_image_conditioning_reference", UPSTREAM_MODEL
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_queryformer_matches_reference():
    reference_class = load_reference_module().LLaDAImageQueryFormerModel
    config = {
        "num_queries": 5,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 24,
        "dropout": 0.0,
        "norm_eps": 1e-6,
    }
    torch.manual_seed(2)
    reference = reference_class(**config).eval()
    model = QueryFormer(
        **{key: value for key, value in config.items() if key != "dropout"},
        operations=comfy.ops.disable_weight_init,
    ).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    inputs = torch.randn(2, 7, 16)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool
    )

    expected = reference(inputs, mask).query_embeds
    actual = model(inputs, mask)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_text_projection_matches_reference():
    reference_class = load_reference_module().LLaDAImageTextProjectionModel
    config = {
        "hidden_size": 16,
        "intermediate_size": 28,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "projection_dim": 20,
        "attention_dropout": 0.0,
        "norm_eps": 1e-6,
    }
    torch.manual_seed(3)
    reference = reference_class(**config).eval()
    model = TextProjection(
        **{key: value for key, value in config.items() if key != "attention_dropout"},
        operations=comfy.ops.disable_weight_init,
    ).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    hidden_states = torch.randn(2, 9, 16)

    expected = reference(hidden_states).hidden_states
    actual = model(hidden_states)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_sigvq_image_and_token_paths_match_reference():
    reference_class = load_reference_module().LLaDAImageSigVQModel
    config = {
        "image_size": 16,
        "patch_size": 4,
        "in_channels": 3,
        "hidden_size": 16,
        "intermediate_size": 28,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "attention_bias": True,
        "attention_dropout": 0.0,
        "norm_eps": 1e-6,
        "codebook_size": 8,
        "codebook_embed_dim": 4,
        "semantic_embed_dim": 10,
    }
    torch.manual_seed(4)
    reference = reference_class(**config).eval()
    model = SigVQ(
        **{key: value for key, value in config.items() if key != "attention_dropout"},
        operations=comfy.ops.disable_weight_init,
    ).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    pixels = torch.randn(2, 3, 8, 12)

    expected_image = reference(pixel_values=pixels)
    actual_semantic, actual_tokens = model(pixel_values=pixels)
    torch.testing.assert_close(
        actual_semantic, expected_image.semantic_features, atol=2e-5, rtol=2e-5
    )
    assert torch.equal(actual_tokens, expected_image.token_ids)

    token_ids = torch.tensor([[0, 2, 4], [1, 3, 5]])
    expected_tokens = reference(token_ids=token_ids)
    actual_semantic, actual_tokens = model(token_ids=token_ids)
    torch.testing.assert_close(
        actual_semantic, expected_tokens.semantic_features, atol=2e-5, rtol=2e-5
    )
    assert torch.equal(actual_tokens, token_ids)
