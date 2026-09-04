import hashlib
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from comfy.cli_args import args

args.cpu = True

import comfy.ops
import comfy.supported_models
from comfy.text_encoders.llada_image import (
    LLaDA2Backbone,
    LLaDA2Config,
    LLaDAImageClipModel,
    LLaDAImageRawTokenizer,
)


def small_backbone():
    config = LLaDA2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=1,
        first_k_dense_replace=1,
        n_group=2,
        topk_group=1,
        pad_token_id=0,
        max_position_embeddings=64,
    )
    operations = comfy.ops.mixed_precision_ops(
        {}, torch.float32, full_precision_mm=True
    )
    model = LLaDA2Backbone(
        config, torch.float32, torch.device("cpu"), operations
    ).eval()
    torch.manual_seed(20)
    state_dict = {
        key: torch.randn_like(value) * 0.02 for key, value in model.state_dict().items()
    }
    for name, module in model.named_modules():
        prefix = f"{name}." if name else ""
        if hasattr(module, "_orig_shape"):
            state_dict[f"{prefix}weight"] = torch.randn(module._orig_shape) * 0.02
        elif (
            hasattr(module, "in_features")
            and hasattr(module, "out_features")
            and getattr(module, "weight", None) is None
        ):
            state_dict[f"{prefix}weight"] = (
                torch.randn(module.out_features, module.in_features) * 0.02
            )
    incompatible = model.load_state_dict(state_dict, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    return model


def test_backbone_bool_mask_blocks_padded_tokens_from_valid_outputs():
    model = small_backbone()
    first = torch.tensor([[1, 2, 3, 4, 5, 0, 0]])
    second = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    valid = torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    allowed = valid[:, None, None, :].expand(-1, 1, 7, -1).repeat(2, 1, 1, 1)
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 4, 4]]).repeat(2, 1)

    with torch.inference_mode():
        output = model.language_model(
            torch.cat((first, second)),
            attention_mask=allowed,
            position_ids=position_ids,
        )

    torch.testing.assert_close(output[0, :5], output[1, :5], atol=2e-5, rtol=2e-5)
    assert torch.isfinite(output).all()


def test_aio_clip_processing_adapts_official_expert_tensor_names():
    config = comfy.supported_models.LLaDAImage(
        {"image_model": "llada_image", "variant": "base"}
    )
    state_dict = {
        "text_encoders.llada2.model.language_model.layers.1.mlp.experts.gate_proj": torch.empty(
            8, 4, 16
        ),
        "text_encoders.llada2.model.language_model.layers.1.mlp.experts.up_proj": torch.empty(
            8, 4, 16
        ),
        "text_encoders.llada2.model.language_model.layers.1.mlp.experts.down_proj": torch.empty(
            8, 16, 4
        ),
    }

    processed = config.process_clip_state_dict(state_dict)

    assert set(processed) == {
        "llada2.model.language_model.layers.1.mlp.experts.gate_proj.weight",
        "llada2.model.language_model.layers.1.mlp.experts.up_proj.weight",
        "llada2.model.language_model.layers.1.mlp.experts.down_proj.weight",
    }


def test_aio_clip_component_prefixes_roundtrip_for_saving():
    config = comfy.supported_models.LLaDAImage(
        {"image_model": "llada_image", "variant": "base"}
    )
    checkpoint_state = {
        "text_encoders.llada2.model.language_model.word_embeddings.weight": torch.empty(
            64, 16
        ),
        "text_encoders.llada2.model.language_model.layers.1.mlp.experts.gate_proj": torch.empty(
            8, 4, 16
        ),
        "text_encoders.queryformer.meta_queries": torch.empty(5, 16),
        "text_encoders.text_projection.projector.weight": torch.empty(8, 16),
        "text_encoders.sigvq.prior_token_embedding.weight": torch.empty(8, 10),
        "text_encoders.tokenizer_json": torch.empty(128, dtype=torch.uint8),
    }

    processed = config.process_clip_state_dict(dict(checkpoint_state))

    assert set(processed) == {
        "llada2.model.language_model.word_embeddings.weight",
        "llada2.model.language_model.layers.1.mlp.experts.gate_proj.weight",
        "llada2.queryformer.meta_queries",
        "llada2.text_projection.projector.weight",
        "llada2.sigvq.prior_token_embedding.weight",
        "tokenizer_json",
    }

    saved = config.process_clip_state_dict_for_saving(processed)

    assert set(saved) == set(checkpoint_state)
    for key in checkpoint_state:
        assert saved[key] is checkpoint_state[key]


class TinyVQGenerator(LLaDAImageClipModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(
            mask_token_id=3,
            end_of_image_token_id=11,
            image_token_offset=4,
        )

    def forward_logits(self, input_ids, attention_mask, position_ids):
        logits = torch.full((*input_ids.shape, 12), -20.0, device=input_ids.device)
        logits[..., 4] = 20.0
        return logits


def test_block_diffusion_vq_generation_is_greedy_and_offset_normalized():
    model = TinyVQGenerator()
    input_ids = torch.tensor([[1, 2]])
    unconditional_ids = torch.tensor([2])

    token_ids = model.generate_vq_tokens(input_ids, unconditional_ids, 4)

    assert token_ids.shape == (1, 4)
    assert torch.equal(token_ids, torch.zeros_like(token_ids))


def test_official_tokenizer_golden_ids_when_reference_is_available():
    tokenizer_path = os.environ.get("LLADA_IMAGE_TOKENIZER_JSON")
    if not tokenizer_path or not os.path.isfile(tokenizer_path):
        pytest.skip(
            "set LLADA_IMAGE_TOKENIZER_JSON to run the pinned tokenizer parity fixture"
        )
    with open(tokenizer_path, "rb") as tokenizer_file:
        tokenizer_bytes = tokenizer_file.read()
    assert hashlib.sha256(tokenizer_bytes).hexdigest() == (
        "2197aeddaf09785316673451ca6fb86dcfcfdb108972a3145d106b8fa4c927e6"
    )
    tokenizer = LLaDAImageRawTokenizer(
        tokenizer_data={
            "tokenizer_json": torch.frombuffer(
                bytearray(tokenizer_bytes), dtype=torch.uint8
            )
        }
    )
    cases = {
        "": [
            157151,
            39,
            116171,
            157152,
            34161,
            289,
            3972,
            13,
            198,
            157151,
            8469,
            7342,
            5468,
            157152,
            198,
            157185,
        ],
        "A red fox sleeping beneath a glass tree.": [
            157151,
            39,
            116171,
            157152,
            34161,
            289,
            3972,
            25,
            355,
            3862,
            46998,
            20666,
            24772,
            259,
            7966,
            6919,
            13,
            198,
            157151,
            8469,
            7342,
            5468,
            157152,
            198,
            157185,
        ],
        "一只白猫坐在月亮上。": [
            157151,
            39,
            116171,
            157152,
            34161,
            289,
            3972,
            25,
            6097,
            1030,
            1474,
            8187,
            8986,
            33413,
            520,
            311,
            198,
            157151,
            8469,
            7342,
            5468,
            157152,
            198,
            157185,
        ],
        "literal <IMAGE1> and <|image|> tokens": [
            157151,
            39,
            116171,
            157152,
            34161,
            289,
            3972,
            25,
            31567,
            220,
            157185,
            301,
            220,
            156901,
            21255,
            198,
            157151,
            8469,
            7342,
            5468,
            157152,
            198,
            157185,
        ],
    }
    for prompt, expected in cases.items():
        actual = [pair[0] for pair in tokenizer.tokenize_with_weights(prompt)[0]]
        assert actual == expected

    long_ids = [
        pair[0]
        for pair in tokenizer.tokenize_with_weights("绘画 " + "blue mountain " * 1500)[
            0
        ]
    ]
    assert len(long_ids) == 2048
    assert hashlib.sha256(",".join(map(str, long_ids)).encode()).hexdigest() == (
        "521e55c839c1429a3e0525bea573025ca76b8a9cb5406f78b5ae896ae1c579bc"
    )


def test_unsupported_conditioning_config_fails_instead_of_approximating():
    with pytest.raises(ValueError, match="score_function"):
        LLaDAImageClipModel(
            dtype=torch.float32,
            model_options={"llada2_config": {"score_function": "softmax"}},
        )
