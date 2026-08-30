from types import SimpleNamespace

import torch

from comfy import model_detection
from comfy.ldm.sensenova import model as sensenova_model
from comfy.ldm.sensenova.conditioning import (
    condition_input_ids,
    conditioned_input_length,
    thw_indexes,
)
from comfy.ldm.sensenova.sampling import (
    SenseNovaModelSampling,
    resolution_noise_scale,
    upstream_sigmas,
)
from comfy.text_encoders.sensenova import SenseNovaTokenizer
from comfy_extras.nodes_sensenova import SenseNovaReferenceImages


def _minimal_state_dict():
    return {
        "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight": torch.empty(
            1024, 3, 16, 16, device="meta"
        ),
        "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight": torch.empty(
            4096, 4096, device="meta"
        ),
    }


def test_sensenova_top_level_checkpoint_detection():
    state_dict = _minimal_state_dict()

    assert model_detection.unet_prefix_from_state_dict(state_dict) == ""
    assert model_detection.detect_unet_config(state_dict, "") == {
        "image_model": "sensenova_u15"
    }
    assert (
        type(model_detection.model_config_from_unet(state_dict, "")).__name__
        == "SenseNovaU15"
    )


def test_sensenova_detection_rejects_incompatible_dimensions():
    state_dict = _minimal_state_dict()
    state_dict["language_model.model.layers.0.self_attn.q_proj_mot_gen.weight"] = (
        torch.empty(2048, 2048, device="meta")
    )

    assert model_detection.detect_unet_config(state_dict, "") is None


def test_sensenova_model_config_builds_pixel_space_outputs():
    model_config = model_detection.model_config_from_unet(_minimal_state_dict(), "")
    state_dict = {
        "language_model.lm_head.weight": torch.empty(1),
        "kept": torch.empty(1),
    }

    processed = model_config.process_unet_state_dict(state_dict)
    assert set(processed) == {"kept"}
    assert torch.equal(processed["kept"], state_dict["kept"])
    assert "pixel_space_vae" in model_config.process_vae_state_dict({})
    assert "_sensenova_te_sentinel" in model_config.process_clip_state_dict({})


def test_sensenova_sampling_matches_upstream_schedule_and_resolution_scale():
    config = SimpleNamespace(sampling_settings={"shift": 3.0, "noise_scale": 1.0})
    sampling = SenseNovaModelSampling(config)

    expected = upstream_sigmas(50, 3.0)
    actual = sampling.sigma(torch.linspace(0.0, 1000.0, 51))
    assert torch.allclose(actual, expected)
    assert sampling.percent_to_sigma(0.0) == 1.0
    assert sampling.percent_to_sigma(1.0) == 0.0
    assert resolution_noise_scale(2048, 2048) == 8.0
    assert resolution_noise_scale(4096, 4096) == 16.0

    scaled_sampling = SenseNovaModelSampling(
        SimpleNamespace(sampling_settings={"shift": 3.0, "noise_scale": 0.5})
    )
    noise = torch.ones(1, 3, 256, 256)
    latent = torch.zeros_like(noise)
    scaled = scaled_sampling.noise_scaling(torch.ones(1), noise, latent)
    assert torch.allclose(scaled, torch.full_like(noise, 0.5))


def test_sensenova_reference_images_preserve_modes_when_chained():
    conditioning = [[torch.empty(1), {}]]
    first_image = torch.empty(1, 8, 8, 3)
    second_image = torch.empty(1, 8, 8, 3)

    first = SenseNovaReferenceImages.execute(
        positive=conditioning,
        negative=conditioning,
        images={"image_1": first_image},
    )
    second = SenseNovaReferenceImages.execute(
        positive=first[0],
        negative=first[1],
        images={"image_1": second_image},
    )

    assert second[0][0][1]["sensenova_reference_mode"] == "condition"
    assert second[1][0][1]["sensenova_reference_mode"] == "image_only"
    references = second[0][0][1]["sensenova_reference_images"]
    assert len(references) == 2
    assert references[0] is first_image
    assert references[1] is second_image


def test_sensenova_reference_tokens_and_indexes():
    input_ids = torch.tensor(
        [
            [
                151644,
                8948,
                198,
                1,
                151645,
                198,
                151644,
                872,
                198,
                2,
                151645,
                198,
                151644,
                77091,
                198,
                151670,
            ]
        ],
        dtype=torch.long,
    )
    grids = [(2, 3), (1, 2)]

    conditioned = condition_input_ids(input_ids, grids)
    indexes = thw_indexes(conditioned, grids)

    assert conditioned.shape[1] == conditioned_input_length(input_ids.shape[1], grids)
    assert torch.count_nonzero(conditioned == 151669) == 8
    assert indexes.shape == (1, 3, conditioned.shape[1])


def test_sensenova_prefix_mask_matches_attention_dtype(monkeypatch):
    query = torch.empty(1, 32, 3, 128, dtype=torch.bfloat16)
    key = torch.empty(1, 8, 3, 128, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    captured = {}

    def optimized_attention(query, key, value, heads, **kwargs):
        captured.update(query=query, key=key, value=value, heads=heads, kwargs=kwargs)
        return torch.empty(1, 3, 4096, dtype=torch.bfloat16)

    monkeypatch.setattr(sensenova_model, "optimized_attention", optimized_attention)
    attention = SimpleNamespace(
        _project=lambda hidden_states, indexes, generation: (query, key, value),
        o_proj=lambda output: output,
    )

    output, _, _ = sensenova_model.Attention.forward_prefix(
        attention,
        torch.empty(1, 3, 4096, dtype=torch.bfloat16),
        torch.empty(3, 1, 3),
        torch.zeros(1, 1, 3, 3, dtype=torch.float32),
        {},
    )

    assert output.shape == (1, 3, 4096)
    assert captured["kwargs"]["mask"].dtype == torch.bfloat16


def test_sensenova_tokenizer_control_token_ids():
    tokenizer = SenseNovaTokenizer()
    backend = tokenizer.sensenova_u15.tokenizer

    assert len(backend) == 151936
    assert backend.convert_tokens_to_ids(
        ["<IMG_CONTEXT>", "<img>", "</img>", "<FAKE_PAD_253>"]
    ) == [151669, 151670, 151671, 151935]
    assert "<|im_start|>" in backend.all_special_tokens
    assert "<|vision_pad|>" in backend.all_special_tokens
    assert tokenizer.tokenize_with_weights("")["sensenova_u15"][0][-1][0] == 151670
