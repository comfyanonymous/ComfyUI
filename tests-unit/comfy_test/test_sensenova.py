from types import SimpleNamespace

import torch

from comfy import model_detection
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


def test_sensenova_model_config_builds_pixel_space_outputs():
    model_config = model_detection.model_config_from_unet(_minimal_state_dict(), "")
    state_dict = {
        "language_model.lm_head.weight": torch.empty(1),
        "kept": torch.empty(1),
    }

    assert model_config.process_unet_state_dict(state_dict) == {
        "kept": state_dict["kept"]
    }
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


def test_sensenova_tokenizer_control_token_ids():
    tokenizer = SenseNovaTokenizer()
    backend = tokenizer.sensenova_u15.tokenizer

    assert len(backend) == 151936
    assert backend.convert_tokens_to_ids(
        ["<IMG_CONTEXT>", "<img>", "</img>", "<FAKE_PAD_253>"]
    ) == [151669, 151670, 151671, 151935]
    assert tokenizer.tokenize_with_weights("")["sensenova_u15"][0][-1][0] == 151670
