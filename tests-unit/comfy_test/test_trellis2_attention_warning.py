import logging
from types import SimpleNamespace

from comfy.cli_args import args

args.cpu = True

from comfy import model_base, model_management


def _model_config():
    return SimpleNamespace(
        unet_config={"disable_unet_model_creation": True},
        latent_format=None,
        manual_cast_dtype=None,
        custom_operations=None,
        optimizations={},
        memory_usage_factor=1.0,
        sampling_settings={},
    )


def test_trellis2_warns_when_ck_attention_is_enabled(monkeypatch, caplog):
    monkeypatch.setattr(model_management, "comfy_kitchen_attention_enabled", lambda: True)

    with caplog.at_level(logging.WARNING):
        model_base.Trellis2(_model_config())

    assert any("use-ck-attention" in record.message for record in caplog.records)


def test_trellis2_does_not_warn_when_ck_attention_is_disabled(monkeypatch, caplog):
    monkeypatch.setattr(model_management, "comfy_kitchen_attention_enabled", lambda: False)

    with caplog.at_level(logging.WARNING):
        model_base.Trellis2(_model_config())

    assert not any("use-ck-attention" in record.message for record in caplog.records)
