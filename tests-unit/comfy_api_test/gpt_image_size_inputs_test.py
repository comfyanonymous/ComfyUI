from comfy_api.latest._io import get_finalized_class_inputs
from comfy_api_nodes.nodes_openai import OpenAIGPTImageNodeV2


def _finalized_inputs(live_inputs):
    class_inputs, _, _ = get_finalized_class_inputs(
        OpenAIGPTImageNodeV2.INPUT_TYPES(), live_inputs
    )
    return class_inputs


def test_gpt_image_2_custom_size_inputs_are_optional():
    """A preset `size` must not require the custom width/height widgets to be present."""
    class_inputs = _finalized_inputs({"model": "gpt-image-2", "model.size": "1536x1024"})

    assert "model.custom_width" not in class_inputs["required"]
    assert "model.custom_height" not in class_inputs["required"]
    assert "model.custom_width" in class_inputs["optional"]
    assert "model.custom_height" in class_inputs["optional"]
    assert "model.size" in class_inputs["required"]


def test_gpt_image_legacy_models_have_no_custom_size_inputs():
    for model in ("gpt-image-1", "gpt-image-1.5"):
        class_inputs = _finalized_inputs({"model": model, "model.size": "1536x1024"})
        assert "model.custom_width" not in class_inputs["required"]
        assert "model.custom_width" not in class_inputs["optional"]
