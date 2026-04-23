import pytest

from comfy_api_nodes.nodes_openai import (
    OpenAIGPTImage1,
    OpenAIGPTImage2,
    _GPT_IMAGE_2_SIZES,
    _resolve_gpt_image_2_size,
    calculate_tokens_price_image_1,
    calculate_tokens_price_image_1_5,
    calculate_tokens_price_image_2,
)
from comfy_api_nodes.apis.openai import OpenAIImageGenerationResponse, Usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(input_tokens: int, output_tokens: int) -> OpenAIImageGenerationResponse:
    return OpenAIImageGenerationResponse(
        data=[],
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


# ---------------------------------------------------------------------------
# Price extractor tests
# ---------------------------------------------------------------------------

def test_price_image_1_formula():
    response = _make_response(input_tokens=1_000_000, output_tokens=1_000_000)
    assert calculate_tokens_price_image_1(response) == pytest.approx(50.0)


def test_price_image_1_5_formula():
    response = _make_response(input_tokens=1_000_000, output_tokens=1_000_000)
    assert calculate_tokens_price_image_1_5(response) == pytest.approx(40.0)


def test_price_image_2_formula():
    response = _make_response(input_tokens=1_000_000, output_tokens=1_000_000)
    assert calculate_tokens_price_image_2(response) == pytest.approx(38.0)


def test_price_image_2_cheaper_than_1():
    response = _make_response(input_tokens=500, output_tokens=196)
    assert calculate_tokens_price_image_2(response) < calculate_tokens_price_image_1(response)


def test_price_image_2_cheaper_output_than_1_5():
    # gpt-image-2 output rate ($30/1M) is lower than gpt-image-1.5 ($32/1M)
    response = _make_response(input_tokens=0, output_tokens=1_000_000)
    assert calculate_tokens_price_image_2(response) < calculate_tokens_price_image_1_5(response)


# ---------------------------------------------------------------------------
# _resolve_gpt_image_2_size tests
# ---------------------------------------------------------------------------

def test_resolve_preset_passthrough_when_custom_zero():
    # 0/0 means "use size preset"
    assert _resolve_gpt_image_2_size("1024x1024", 0, 0) == "1024x1024"
    assert _resolve_gpt_image_2_size("auto", 0, 0) == "auto"
    assert _resolve_gpt_image_2_size("3840x2160", 0, 0) == "3840x2160"


def test_resolve_preset_passthrough_when_only_one_dim_set():
    # only one dimension set → still use preset
    assert _resolve_gpt_image_2_size("auto", 1024, 0) == "auto"
    assert _resolve_gpt_image_2_size("auto", 0, 1024) == "auto"


def test_resolve_custom_overrides_preset():
    assert _resolve_gpt_image_2_size("auto", 1024, 1024) == "1024x1024"
    assert _resolve_gpt_image_2_size("1024x1024", 2048, 1152) == "2048x1152"
    assert _resolve_gpt_image_2_size("auto", 3840, 2160) == "3840x2160"


def test_resolve_custom_rejects_edge_too_large():
    with pytest.raises(ValueError, match="3840"):
        _resolve_gpt_image_2_size("auto", 4096, 1024)


def test_resolve_custom_rejects_non_multiple_of_16():
    with pytest.raises(ValueError, match="multiple of 16"):
        _resolve_gpt_image_2_size("auto", 1025, 1024)


def test_resolve_custom_rejects_bad_ratio():
    with pytest.raises(ValueError, match="ratio"):
        _resolve_gpt_image_2_size("auto", 3840, 1024)  # 3.75:1 > 3:1


def test_resolve_custom_rejects_too_few_pixels():
    with pytest.raises(ValueError, match="Total pixels"):
        _resolve_gpt_image_2_size("auto", 16, 16)


def test_resolve_custom_rejects_too_many_pixels():
    # 3840x2176 exceeds 8,294,400
    with pytest.raises(ValueError, match="Total pixels"):
        _resolve_gpt_image_2_size("auto", 3840, 2176)


# ---------------------------------------------------------------------------
# OpenAIGPTImage1 schema tests
# ---------------------------------------------------------------------------

class TestOpenAIGPTImage1Schema:
    def setup_method(self):
        self.schema = OpenAIGPTImage1.define_schema()

    def test_node_id(self):
        assert self.schema.node_id == "OpenAIGPTImage1"

    def test_display_name(self):
        assert self.schema.display_name == "OpenAI GPT Image 1 & 1.5"

    def test_model_options_exclude_gpt_image_2(self):
        model_input = next(i for i in self.schema.inputs if i.name == "model")
        assert "gpt-image-2" not in model_input.options

    def test_model_options_include_legacy_models(self):
        model_input = next(i for i in self.schema.inputs if i.name == "model")
        assert "gpt-image-1" in model_input.options
        assert "gpt-image-1.5" in model_input.options

    def test_has_background_with_transparent(self):
        bg_input = next(i for i in self.schema.inputs if i.name == "background")
        assert "transparent" in bg_input.options


# ---------------------------------------------------------------------------
# OpenAIGPTImage2 schema tests
# ---------------------------------------------------------------------------

class TestOpenAIGPTImage2Schema:
    def setup_method(self):
        self.schema = OpenAIGPTImage2.define_schema()

    def test_node_id(self):
        assert self.schema.node_id == "OpenAIGPTImage2"

    def test_display_name(self):
        assert self.schema.display_name == "OpenAI GPT Image 2"

    def test_category(self):
        assert "OpenAI" in self.schema.category

    def test_no_transparent_background(self):
        bg_input = next(i for i in self.schema.inputs if i.name == "background")
        assert "transparent" not in bg_input.options

    def test_background_options(self):
        bg_input = next(i for i in self.schema.inputs if i.name == "background")
        assert set(bg_input.options) == {"auto", "opaque"}

    def test_quality_options(self):
        quality_input = next(i for i in self.schema.inputs if i.name == "quality")
        assert set(quality_input.options) == {"auto", "low", "medium", "high"}

    def test_quality_default_is_auto(self):
        quality_input = next(i for i in self.schema.inputs if i.name == "quality")
        assert quality_input.default == "auto"

    def test_all_popular_sizes_present(self):
        size_input = next(i for i in self.schema.inputs if i.name == "size")
        for size in ["1024x1024", "1536x1024", "1024x1536", "2048x2048", "2048x1152", "3840x2160", "2160x3840"]:
            assert size in size_input.options, f"Missing size: {size}"

    def test_no_custom_size_option(self):
        size_input = next(i for i in self.schema.inputs if i.name == "size")
        assert "custom" not in size_input.options

    def test_size_default_is_auto(self):
        size_input = next(i for i in self.schema.inputs if i.name == "size")
        assert size_input.default == "auto"

    def test_custom_width_and_height_inputs_exist(self):
        input_names = [i.name for i in self.schema.inputs]
        assert "custom_width" in input_names
        assert "custom_height" in input_names

    def test_custom_width_height_default_zero(self):
        width_input = next(i for i in self.schema.inputs if i.name == "custom_width")
        height_input = next(i for i in self.schema.inputs if i.name == "custom_height")
        assert width_input.default == 0
        assert height_input.default == 0

    def test_custom_width_height_step_is_16(self):
        width_input = next(i for i in self.schema.inputs if i.name == "custom_width")
        height_input = next(i for i in self.schema.inputs if i.name == "custom_height")
        assert width_input.step == 16
        assert height_input.step == 16

    def test_custom_width_height_max_is_3840(self):
        width_input = next(i for i in self.schema.inputs if i.name == "custom_width")
        height_input = next(i for i in self.schema.inputs if i.name == "custom_height")
        assert width_input.max == 3840
        assert height_input.max == 3840

    def test_uses_num_images_not_n(self):
        input_names = [i.name for i in self.schema.inputs]
        assert "num_images" in input_names
        assert "n" not in input_names

    def test_model_input_shows_gpt_image_2(self):
        model_input = next(i for i in self.schema.inputs if i.name == "model")
        assert model_input.options == ["gpt-image-2"]
        assert model_input.default == "gpt-image-2"

    def test_has_image_and_mask_inputs(self):
        input_names = [i.name for i in self.schema.inputs]
        assert "image" in input_names
        assert "mask" in input_names

    def test_is_api_node(self):
        assert self.schema.is_api_node is True

    def test_sizes_match_constant(self):
        size_input = next(i for i in self.schema.inputs if i.name == "size")
        assert size_input.options == _GPT_IMAGE_2_SIZES


# ---------------------------------------------------------------------------
# OpenAIGPTImage2 execute validation tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_raises_on_empty_prompt():
    with pytest.raises(Exception):
        await OpenAIGPTImage2.execute(prompt="   ")


@pytest.mark.asyncio
async def test_execute_raises_mask_without_image():
    import torch
    mask = torch.ones(1, 64, 64)
    with pytest.raises(ValueError, match="mask without an input image"):
        await OpenAIGPTImage2.execute(prompt="test", mask=mask)


@pytest.mark.asyncio
async def test_execute_raises_invalid_custom_size():
    with pytest.raises(ValueError):
        await OpenAIGPTImage2.execute(prompt="test", custom_width=4096, custom_height=1024)
