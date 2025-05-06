from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field

from comfy.comfy_types.node_typing import IO
from comfy_api_nodes.mapper_utils import model_field_to_node_input


def test_model_field_to_float_input():
    """Tests mapping a float field with constraints."""

    class ModelWithFloatField(BaseModel):
        cfg_scale: Optional[float] = Field(
            default=0.5,
            description="Flexibility in video generation",
            ge=0.0,
            le=1.0,
            multiple_of=0.001,
        )

    expected_output = (
        IO.FLOAT,
        {
            "default": 0.5,
            "tooltip": "Flexibility in video generation",
            "min": 0.0,
            "max": 1.0,
            "step": 0.001,
        },
    )

    actual_output = model_field_to_node_input(
        IO.FLOAT, ModelWithFloatField, "cfg_scale"
    )

    assert actual_output[0] == expected_output[0]
    assert actual_output[1] == expected_output[1]


def test_model_field_to_float_input_no_constraints():
    """Tests mapping a float field with no constraints."""

    class ModelWithFloatField(BaseModel):
        cfg_scale: Optional[float] = Field(default=0.5)

    expected_output = (
        IO.FLOAT,
        {
            "default": 0.5,
        },
    )

    actual_output = model_field_to_node_input(
        IO.FLOAT, ModelWithFloatField, "cfg_scale"
    )

    assert actual_output[0] == expected_output[0]
    assert actual_output[1] == expected_output[1]


def test_model_field_to_int_input():
    """Tests mapping an int field with constraints."""

    class ModelWithIntField(BaseModel):
        num_frames: Optional[int] = Field(
            default=10,
            description="Number of frames to generate",
            ge=1,
            le=100,
            multiple_of=1,
        )

    expected_output = (
        IO.INT,
        {
            "default": 10,
            "tooltip": "Number of frames to generate",
            "min": 1,
            "max": 100,
            "step": 1,
        },
    )

    actual_output = model_field_to_node_input(IO.INT, ModelWithIntField, "num_frames")

    assert actual_output[0] == expected_output[0]
    assert actual_output[1] == expected_output[1]


def test_model_field_to_string_input():
    """Tests mapping a string field."""

    class ModelWithStringField(BaseModel):
        prompt: Optional[str] = Field(
            default="A beautiful sunset over a calm ocean",
            description="A prompt for the video generation",
        )

    expected_output = (
        IO.STRING,
        {
            "default": "A beautiful sunset over a calm ocean",
            "tooltip": "A prompt for the video generation",
        },
    )

    actual_output = model_field_to_node_input(IO.STRING, ModelWithStringField, "prompt")

    assert actual_output[0] == expected_output[0]
    assert actual_output[1] == expected_output[1]


def test_model_field_to_string_input_multiline():
    """Tests mapping a string field."""

    class ModelWithStringField(BaseModel):
        prompt: Optional[str] = Field(
            default="A beautiful sunset over a calm ocean",
            description="A prompt for the video generation",
        )

    expected_output = (
        IO.STRING,
        {
            "default": "A beautiful sunset over a calm ocean",
            "tooltip": "A prompt for the video generation",
            "multiline": True,
        },
    )

    actual_output = model_field_to_node_input(
        IO.STRING, ModelWithStringField, "prompt", multiline=True
    )

    assert actual_output[0] == expected_output[0]
    assert actual_output[1] == expected_output[1]


def test_model_field_to_combo_input():
    """Tests mapping a combo field."""

    class MockEnum(str, Enum):
        option_1 = "option 1"
        option_2 = "option 2"
        option_3 = "option 3"

    class ModelWithComboField(BaseModel):
        model_name: Optional[MockEnum] = Field("option 1", description="Model Name")

    expected_output = (
        IO.COMBO,
        {
            "options": ["option 1", "option 2", "option 3"],
            "default": "option 1",
            "tooltip": "Model Name",
        },
    )

    actual_output = model_field_to_node_input(
        IO.COMBO, ModelWithComboField, "model_name", enum_type=MockEnum
    )

    assert actual_output[0] == expected_output[0]
    assert actual_output[1] == expected_output[1]


def test_model_field_to_combo_input_no_options():
    """Tests mapping a combo field with no options."""

    class ModelWithComboField(BaseModel):
        model_name: Optional[str] = Field(description="Model Name")

    expected_output = (
        IO.COMBO,
        {
            "tooltip": "Model Name",
        },
    )

    actual_output = model_field_to_node_input(
        IO.COMBO, ModelWithComboField, "model_name"
    )

    assert actual_output[0] == expected_output[0]
    assert actual_output[1] == expected_output[1]


def test_model_field_to_image_input():
    """Tests mapping an image field."""

    class ModelWithImageField(BaseModel):
        image: Optional[str] = Field(
            default=None,
            description="An image for the video generation",
        )

    expected_output = (
        IO.IMAGE,
        {
            "default": None,
            "tooltip": "An image for the video generation",
        },
    )

    actual_output = model_field_to_node_input(IO.IMAGE, ModelWithImageField, "image")

    assert actual_output[0] == expected_output[0]
    assert actual_output[1] == expected_output[1]


def test_model_field_to_node_input_no_description():
    """Tests mapping a field with no description."""

    class ModelWithNoDescriptionField(BaseModel):
        field: Optional[str] = Field(default="default value")

    expected_output = (
        IO.STRING,
        {
            "default": "default value",
        },
    )

    actual_output = model_field_to_node_input(
        IO.STRING, ModelWithNoDescriptionField, "field"
    )

    assert actual_output[0] == expected_output[0]
    assert actual_output[1] == expected_output[1]


def test_model_field_to_node_input_no_default():
    """Tests mapping a field with no default."""

    class ModelWithNoDefaultField(BaseModel):
        field: Optional[str] = Field(description="A field with no default")

    expected_output = (
        IO.STRING,
        {
            "tooltip": "A field with no default",
        },
    )

    actual_output = model_field_to_node_input(
        IO.STRING, ModelWithNoDefaultField, "field"
    )

    assert actual_output[0] == expected_output[0]
    assert actual_output[1] == expected_output[1]


def test_model_field_to_node_input_no_metadata():
    """Tests mapping a field with no metadata or properties defined on the schema."""

    class ModelWithNoMetadataField(BaseModel):
        field: Optional[str] = Field()

    expected_output = (
        IO.STRING,
        {},
    )

    actual_output = model_field_to_node_input(
        IO.STRING, ModelWithNoMetadataField, "field"
    )

    assert actual_output[0] == expected_output[0]
    assert actual_output[1] == expected_output[1]


def test_model_field_to_node_input_default_is_none():
    """
    Tests mapping a field with a default of `None`.
    I.e., the default field should be included as the schema explicitly sets it to `None`.
    """

    class ModelWithNoneDefaultField(BaseModel):
        field: Optional[str] = Field(
            default=None, description="A field with a default of None"
        )

    expected_output = (
        IO.STRING,
        {
            "default": None,
            "tooltip": "A field with a default of None",
        },
    )

    actual_output = model_field_to_node_input(
        IO.STRING, ModelWithNoneDefaultField, "field"
    )

    assert actual_output[0] == expected_output[0]
    assert actual_output[1] == expected_output[1]
