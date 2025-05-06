from enum import Enum

from pydantic.fields import FieldInfo
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from comfy.comfy_types.node_typing import IO, InputTypeOptions

NodeInput = tuple[IO, InputTypeOptions]


def _create_base_config(field_info: FieldInfo) -> InputTypeOptions:
    config = {}
    if hasattr(field_info, "default") and field_info.default is not PydanticUndefined:
        config["default"] = field_info.default
    if hasattr(field_info, "description") and field_info.description is not None:
        config["tooltip"] = field_info.description
    return config


def _get_number_constraints_config(field_info: FieldInfo) -> dict:
    config = {}
    if hasattr(field_info, "metadata"):
        metadata = field_info.metadata
        for constraint in metadata:
            if hasattr(constraint, "ge"):
                config["min"] = constraint.ge
            if hasattr(constraint, "le"):
                config["max"] = constraint.le
            if hasattr(constraint, "multiple_of"):
                config["step"] = constraint.multiple_of
    return config


def _model_field_to_image_input(field_info: FieldInfo, **kwargs) -> NodeInput:
    return IO.IMAGE, {
        **_create_base_config(field_info),
        **kwargs,
    }


def _model_field_to_string_input(field_info: FieldInfo, **kwargs) -> NodeInput:
    return IO.STRING, {
        **_create_base_config(field_info),
        **kwargs,
    }


def _model_field_to_float_input(field_info: FieldInfo, **kwargs) -> NodeInput:
    return IO.FLOAT, {
        **_create_base_config(field_info),
        **_get_number_constraints_config(field_info),
        **kwargs,
    }


def _model_field_to_int_input(field_info: FieldInfo, **kwargs) -> NodeInput:
    return IO.INT, {
        **_create_base_config(field_info),
        **_get_number_constraints_config(field_info),
        **kwargs,
    }


def _model_field_to_combo_input(
    field_info: FieldInfo, enum_type: type[Enum] = None, **kwargs
) -> NodeInput:
    combo_config = {}
    if enum_type is not None:
        combo_config["options"] = [option.value for option in enum_type]
    combo_config = {
        **combo_config,
        **_create_base_config(field_info),
        **kwargs,
    }
    return IO.COMBO, combo_config


def model_field_to_node_input(
    input_type: IO, base_model: type[BaseModel], field_name: str, **kwargs
) -> NodeInput:
    """
    Maps a field from a Pydantic model to a Comfy node input.

    Args:
        input_type: The type of the input.
        base_model: The Pydantic model to map the field from.
        field_name: The name of the field to map.
        **kwargs: Additional key/values to include in the input options.

    Note:
        For combo inputs, pass an `Enum` to the `enum_type` keyword argument to populate the options automatically.

    Example:
        >>> model_field_to_node_input(IO.STRING, MyModel, "my_field", multiline=True)
        >>> model_field_to_node_input(IO.COMBO, MyModel, "my_field", enum_type=MyEnum)
        >>> model_field_to_node_input(IO.FLOAT, MyModel, "my_field", slider=True)
    """
    field_info: FieldInfo = base_model.model_fields[field_name]
    result: NodeInput

    if input_type == IO.IMAGE:
        result = _model_field_to_image_input(field_info, **kwargs)
    elif input_type == IO.STRING:
        result = _model_field_to_string_input(field_info, **kwargs)
    elif input_type == IO.FLOAT:
        result = _model_field_to_float_input(field_info, **kwargs)
    elif input_type == IO.INT:
        result = _model_field_to_int_input(field_info, **kwargs)
    elif input_type == IO.COMBO:
        result = _model_field_to_combo_input(field_info, **kwargs)
    else:
        message = f"Invalid input type: {input_type}"
        raise ValueError(message)

    return result
