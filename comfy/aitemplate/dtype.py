#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
dtype definitions and utility functions of AITemplate
"""


_DTYPE2BYTE = {
    "bool": 1,
    "float16": 2,
    "float32": 4,
    "float": 4,
    "int": 4,
    "int32": 4,
    "int64": 8,
    "bfloat16": 2,
}


# Maps dtype strings to AITemplateDtype enum in model_interface.h.
# Must be kept in sync!
# We can consider defining an AITemplateDtype enum to use on the Python
# side at some point, but stick to strings for now to keep things consistent
# with other Python APIs.
_DTYPE_TO_ENUM = {
    "float16": 1,
    "float32": 2,
    "float": 2,
    "int": 3,
    "int32": 3,
    "int64": 4,
    "bool": 5,
    "bfloat16": 6,
}


def get_dtype_size(dtype: str) -> int:
    """Returns size (in bytes) of the given dtype str.

    Parameters
    ----------
    dtype: str
        A data type string.

    Returns
    ----------
    int
        Size (in bytes) of this dtype.
    """

    if dtype not in _DTYPE2BYTE:
        raise KeyError(f"Unknown dtype: {dtype}. Expected one of {_DTYPE2BYTE.keys()}")
    return _DTYPE2BYTE[dtype]


def normalize_dtype(dtype: str) -> str:
    """Returns a normalized dtype str.

    Parameters
    ----------
    dtype: str
        A data type string.

    Returns
    ----------
    str
        normalized dtype str.
    """
    if dtype == "int":
        return "int32"
    if dtype == "float":
        return "float32"
    return dtype


def dtype_str_to_enum(dtype: str) -> int:
    """Returns the AITemplateDtype enum value (defined in model_interface.h) of
    the given dtype str.

    Parameters
    ----------
    dtype: str
        A data type string.

    Returns
    ----------
    int
        the AITemplateDtype enum value.
    """
    if dtype not in _DTYPE_TO_ENUM:
        raise ValueError(
            f"Got unsupported input dtype {dtype}! Supported dtypes are: {list(_DTYPE_TO_ENUM.keys())}"
        )
    return _DTYPE_TO_ENUM[dtype]


def dtype_to_enumerator(dtype: str) -> str:
    """Returns the string representation of the AITemplateDtype enum
    (defined in model_interface.h) for the given dtype str.

    Parameters
    ----------
    dtype: str
        A data type string.

    Returns
    ----------
    str
        the AITemplateDtype enum string representation.
    """

    def _impl(dtype):
        if dtype == "float16":
            return "kHalf"
        elif dtype == "float32" or dtype == "float":
            return "kFloat"
        elif dtype == "int32" or dtype == "int":
            return "kInt"
        elif dtype == "int64":
            return "kLong"
        elif dtype == "bool":
            return "kBool"
        elif dtype == "bfloat16":
            return "kBFloat16"
        else:
            raise AssertionError(f"unknown dtype {dtype}")

    return f"AITemplateDtype::{_impl(dtype)}"


def is_same_dtype(dtype1: str, dtype2: str) -> bool:
    """Returns True if dtype1 and dtype2 are the same dtype and False otherwise.

    Parameters
    ----------
    dtype1: str
        A data type string.
    dtype2: str
        A data type string.

    Returns
    ----------
    bool
        whether dtype1 and dtype2 are the same dtype
    """
    return normalize_dtype(dtype1) == normalize_dtype(dtype2)
