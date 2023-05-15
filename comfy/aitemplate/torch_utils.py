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
Functions for working with torch Tensors.
AITemplate doesn't depend on PyTorch, but it exposes
many APIs that work with torch Tensors anyways.

The functions in this file may assume that
`import torch` will work.
"""


def types_mapping():
    from torch import bfloat16, bool, float16, float32, int32, int64

    yield (float16, "float16")
    yield (bfloat16, "bfloat16")
    yield (float32, "float32")
    yield (int32, "int32")
    yield (int64, "int64")
    yield (bool, "bool")


def torch_dtype_to_string(dtype):
    for (torch_dtype, ait_dtype) in types_mapping():
        if dtype == torch_dtype:
            return ait_dtype
    raise ValueError(
        f"Got unsupported input dtype {dtype}! "
        f"Supported dtypes are: {list(types_mapping())}"
    )


def string_to_torch_dtype(string_dtype):
    if string_dtype is None:
        # Many torch functions take optional dtypes, so
        # handling None is useful here.
        return None

    for (torch_dtype, ait_dtype) in types_mapping():
        if string_dtype == ait_dtype:
            return torch_dtype
    raise ValueError(
        f"Got unsupported ait dtype {string_dtype}! "
        f"Supported dtypes are: {list(types_mapping())}"
    )
