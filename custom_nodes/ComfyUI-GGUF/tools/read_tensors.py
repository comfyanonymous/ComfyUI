#!/usr/bin/python3
import os
import sys
import gguf

def read_tensors(path):
    reader = gguf.GGUFReader(path)
    for tensor in reader.tensors:
        if tensor.tensor_type == gguf.GGMLQuantizationType.F32:
            continue
        print(f"{str(tensor.tensor_type):32}: {tensor.name}")

try:
    path = sys.argv[1]
    assert os.path.isfile(path), "Invalid path"
    print(f"input: {path}")
except Exception as e:
    input(f"failed: {e}")
else:
    read_tensors(path)
    input()
