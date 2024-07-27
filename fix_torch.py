import importlib.util
import shutil
import os
import ctypes
import logging


torch_spec = importlib.util.find_spec("torch")
for folder in torch_spec.submodule_search_locations:
    lib_folder = os.path.join(folder, "lib")
    test_file = os.path.join(lib_folder, "fbgemm.dll")
    dest = os.path.join(lib_folder, "libomp140.x86_64.dll")
    if os.path.exists(dest):
        break

    with open(test_file, 'rb') as f:
        contents = f.read()
        if b"libomp140.x86_64.dll" not in contents:
            break
    try:
        mydll = ctypes.cdll.LoadLibrary(test_file)
    except FileNotFoundError as e:
        logging.warning("Detected pytorch version with libomp issue, patching.")
        shutil.copyfile(os.path.join(lib_folder, "libiomp5md.dll"), dest)
