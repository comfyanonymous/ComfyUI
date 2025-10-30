import hashlib
from PIL import ImageFile, UnidentifiedImageError

from comfy_api.latest import io
from .component_model.files import get_package_as_path
from .execution_context import current_execution_context


def conditioning_set_values(conditioning, values: dict = None, append=False) -> io.Conditioning.CondList:
    if values is None:
        values = {}
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            val = values[k]
            if append:
                old_val = n[1].get(k, None)
                if old_val is not None:
                    val = old_val + val

            n[1][k] = val
        c.append(n)

    return c


def pillow(fn, arg):
    prev_value = None
    try:
        x = fn(arg)
    except (OSError, UnidentifiedImageError, ValueError):  # PIL issues #4472 and #2445, also fixes ComfyUI issue #3416
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = fn(arg)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
    return x


def hasher():
    hashfuncs = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512
    }
    args = current_execution_context().configuration
    return hashfuncs[args.default_hashing_function]


def export_custom_nodes():
    """
    Finds all non-abstract classes in the current module that extend CustomNode and creates
    a NODE_CLASS_MAPPINGS dictionary mapping class names to class objects.
    Must be called from within the module where the CustomNode classes are defined.
    """
    import inspect
    from .nodes.package_typing import CustomNode

    # Get the calling module
    frame = inspect.currentframe()
    try:
        module = inspect.getmodule(frame.f_back)

        custom_nodes = {}
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                    CustomNode in obj.__mro__ and
                    obj != CustomNode and
                    not inspect.isabstract(obj)):
                custom_nodes[name] = obj
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            node_class_mappings: dict = getattr(module, 'NODE_CLASS_MAPPINGS')
            node_class_mappings.update(custom_nodes)
        else:
            setattr(module, 'NODE_CLASS_MAPPINGS', custom_nodes)

    finally:
        # Clean up circular reference
        del frame

    return custom_nodes


def export_package_as_web_directory(package: str):
    import inspect

    # Get the calling module
    frame = inspect.currentframe()
    try:
        module = inspect.getmodule(frame.f_back)
        setattr(module, 'WEB_DIRECTORY', get_package_as_path(package))

    finally:
        # Clean up circular reference
        del frame


def string_to_torch_dtype(string):
    import torch
    if string == "fp32":
        return torch.float32
    if string == "fp16":
        return torch.float16
    if string == "bf16":
        return torch.bfloat16


def image_alpha_fix(destination, source):
    import torch
    if destination.shape[-1] < source.shape[-1]:
        source = source[..., :destination.shape[-1]]
    elif destination.shape[-1] > source.shape[-1]:
        destination = torch.nn.functional.pad(destination, (0, 1))
        destination[..., -1] = 1.0
    return destination, source
