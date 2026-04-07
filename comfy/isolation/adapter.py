# pylint: disable=import-outside-toplevel,logging-fstring-interpolation,protected-access,raise-missing-from,useless-return,wrong-import-position
from __future__ import annotations

import logging
import os
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

from pyisolate.interfaces import IsolationAdapter, SerializerRegistryProtocol  # type: ignore[import-untyped]
from pyisolate._internal.rpc_protocol import AsyncRPC, ProxiedSingleton  # type: ignore[import-untyped]

_IMPORT_TORCH = os.environ.get("PYISOLATE_IMPORT_TORCH", "1") == "1"

# Singleton proxies that do NOT transitively import torch/PIL/psutil/aiohttp.
# Safe to import in sealed workers without host framework modules.
from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
from comfy.isolation.proxies.helper_proxies import HelperProxiesService
from comfy.isolation.proxies.web_directory_proxy import WebDirectoryProxy

# Singleton proxies that transitively import torch, PIL, or heavy host modules.
# Only available when torch/host framework is present.
CLIPProxy = None
CLIPRegistry = None
ModelPatcherProxy = None
ModelPatcherRegistry = None
ModelSamplingProxy = None
ModelSamplingRegistry = None
VAEProxy = None
VAERegistry = None
FirstStageModelRegistry = None
ModelManagementProxy = None
PromptServerService = None
ProgressProxy = None
UtilsProxy = None
_HAS_TORCH_PROXIES = False
if _IMPORT_TORCH:
    from comfy.isolation.clip_proxy import CLIPProxy, CLIPRegistry
    from comfy.isolation.model_patcher_proxy import (
        ModelPatcherProxy,
        ModelPatcherRegistry,
    )
    from comfy.isolation.model_sampling_proxy import (
        ModelSamplingProxy,
        ModelSamplingRegistry,
    )
    from comfy.isolation.vae_proxy import VAEProxy, VAERegistry, FirstStageModelRegistry
    from comfy.isolation.proxies.model_management_proxy import ModelManagementProxy
    from comfy.isolation.proxies.prompt_server_impl import PromptServerService
    from comfy.isolation.proxies.progress_proxy import ProgressProxy
    from comfy.isolation.proxies.utils_proxy import UtilsProxy
    _HAS_TORCH_PROXIES = True

logger = logging.getLogger(__name__)

# Force /dev/shm for shared memory (bwrap makes /tmp private)
import tempfile

if os.path.exists("/dev/shm"):
    # Only override if not already set or if default is not /dev/shm
    current_tmp = tempfile.gettempdir()
    if not current_tmp.startswith("/dev/shm"):
        logger.debug(
            f"Configuring shared memory: Changing TMPDIR from {current_tmp} to /dev/shm"
        )
        os.environ["TMPDIR"] = "/dev/shm"
        tempfile.tempdir = None  # Clear cache to force re-evaluation


class ComfyUIAdapter(IsolationAdapter):
    # ComfyUI-specific IsolationAdapter implementation

    @property
    def identifier(self) -> str:
        return "comfyui"

    def get_path_config(self, module_path: str) -> Optional[Dict[str, Any]]:
        if "ComfyUI" in module_path and "custom_nodes" in module_path:
            parts = module_path.split("ComfyUI")
            if len(parts) > 1:
                comfy_root = parts[0] + "ComfyUI"
                return {
                    "preferred_root": comfy_root,
                    "additional_paths": [
                        os.path.join(comfy_root, "custom_nodes"),
                        os.path.join(comfy_root, "comfy"),
                    ],
                    "filtered_subdirs": ["comfy", "app", "comfy_execution", "utils"],
                }
        return None

    def get_sandbox_system_paths(self) -> Optional[List[str]]:
        """Returns required application paths to mount in the sandbox."""
        # By inspecting where our adapter is loaded from, we can determine the comfy root
        adapter_file = inspect.getfile(self.__class__)
        # adapter_file = /home/johnj/ComfyUI/comfy/isolation/adapter.py
        comfy_root = os.path.dirname(os.path.dirname(os.path.dirname(adapter_file)))
        if os.path.exists(comfy_root):
            return [comfy_root]
        return None

    def setup_child_environment(self, snapshot: Dict[str, Any]) -> None:
        comfy_root = snapshot.get("preferred_root")
        if not comfy_root:
            return

        requirements_path = Path(comfy_root) / "requirements.txt"
        if requirements_path.exists():
            import re

            for line in requirements_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pkg_name = re.split(r"[<>=!~\[]", line)[0].strip()
                if pkg_name:
                    logging.getLogger(pkg_name).setLevel(logging.ERROR)

    def register_serializers(self, registry: SerializerRegistryProtocol) -> None:
        if not _IMPORT_TORCH:
            # Sealed worker without torch — register torch-free TensorValue handler
            # so IMAGE/MASK/LATENT tensors arrive as numpy arrays, not raw dicts.
            import numpy as np

            _TORCH_DTYPE_TO_NUMPY = {
                "torch.float32": np.float32,
                "torch.float64": np.float64,
                "torch.float16": np.float16,
                "torch.bfloat16": np.float32,  # numpy has no bfloat16; upcast
                "torch.int32": np.int32,
                "torch.int64": np.int64,
                "torch.int16": np.int16,
                "torch.int8": np.int8,
                "torch.uint8": np.uint8,
                "torch.bool": np.bool_,
            }

            def _deserialize_tensor_value(data: Dict[str, Any]) -> Any:
                dtype_str = data["dtype"]
                np_dtype = _TORCH_DTYPE_TO_NUMPY.get(dtype_str, np.float32)
                shape = tuple(data["tensor_size"])
                arr = np.array(data["data"], dtype=np_dtype).reshape(shape)
                return arr

            _NUMPY_TO_TORCH_DTYPE = {
                np.float32: "torch.float32",
                np.float64: "torch.float64",
                np.float16: "torch.float16",
                np.int32: "torch.int32",
                np.int64: "torch.int64",
                np.int16: "torch.int16",
                np.int8: "torch.int8",
                np.uint8: "torch.uint8",
                np.bool_: "torch.bool",
            }

            def _serialize_tensor_value(obj: Any) -> Dict[str, Any]:
                arr = np.asarray(obj, dtype=np.float32) if obj.dtype not in _NUMPY_TO_TORCH_DTYPE else np.asarray(obj)
                dtype_str = _NUMPY_TO_TORCH_DTYPE.get(arr.dtype.type, "torch.float32")
                return {
                    "__type__": "TensorValue",
                    "dtype": dtype_str,
                    "tensor_size": list(arr.shape),
                    "requires_grad": False,
                    "data": arr.tolist(),
                }

            registry.register("TensorValue", _serialize_tensor_value, _deserialize_tensor_value, data_type=True)
            # ndarray output from sealed workers serializes as TensorValue for host torch reconstruction
            registry.register("ndarray", _serialize_tensor_value, _deserialize_tensor_value, data_type=True)
            return

        import torch

        def serialize_device(obj: Any) -> Dict[str, Any]:
            return {"__type__": "device", "device_str": str(obj)}

        def deserialize_device(data: Dict[str, Any]) -> Any:
            return torch.device(data["device_str"])

        registry.register("device", serialize_device, deserialize_device)

        _VALID_DTYPES = {
            "float16", "float32", "float64", "bfloat16",
            "int8", "int16", "int32", "int64",
            "uint8", "bool",
        }

        def serialize_dtype(obj: Any) -> Dict[str, Any]:
            return {"__type__": "dtype", "dtype_str": str(obj)}

        def deserialize_dtype(data: Dict[str, Any]) -> Any:
            dtype_name = data["dtype_str"].replace("torch.", "")
            if dtype_name not in _VALID_DTYPES:
                raise ValueError(f"Invalid dtype: {data['dtype_str']}")
            return getattr(torch, dtype_name)

        registry.register("dtype", serialize_dtype, deserialize_dtype)

        from comfy_api.latest._io import FolderType
        from comfy_api.latest._ui import SavedImages, SavedResult

        def serialize_saved_result(obj: Any) -> Dict[str, Any]:
            return {
                "__type__": "SavedResult",
                "filename": obj.filename,
                "subfolder": obj.subfolder,
                "folder_type": obj.type.value,
            }

        def deserialize_saved_result(data: Dict[str, Any]) -> Any:
            if isinstance(data, SavedResult):
                return data
            folder_type = data["folder_type"] if "folder_type" in data else data["type"]
            return SavedResult(
                filename=data["filename"],
                subfolder=data["subfolder"],
                type=FolderType(folder_type),
            )

        registry.register(
            "SavedResult",
            serialize_saved_result,
            deserialize_saved_result,
            data_type=True,
        )

        def serialize_saved_images(obj: Any) -> Dict[str, Any]:
            return {
                "__type__": "SavedImages",
                "results": [serialize_saved_result(result) for result in obj.results],
                "is_animated": obj.is_animated,
            }

        def deserialize_saved_images(data: Dict[str, Any]) -> Any:
            return SavedImages(
                results=[deserialize_saved_result(result) for result in data["results"]],
                is_animated=data.get("is_animated", False),
            )

        registry.register(
            "SavedImages",
            serialize_saved_images,
            deserialize_saved_images,
            data_type=True,
        )

        def serialize_model_patcher(obj: Any) -> Dict[str, Any]:
            # Child-side: must already have _instance_id (proxy)
            if os.environ.get("PYISOLATE_CHILD") == "1":
                if hasattr(obj, "_instance_id"):
                    return {"__type__": "ModelPatcherRef", "model_id": obj._instance_id}
                raise RuntimeError(
                    f"ModelPatcher in child lacks _instance_id: "
                    f"{type(obj).__module__}.{type(obj).__name__}"
                )
            # Host-side: register with registry
            if hasattr(obj, "_instance_id"):
                return {"__type__": "ModelPatcherRef", "model_id": obj._instance_id}
            model_id = ModelPatcherRegistry().register(obj)
            return {"__type__": "ModelPatcherRef", "model_id": model_id}

        def deserialize_model_patcher(data: Any) -> Any:
            """Deserialize ModelPatcher refs; pass through already-materialized objects."""
            if isinstance(data, dict):
                return ModelPatcherProxy(
                    data["model_id"], registry=None, manage_lifecycle=False
                )
            return data

        def deserialize_model_patcher_ref(data: Dict[str, Any]) -> Any:
            """Context-aware ModelPatcherRef deserializer for both host and child."""
            is_child = os.environ.get("PYISOLATE_CHILD") == "1"
            if is_child:
                return ModelPatcherProxy(
                    data["model_id"], registry=None, manage_lifecycle=False
                )
            else:
                return ModelPatcherRegistry()._get_instance(data["model_id"])

        # Register ModelPatcher type for serialization
        registry.register(
            "ModelPatcher", serialize_model_patcher, deserialize_model_patcher
        )
        # Register ModelPatcherProxy type (already a proxy, just return ref)
        registry.register(
            "ModelPatcherProxy", serialize_model_patcher, deserialize_model_patcher
        )
        # Register ModelPatcherRef for deserialization (context-aware: host or child)
        registry.register("ModelPatcherRef", None, deserialize_model_patcher_ref)

        def serialize_clip(obj: Any) -> Dict[str, Any]:
            if hasattr(obj, "_instance_id"):
                return {"__type__": "CLIPRef", "clip_id": obj._instance_id}
            clip_id = CLIPRegistry().register(obj)
            return {"__type__": "CLIPRef", "clip_id": clip_id}

        def deserialize_clip(data: Any) -> Any:
            if isinstance(data, dict):
                return CLIPProxy(data["clip_id"], registry=None, manage_lifecycle=False)
            return data

        def deserialize_clip_ref(data: Dict[str, Any]) -> Any:
            """Context-aware CLIPRef deserializer for both host and child."""
            is_child = os.environ.get("PYISOLATE_CHILD") == "1"
            if is_child:
                return CLIPProxy(data["clip_id"], registry=None, manage_lifecycle=False)
            else:
                return CLIPRegistry()._get_instance(data["clip_id"])

        # Register CLIP type for serialization
        registry.register("CLIP", serialize_clip, deserialize_clip)
        # Register CLIPProxy type (already a proxy, just return ref)
        registry.register("CLIPProxy", serialize_clip, deserialize_clip)
        # Register CLIPRef for deserialization (context-aware: host or child)
        registry.register("CLIPRef", None, deserialize_clip_ref)

        def serialize_vae(obj: Any) -> Dict[str, Any]:
            if hasattr(obj, "_instance_id"):
                return {"__type__": "VAERef", "vae_id": obj._instance_id}
            vae_id = VAERegistry().register(obj)
            return {"__type__": "VAERef", "vae_id": vae_id}

        def deserialize_vae(data: Any) -> Any:
            if isinstance(data, dict):
                return VAEProxy(data["vae_id"])
            return data

        def deserialize_vae_ref(data: Dict[str, Any]) -> Any:
            """Context-aware VAERef deserializer for both host and child."""
            is_child = os.environ.get("PYISOLATE_CHILD") == "1"
            if is_child:
                # Child: create a proxy
                return VAEProxy(data["vae_id"])
            else:
                # Host: lookup real VAE from registry
                return VAERegistry()._get_instance(data["vae_id"])

        # Register VAE type for serialization
        registry.register("VAE", serialize_vae, deserialize_vae)
        # Register VAEProxy type (already a proxy, just return ref)
        registry.register("VAEProxy", serialize_vae, deserialize_vae)
        # Register VAERef for deserialization (context-aware: host or child)
        registry.register("VAERef", None, deserialize_vae_ref)

        # ModelSampling serialization - handles ModelSampling* types
        # copyreg removed - no pickle fallback allowed

        def serialize_model_sampling(obj: Any) -> Dict[str, Any]:
            # Proxy with _instance_id — return ref (works from both host and child)
            if hasattr(obj, "_instance_id"):
                return {"__type__": "ModelSamplingRef", "ms_id": obj._instance_id}
            # Child-side: object created locally in child (e.g. ModelSamplingAdvanced
            # in nodes_z_image_turbo.py). Serialize as inline data so the host can
            # reconstruct the real torch.nn.Module.
            if os.environ.get("PYISOLATE_CHILD") == "1":
                import base64
                import io as _io

                # Identify base classes from comfy.model_sampling
                bases = []
                for base in type(obj).__mro__:
                    if base.__module__ == "comfy.model_sampling" and base.__name__ != "object":
                        bases.append(base.__name__)
                # Serialize state_dict as base64 safetensors-like
                sd = obj.state_dict()
                sd_serialized = {}
                for k, v in sd.items():
                    buf = _io.BytesIO()
                    torch.save(v, buf)
                    sd_serialized[k] = base64.b64encode(buf.getvalue()).decode("ascii")
                # Capture plain attrs (shift, multiplier, sigma_data, etc.)
                plain_attrs = {}
                for k, v in obj.__dict__.items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, (bool, int, float, str)):
                        plain_attrs[k] = v
                return {
                    "__type__": "ModelSamplingInline",
                    "bases": bases,
                    "state_dict": sd_serialized,
                    "attrs": plain_attrs,
                }
            # Host-side: register with ModelSamplingRegistry and return JSON-safe dict
            ms_id = ModelSamplingRegistry().register(obj)
            return {"__type__": "ModelSamplingRef", "ms_id": ms_id}

        def deserialize_model_sampling(data: Any) -> Any:
            """Deserialize ModelSampling refs or inline data."""
            if isinstance(data, dict):
                if data.get("__type__") == "ModelSamplingInline":
                    return _reconstruct_model_sampling_inline(data)
                return ModelSamplingProxy(data["ms_id"])
            return data

        def _reconstruct_model_sampling_inline(data: Dict[str, Any]) -> Any:
            """Reconstruct a ModelSampling object on the host from inline child data."""
            import comfy.model_sampling as _ms
            import base64
            import io as _io

            # Resolve base classes
            base_classes = []
            for name in data["bases"]:
                cls = getattr(_ms, name, None)
                if cls is not None:
                    base_classes.append(cls)
            if not base_classes:
                raise RuntimeError(
                    f"Cannot reconstruct ModelSampling: no known bases in {data['bases']}"
                )
            # Create dynamic class matching the child's class hierarchy
            ReconstructedSampling = type("ReconstructedSampling", tuple(base_classes), {})
            obj = ReconstructedSampling.__new__(ReconstructedSampling)
            torch.nn.Module.__init__(obj)
            # Restore plain attributes first
            for k, v in data.get("attrs", {}).items():
                setattr(obj, k, v)
            # Restore state_dict (buffers like sigmas)
            for k, v_b64 in data.get("state_dict", {}).items():
                buf = _io.BytesIO(base64.b64decode(v_b64))
                tensor = torch.load(buf, weights_only=True)
                # Register as buffer so it's part of state_dict
                parts = k.split(".")
                if len(parts) == 1:
                    cast(Any, obj).register_buffer(parts[0], tensor)  # pylint: disable=no-member
                else:
                    setattr(obj, parts[0], tensor)
            # Register on host so future references use proxy pattern.
            # Skip in child process — register() is async RPC and cannot be
            # called synchronously during deserialization.
            if os.environ.get("PYISOLATE_CHILD") != "1":
                ModelSamplingRegistry().register(obj)
            return obj

        def deserialize_model_sampling_ref(data: Dict[str, Any]) -> Any:
            """Context-aware ModelSamplingRef deserializer for both host and child."""
            is_child = os.environ.get("PYISOLATE_CHILD") == "1"
            if is_child:
                return ModelSamplingProxy(data["ms_id"])
            else:
                return ModelSamplingRegistry()._get_instance(data["ms_id"])

        # Register all ModelSampling* and StableCascadeSampling classes dynamically
        import comfy.model_sampling

        for ms_cls in vars(comfy.model_sampling).values():
            if not isinstance(ms_cls, type):
                continue
            if not issubclass(ms_cls, torch.nn.Module):
                continue
            if not (ms_cls.__name__.startswith("ModelSampling") or ms_cls.__name__ == "StableCascadeSampling"):
                continue
            registry.register(
                ms_cls.__name__,
                serialize_model_sampling,
                deserialize_model_sampling,
            )
        registry.register(
            "ModelSamplingProxy", serialize_model_sampling, deserialize_model_sampling
        )
        # Register ModelSamplingRef for deserialization (context-aware: host or child)
        registry.register("ModelSamplingRef", None, deserialize_model_sampling_ref)
        # Register ModelSamplingInline for deserialization (child→host inline transfer)
        registry.register(
            "ModelSamplingInline", None, lambda data: _reconstruct_model_sampling_inline(data)
        )

        def serialize_cond(obj: Any) -> Dict[str, Any]:
            type_key = f"{type(obj).__module__}.{type(obj).__name__}"
            return {
                "__type__": type_key,
                "cond": obj.cond,
            }

        def deserialize_cond(data: Dict[str, Any]) -> Any:
            import importlib

            type_key = data["__type__"]
            module_name, class_name = type_key.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls(data["cond"])

        def _serialize_public_state(obj: Any) -> Dict[str, Any]:
            state: Dict[str, Any] = {}
            for key, value in obj.__dict__.items():
                if key.startswith("_"):
                    continue
                if callable(value):
                    continue
                state[key] = value
            return state

        def serialize_latent_format(obj: Any) -> Dict[str, Any]:
            type_key = f"{type(obj).__module__}.{type(obj).__name__}"
            return {
                "__type__": type_key,
                "state": _serialize_public_state(obj),
            }

        def deserialize_latent_format(data: Dict[str, Any]) -> Any:
            import importlib

            type_key = data["__type__"]
            module_name, class_name = type_key.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            obj = cls()
            for key, value in data.get("state", {}).items():
                prop = getattr(type(obj), key, None)
                if isinstance(prop, property) and prop.fset is None:
                    continue
                setattr(obj, key, value)
            return obj

        import comfy.conds

        for cond_cls in vars(comfy.conds).values():
            if not isinstance(cond_cls, type):
                continue
            if not issubclass(cond_cls, comfy.conds.CONDRegular):
                continue
            type_key = f"{cond_cls.__module__}.{cond_cls.__name__}"
            registry.register(type_key, serialize_cond, deserialize_cond)
            registry.register(cond_cls.__name__, serialize_cond, deserialize_cond)

        import comfy.latent_formats

        for latent_cls in vars(comfy.latent_formats).values():
            if not isinstance(latent_cls, type):
                continue
            if not issubclass(latent_cls, comfy.latent_formats.LatentFormat):
                continue
            type_key = f"{latent_cls.__module__}.{latent_cls.__name__}"
            registry.register(
                type_key, serialize_latent_format, deserialize_latent_format
            )
            registry.register(
                latent_cls.__name__, serialize_latent_format, deserialize_latent_format
            )

        # V3 API: unwrap NodeOutput.args
        def deserialize_node_output(data: Any) -> Any:
            return getattr(data, "args", data)

        registry.register("NodeOutput", None, deserialize_node_output)

        # KSAMPLER serializer: stores sampler name instead of function object
        # sampler_function is a callable which gets filtered out by JSONSocketTransport
        def serialize_ksampler(obj: Any) -> Dict[str, Any]:
            func_name = obj.sampler_function.__name__
            # Map function name back to sampler name
            if func_name == "sample_unipc":
                sampler_name = "uni_pc"
            elif func_name == "sample_unipc_bh2":
                sampler_name = "uni_pc_bh2"
            elif func_name == "dpm_fast_function":
                sampler_name = "dpm_fast"
            elif func_name == "dpm_adaptive_function":
                sampler_name = "dpm_adaptive"
            elif func_name.startswith("sample_"):
                sampler_name = func_name[7:]  # Remove "sample_" prefix
            else:
                sampler_name = func_name
            return {
                "__type__": "KSAMPLER",
                "sampler_name": sampler_name,
                "extra_options": obj.extra_options,
                "inpaint_options": obj.inpaint_options,
            }

        def deserialize_ksampler(data: Dict[str, Any]) -> Any:
            import comfy.samplers

            return comfy.samplers.ksampler(
                data["sampler_name"],
                data.get("extra_options", {}),
                data.get("inpaint_options", {}),
            )

        registry.register("KSAMPLER", serialize_ksampler, deserialize_ksampler)

        from comfy.isolation.model_patcher_proxy_utils import register_hooks_serializers

        register_hooks_serializers(registry)

        # Generic Numpy Serializer
        def serialize_numpy(obj: Any) -> Any:
            import torch

            try:
                # Attempt zero-copy conversion to Tensor
                return torch.from_numpy(obj)
            except Exception:
                # Fallback for non-numeric arrays (strings, objects, mixes)
                return obj.tolist()

        def deserialize_numpy_b64(data: Any) -> Any:
            """Deserialize base64-encoded ndarray from sealed worker."""
            import base64
            import numpy as np
            if isinstance(data, dict) and "data" in data and "dtype" in data:
                raw = base64.b64decode(data["data"])
                arr = np.frombuffer(raw, dtype=np.dtype(data["dtype"])).reshape(data["shape"])
                return torch.from_numpy(arr.copy())
            return data

        registry.register("ndarray", serialize_numpy, deserialize_numpy_b64)

        # -- File3D (comfy_api.latest._util.geometry_types) ---------------------
        # Origin: comfy_api by ComfyOrg (Alexander Piskun), PR #12129

        def serialize_file3d(obj: Any) -> Dict[str, Any]:
            import base64
            return {
                "__type__": "File3D",
                "format": obj.format,
                "data": base64.b64encode(obj.get_bytes()).decode("ascii"),
            }

        def deserialize_file3d(data: Any) -> Any:
            import base64
            from io import BytesIO
            from comfy_api.latest._util.geometry_types import File3D
            return File3D(BytesIO(base64.b64decode(data["data"])), file_format=data["format"])

        registry.register("File3D", serialize_file3d, deserialize_file3d, data_type=True)

        # -- VIDEO (comfy_api.latest._input_impl.video_types) -------------------
        # Origin: ComfyAPI Core v0.0.2 by ComfyOrg (guill), PR #8962

        def serialize_video(obj: Any) -> Dict[str, Any]:
            components = obj.get_components()
            images = components.images.detach() if components.images.requires_grad else components.images
            result: Dict[str, Any] = {
                "__type__": "VIDEO",
                "images": images,
                "frame_rate_num": components.frame_rate.numerator,
                "frame_rate_den": components.frame_rate.denominator,
            }
            if components.audio is not None:
                waveform = components.audio["waveform"]
                if waveform.requires_grad:
                    waveform = waveform.detach()
                result["audio_waveform"] = waveform
                result["audio_sample_rate"] = components.audio["sample_rate"]
            if components.metadata is not None:
                result["metadata"] = components.metadata
            return result

        def deserialize_video(data: Any) -> Any:
            from fractions import Fraction
            from comfy_api.latest._input_impl.video_types import VideoFromComponents
            from comfy_api.latest._util.video_types import VideoComponents
            audio = None
            if "audio_waveform" in data:
                audio = {"waveform": data["audio_waveform"], "sample_rate": data["audio_sample_rate"]}
            components = VideoComponents(
                images=data["images"],
                frame_rate=Fraction(data["frame_rate_num"], data["frame_rate_den"]),
                audio=audio,
                metadata=data.get("metadata"),
            )
            return VideoFromComponents(components)

        registry.register("VIDEO", serialize_video, deserialize_video, data_type=True)
        registry.register("VideoFromFile", serialize_video, deserialize_video, data_type=True)
        registry.register("VideoFromComponents", serialize_video, deserialize_video, data_type=True)

    def setup_web_directory(self, module: Any) -> None:
        """Detect WEB_DIRECTORY on a module and populate/register it.

        Called by the sealed worker after loading the node module.
        Mirrors extension_wrapper.py:216-227 for host-coupled nodes.
        Does NOT import extension_wrapper.py (it has `import torch` at module level).
        """
        import shutil

        web_dir_attr = getattr(module, "WEB_DIRECTORY", None)
        if web_dir_attr is None:
            return

        module_dir = os.path.dirname(os.path.abspath(module.__file__))
        web_dir_path = os.path.abspath(os.path.join(module_dir, web_dir_attr))

        # Read extension name from pyproject.toml
        ext_name = os.path.basename(module_dir)
        pyproject = os.path.join(module_dir, "pyproject.toml")
        if os.path.exists(pyproject):
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore[no-redef]
            try:
                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                name = data.get("project", {}).get("name")
                if name:
                    ext_name = name
            except Exception:
                pass

        # Populate web dir if empty (mirrors _run_prestartup_web_copy)
        if not (os.path.isdir(web_dir_path) and any(os.scandir(web_dir_path))):
            os.makedirs(web_dir_path, exist_ok=True)

            # Module-defined copy spec
            copy_spec = getattr(module, "_PRESTARTUP_WEB_COPY", None)
            if copy_spec is not None and callable(copy_spec):
                try:
                    copy_spec(web_dir_path)
                except Exception as e:
                    logger.warning("][ _PRESTARTUP_WEB_COPY failed: %s", e)

            # Fallback: comfy_3d_viewers
            try:
                from comfy_3d_viewers import copy_viewer, VIEWER_FILES
                for viewer in VIEWER_FILES:
                    try:
                        copy_viewer(viewer, web_dir_path)
                    except Exception:
                        pass
            except ImportError:
                pass

            # Fallback: comfy_dynamic_widgets
            try:
                from comfy_dynamic_widgets import get_js_path
                src = os.path.realpath(get_js_path())
                if os.path.exists(src):
                    dst_dir = os.path.join(web_dir_path, "js")
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy2(src, os.path.join(dst_dir, "dynamic_widgets.js"))
            except ImportError:
                pass

        if os.path.isdir(web_dir_path) and any(os.scandir(web_dir_path)):
            WebDirectoryProxy.register_web_dir(ext_name, web_dir_path)
            logger.info(
                "][ Adapter: registered web dir for %s (%d files)",
                ext_name,
                sum(1 for _ in Path(web_dir_path).rglob("*") if _.is_file()),
            )

    @staticmethod
    def register_host_event_handlers(extension: Any) -> None:
        """Register host-side event handlers for an isolated extension.

        Wires ``"progress"`` events from the child to ``comfy.utils.PROGRESS_BAR_HOOK``
        so the ComfyUI frontend receives progress bar updates.
        """
        register_event_handler = inspect.getattr_static(
            extension, "register_event_handler", None
        )
        if not callable(register_event_handler):
            return

        def _host_progress_handler(payload: dict) -> None:
            import comfy.utils

            hook = comfy.utils.PROGRESS_BAR_HOOK
            if hook is not None:
                hook(
                    payload.get("value", 0),
                    payload.get("total", 0),
                    payload.get("preview"),
                    payload.get("node_id"),
                )

        extension.register_event_handler("progress", _host_progress_handler)

    def setup_child_event_hooks(self, extension: Any) -> None:
        """Wire PROGRESS_BAR_HOOK in the child to emit_event on the extension.

        Host-coupled only — sealed workers do not have comfy.utils (torch).
        """
        is_child = os.environ.get("PYISOLATE_CHILD") == "1"
        logger.info("][ ISO:setup_child_event_hooks called, PYISOLATE_CHILD=%s", is_child)
        if not is_child:
            return

        if not _IMPORT_TORCH:
            logger.info("][ ISO:setup_child_event_hooks skipped — sealed worker (no torch)")
            return

        import comfy.utils

        def _event_progress_hook(value, total, preview=None, node_id=None):
            logger.debug("][ ISO:event_progress value=%s/%s node_id=%s", value, total, node_id)
            extension.emit_event("progress", {
                "value": value,
                "total": total,
                "node_id": node_id,
            })

        comfy.utils.PROGRESS_BAR_HOOK = _event_progress_hook
        logger.info("][ ISO:PROGRESS_BAR_HOOK wired to event channel")

    def provide_rpc_services(self) -> List[type[ProxiedSingleton]]:
        # Always available — no torch/PIL dependency
        services: List[type[ProxiedSingleton]] = [
            FolderPathsProxy,
            HelperProxiesService,
            WebDirectoryProxy,
        ]
        # Torch/PIL-dependent proxies
        if _HAS_TORCH_PROXIES:
            services.extend([
                PromptServerService,
                ModelManagementProxy,
                UtilsProxy,
                ProgressProxy,
                VAERegistry,
                CLIPRegistry,
                ModelPatcherRegistry,
                ModelSamplingRegistry,
                FirstStageModelRegistry,
            ])
        return services

    def handle_api_registration(self, api: ProxiedSingleton, rpc: AsyncRPC) -> None:
        # Resolve the real name whether it's an instance or the Singleton class itself
        api_name = api.__name__ if isinstance(api, type) else api.__class__.__name__

        if api_name == "FolderPathsProxy":
            import folder_paths

            # Replace module-level functions with proxy methods
            # This is aggressive but necessary for transparent proxying
            # Handle both instance and class cases
            instance = api() if isinstance(api, type) else api
            for name in dir(instance):
                if not name.startswith("_"):
                    setattr(folder_paths, name, getattr(instance, name))

            # Fence: isolated children get writable temp inside sandbox
            if os.environ.get("PYISOLATE_CHILD") == "1":
                import tempfile
                _child_temp = os.path.join(tempfile.gettempdir(), "comfyui_temp")
                os.makedirs(_child_temp, exist_ok=True)
                folder_paths.temp_directory = _child_temp

            return

        if api_name == "ModelManagementProxy":
            if _IMPORT_TORCH:
                import comfy.model_management

                instance = api() if isinstance(api, type) else api
                # Replace module-level functions with proxy methods
                for name in dir(instance):
                    if not name.startswith("_"):
                        setattr(comfy.model_management, name, getattr(instance, name))
            return

        if api_name == "UtilsProxy":
            if not _IMPORT_TORCH:
                logger.info("][ ISO:UtilsProxy handle_api_registration skipped — sealed worker (no torch)")
                return

            import comfy.utils

            # Static Injection of RPC mechanism to ensure Child can access it
            # independent of instance lifecycle.
            api.set_rpc(rpc)

            is_child = os.environ.get("PYISOLATE_CHILD") == "1"
            logger.info("][ ISO:UtilsProxy handle_api_registration PYISOLATE_CHILD=%s", is_child)

            # Progress hook wiring moved to setup_child_event_hooks via event channel

            return

        if api_name == "PromptServerProxy":
            if not _IMPORT_TORCH:
                return
            # Defer heavy import to child context
            import server

            instance = api() if isinstance(api, type) else api
            proxy = (
                instance.instance
            )  # PromptServerProxy instance has .instance property returning self

            original_register_route = proxy.register_route

            def register_route_wrapper(
                method: str, path: str, handler: Callable[..., Any]
            ) -> None:
                callback_id = rpc.register_callback(handler)
                loop = getattr(rpc, "loop", None)
                if loop and loop.is_running():
                    import asyncio

                    asyncio.create_task(
                        original_register_route(
                            method, path, handler=callback_id, is_callback=True
                        )
                    )
                else:
                    original_register_route(
                        method, path, handler=callback_id, is_callback=True
                    )
                return None

            proxy.register_route = register_route_wrapper

            class RouteTableDefProxy:
                def __init__(self, proxy_instance: Any):
                    self.proxy = proxy_instance

                def get(
                    self, path: str, **kwargs: Any
                ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
                        self.proxy.register_route("GET", path, handler)
                        return handler

                    return decorator

                def post(
                    self, path: str, **kwargs: Any
                ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
                        self.proxy.register_route("POST", path, handler)
                        return handler

                    return decorator

                def patch(
                    self, path: str, **kwargs: Any
                ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
                        self.proxy.register_route("PATCH", path, handler)
                        return handler

                    return decorator

                def put(
                    self, path: str, **kwargs: Any
                ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
                        self.proxy.register_route("PUT", path, handler)
                        return handler

                    return decorator

                def delete(
                    self, path: str, **kwargs: Any
                ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
                        self.proxy.register_route("DELETE", path, handler)
                        return handler

                    return decorator

            proxy.routes = RouteTableDefProxy(proxy)

            if (
                hasattr(server, "PromptServer")
                and getattr(server.PromptServer, "instance", None) != proxy
            ):
                server.PromptServer.instance = proxy
