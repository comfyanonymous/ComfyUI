# pylint: disable=attribute-defined-outside-init
import logging
from typing import Any

from comfy.isolation.proxies.base import (
    IS_CHILD_PROCESS,
    BaseProxy,
    BaseRegistry,
    detach_if_grad,
)
from comfy.isolation.model_patcher_proxy import ModelPatcherProxy, ModelPatcherRegistry

logger = logging.getLogger(__name__)


class FirstStageModelRegistry(BaseRegistry[Any]):
    _type_prefix = "first_stage_model"

    async def get_property(self, instance_id: str, name: str) -> Any:
        obj = self._get_instance(instance_id)
        return getattr(obj, name)

    async def has_property(self, instance_id: str, name: str) -> bool:
        obj = self._get_instance(instance_id)
        return hasattr(obj, name)


class FirstStageModelProxy(BaseProxy[FirstStageModelRegistry]):
    _registry_class = FirstStageModelRegistry
    __module__ = "comfy.ldm.models.autoencoder"

    def __getattr__(self, name: str) -> Any:
        try:
            return self._call_rpc("get_property", name)
        except Exception as e:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from e

    def __repr__(self) -> str:
        return f"<FirstStageModelProxy {self._instance_id}>"


class VAERegistry(BaseRegistry[Any]):
    _type_prefix = "vae"

    async def get_patcher_id(self, instance_id: str) -> str:
        vae = self._get_instance(instance_id)
        return ModelPatcherRegistry().register(vae.patcher)

    async def get_first_stage_model_id(self, instance_id: str) -> str:
        vae = self._get_instance(instance_id)
        return FirstStageModelRegistry().register(vae.first_stage_model)

    async def encode(self, instance_id: str, pixels: Any) -> Any:
        return detach_if_grad(self._get_instance(instance_id).encode(pixels))

    async def encode_tiled(
        self,
        instance_id: str,
        pixels: Any,
        tile_x: int = 512,
        tile_y: int = 512,
        overlap: int = 64,
    ) -> Any:
        return detach_if_grad(
            self._get_instance(instance_id).encode_tiled(
                pixels, tile_x=tile_x, tile_y=tile_y, overlap=overlap
            )
        )

    async def decode(self, instance_id: str, samples: Any, **kwargs: Any) -> Any:
        return detach_if_grad(self._get_instance(instance_id).decode(samples, **kwargs))

    async def decode_tiled(
        self,
        instance_id: str,
        samples: Any,
        tile_x: int = 64,
        tile_y: int = 64,
        overlap: int = 16,
        **kwargs: Any,
    ) -> Any:
        return detach_if_grad(
            self._get_instance(instance_id).decode_tiled(
                samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap, **kwargs
            )
        )

    async def get_property(self, instance_id: str, name: str) -> Any:
        return getattr(self._get_instance(instance_id), name)

    async def memory_used_encode(self, instance_id: str, shape: Any, dtype: Any) -> int:
        return self._get_instance(instance_id).memory_used_encode(shape, dtype)

    async def memory_used_decode(self, instance_id: str, shape: Any, dtype: Any) -> int:
        return self._get_instance(instance_id).memory_used_decode(shape, dtype)

    async def process_input(self, instance_id: str, image: Any) -> Any:
        return detach_if_grad(self._get_instance(instance_id).process_input(image))

    async def process_output(self, instance_id: str, image: Any) -> Any:
        return detach_if_grad(self._get_instance(instance_id).process_output(image))


class VAEProxy(BaseProxy[VAERegistry]):
    _registry_class = VAERegistry
    __module__ = "comfy.sd"

    @property
    def patcher(self) -> ModelPatcherProxy:
        if not hasattr(self, "_patcher_proxy"):
            patcher_id = self._call_rpc("get_patcher_id")
            self._patcher_proxy = ModelPatcherProxy(patcher_id, manage_lifecycle=False)
        return self._patcher_proxy

    @property
    def first_stage_model(self) -> FirstStageModelProxy:
        if not hasattr(self, "_first_stage_model_proxy"):
            fsm_id = self._call_rpc("get_first_stage_model_id")
            self._first_stage_model_proxy = FirstStageModelProxy(
                fsm_id, manage_lifecycle=False
            )
        return self._first_stage_model_proxy

    @property
    def vae_dtype(self) -> Any:
        return self._get_property("vae_dtype")

    def encode(self, pixels: Any) -> Any:
        return self._call_rpc("encode", pixels)

    def encode_tiled(
        self, pixels: Any, tile_x: int = 512, tile_y: int = 512, overlap: int = 64
    ) -> Any:
        return self._call_rpc("encode_tiled", pixels, tile_x, tile_y, overlap)

    def decode(self, samples: Any, **kwargs: Any) -> Any:
        return self._call_rpc("decode", samples, **kwargs)

    def decode_tiled(
        self,
        samples: Any,
        tile_x: int = 64,
        tile_y: int = 64,
        overlap: int = 16,
        **kwargs: Any,
    ) -> Any:
        return self._call_rpc(
            "decode_tiled", samples, tile_x, tile_y, overlap, **kwargs
        )

    def get_sd(self) -> Any:
        return self._call_rpc("get_sd")

    def _get_property(self, name: str) -> Any:
        return self._call_rpc("get_property", name)

    @property
    def latent_dim(self) -> int:
        return self._get_property("latent_dim")

    @property
    def latent_channels(self) -> int:
        return self._get_property("latent_channels")

    @property
    def downscale_ratio(self) -> Any:
        return self._get_property("downscale_ratio")

    @property
    def upscale_ratio(self) -> Any:
        return self._get_property("upscale_ratio")

    @property
    def output_channels(self) -> int:
        return self._get_property("output_channels")

    @property
    def check_not_vide(self) -> bool:
        return self._get_property("not_video")

    @property
    def device(self) -> Any:
        return self._get_property("device")

    @property
    def working_dtypes(self) -> Any:
        return self._get_property("working_dtypes")

    @property
    def disable_offload(self) -> bool:
        return self._get_property("disable_offload")

    @property
    def size(self) -> Any:
        return self._get_property("size")

    def memory_used_encode(self, shape: Any, dtype: Any) -> int:
        return self._call_rpc("memory_used_encode", shape, dtype)

    def memory_used_decode(self, shape: Any, dtype: Any) -> int:
        return self._call_rpc("memory_used_decode", shape, dtype)

    def process_input(self, image: Any) -> Any:
        return self._call_rpc("process_input", image)

    def process_output(self, image: Any) -> Any:
        return self._call_rpc("process_output", image)


if not IS_CHILD_PROCESS:
    _VAE_REGISTRY_SINGLETON = VAERegistry()
    _FIRST_STAGE_MODEL_REGISTRY_SINGLETON = FirstStageModelRegistry()
