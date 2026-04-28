# pylint: disable=attribute-defined-outside-init,import-outside-toplevel,logging-fstring-interpolation
# CLIP Proxy implementation
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from comfy.isolation.proxies.base import (
    IS_CHILD_PROCESS,
    BaseProxy,
    BaseRegistry,
    detach_if_grad,
)

if TYPE_CHECKING:
    from comfy.isolation.model_patcher_proxy import ModelPatcherProxy


class CondStageModelRegistry(BaseRegistry[Any]):
    _type_prefix = "cond_stage_model"

    async def get_property(self, instance_id: str, name: str) -> Any:
        obj = self._get_instance(instance_id)
        return getattr(obj, name)


class CondStageModelProxy(BaseProxy[CondStageModelRegistry]):
    _registry_class = CondStageModelRegistry
    __module__ = "comfy.sd"

    def __getattr__(self, name: str) -> Any:
        try:
            return self._call_rpc("get_property", name)
        except Exception as e:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from e

    def __repr__(self) -> str:
        return f"<CondStageModelProxy {self._instance_id}>"


class TokenizerRegistry(BaseRegistry[Any]):
    _type_prefix = "tokenizer"

    async def get_property(self, instance_id: str, name: str) -> Any:
        obj = self._get_instance(instance_id)
        return getattr(obj, name)


class TokenizerProxy(BaseProxy[TokenizerRegistry]):
    _registry_class = TokenizerRegistry
    __module__ = "comfy.sd"

    def __getattr__(self, name: str) -> Any:
        try:
            return self._call_rpc("get_property", name)
        except Exception as e:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from e

    def __repr__(self) -> str:
        return f"<TokenizerProxy {self._instance_id}>"


logger = logging.getLogger(__name__)


class CLIPRegistry(BaseRegistry[Any]):
    _type_prefix = "clip"
    _allowed_setters = {
        "layer_idx",
        "tokenizer_options",
        "use_clip_schedule",
        "apply_hooks_to_conds",
    }

    async def get_ram_usage(self, instance_id: str) -> int:
        return self._get_instance(instance_id).get_ram_usage()

    async def get_patcher_id(self, instance_id: str) -> str:
        from comfy.isolation.model_patcher_proxy import ModelPatcherRegistry

        return ModelPatcherRegistry().register(self._get_instance(instance_id).patcher)

    async def get_cond_stage_model_id(self, instance_id: str) -> str:
        return CondStageModelRegistry().register(
            self._get_instance(instance_id).cond_stage_model
        )

    async def get_tokenizer_id(self, instance_id: str) -> str:
        return TokenizerRegistry().register(self._get_instance(instance_id).tokenizer)

    async def load_model(self, instance_id: str) -> None:
        self._get_instance(instance_id).load_model()

    async def clip_layer(self, instance_id: str, layer_idx: int) -> None:
        self._get_instance(instance_id).clip_layer(layer_idx)

    async def set_tokenizer_option(
        self, instance_id: str, option_name: str, value: Any
    ) -> None:
        self._get_instance(instance_id).set_tokenizer_option(option_name, value)

    async def get_property(self, instance_id: str, name: str) -> Any:
        return getattr(self._get_instance(instance_id), name)

    async def set_property(self, instance_id: str, name: str, value: Any) -> None:
        if name not in self._allowed_setters:
            raise PermissionError(f"Setting '{name}' is not allowed via RPC")
        setattr(self._get_instance(instance_id), name, value)

    async def tokenize(
        self, instance_id: str, text: str, return_word_ids: bool = False, **kwargs: Any
    ) -> Any:
        return self._get_instance(instance_id).tokenize(
            text, return_word_ids=return_word_ids, **kwargs
        )

    async def encode(self, instance_id: str, text: str) -> Any:
        return detach_if_grad(self._get_instance(instance_id).encode(text))

    async def encode_from_tokens(
        self,
        instance_id: str,
        tokens: Any,
        return_pooled: bool = False,
        return_dict: bool = False,
    ) -> Any:
        return detach_if_grad(
            self._get_instance(instance_id).encode_from_tokens(
                tokens, return_pooled=return_pooled, return_dict=return_dict
            )
        )

    async def encode_from_tokens_scheduled(
        self,
        instance_id: str,
        tokens: Any,
        unprojected: bool = False,
        add_dict: Optional[dict] = None,
        show_pbar: bool = True,
    ) -> Any:
        add_dict = add_dict or {}
        return detach_if_grad(
            self._get_instance(instance_id).encode_from_tokens_scheduled(
                tokens, unprojected=unprojected, add_dict=add_dict, show_pbar=show_pbar
            )
        )

    async def add_patches(
        self,
        instance_id: str,
        patches: Any,
        strength_patch: float = 1.0,
        strength_model: float = 1.0,
    ) -> Any:
        return self._get_instance(instance_id).add_patches(
            patches, strength_patch=strength_patch, strength_model=strength_model
        )

    async def get_key_patches(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).get_key_patches()

    async def load_sd(
        self, instance_id: str, sd: dict, full_model: bool = False
    ) -> Any:
        return self._get_instance(instance_id).load_sd(sd, full_model=full_model)

    async def get_sd(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).get_sd()

    async def clone(self, instance_id: str) -> str:
        return self.register(self._get_instance(instance_id).clone())


class CLIPProxy(BaseProxy[CLIPRegistry]):
    _registry_class = CLIPRegistry
    __module__ = "comfy.sd"

    def get_ram_usage(self) -> int:
        return self._call_rpc("get_ram_usage")

    @property
    def patcher(self) -> "ModelPatcherProxy":
        from comfy.isolation.model_patcher_proxy import ModelPatcherProxy

        if not hasattr(self, "_patcher_proxy"):
            patcher_id = self._call_rpc("get_patcher_id")
            self._patcher_proxy = ModelPatcherProxy(patcher_id, manage_lifecycle=False)
        return self._patcher_proxy

    @patcher.setter
    def patcher(self, value: Any) -> None:
        from comfy.isolation.model_patcher_proxy import ModelPatcherProxy

        if isinstance(value, ModelPatcherProxy):
            self._patcher_proxy = value
        else:
            logger.warning(
                f"Attempted to set CLIPProxy.patcher to non-proxy object: {value}"
            )

    @property
    def cond_stage_model(self) -> CondStageModelProxy:
        if not hasattr(self, "_cond_stage_model_proxy"):
            csm_id = self._call_rpc("get_cond_stage_model_id")
            self._cond_stage_model_proxy = CondStageModelProxy(
                csm_id, manage_lifecycle=False
            )
        return self._cond_stage_model_proxy

    @property
    def tokenizer(self) -> TokenizerProxy:
        if not hasattr(self, "_tokenizer_proxy"):
            tok_id = self._call_rpc("get_tokenizer_id")
            self._tokenizer_proxy = TokenizerProxy(tok_id, manage_lifecycle=False)
        return self._tokenizer_proxy

    def load_model(self) -> ModelPatcherProxy:
        self._call_rpc("load_model")
        return self.patcher

    @property
    def layer_idx(self) -> Optional[int]:
        return self._call_rpc("get_property", "layer_idx")

    @layer_idx.setter
    def layer_idx(self, value: Optional[int]) -> None:
        self._call_rpc("set_property", "layer_idx", value)

    @property
    def tokenizer_options(self) -> dict:
        return self._call_rpc("get_property", "tokenizer_options")

    @tokenizer_options.setter
    def tokenizer_options(self, value: dict) -> None:
        self._call_rpc("set_property", "tokenizer_options", value)

    @property
    def use_clip_schedule(self) -> bool:
        return self._call_rpc("get_property", "use_clip_schedule")

    @use_clip_schedule.setter
    def use_clip_schedule(self, value: bool) -> None:
        self._call_rpc("set_property", "use_clip_schedule", value)

    @property
    def apply_hooks_to_conds(self) -> Any:
        return self._call_rpc("get_property", "apply_hooks_to_conds")

    @apply_hooks_to_conds.setter
    def apply_hooks_to_conds(self, value: Any) -> None:
        self._call_rpc("set_property", "apply_hooks_to_conds", value)

    def clip_layer(self, layer_idx: int) -> None:
        return self._call_rpc("clip_layer", layer_idx)

    def set_tokenizer_option(self, option_name: str, value: Any) -> None:
        return self._call_rpc("set_tokenizer_option", option_name, value)

    def tokenize(self, text: str, return_word_ids: bool = False, **kwargs: Any) -> Any:
        return self._call_rpc(
            "tokenize", text, return_word_ids=return_word_ids, **kwargs
        )

    def encode(self, text: str) -> Any:
        return self._call_rpc("encode", text)

    def encode_from_tokens(
        self, tokens: Any, return_pooled: bool = False, return_dict: bool = False
    ) -> Any:
        res = self._call_rpc(
            "encode_from_tokens",
            tokens,
            return_pooled=return_pooled,
            return_dict=return_dict,
        )
        if return_pooled and isinstance(res, list) and not return_dict:
            return tuple(res)
        return res

    def encode_from_tokens_scheduled(
        self,
        tokens: Any,
        unprojected: bool = False,
        add_dict: Optional[dict] = None,
        show_pbar: bool = True,
    ) -> Any:
        add_dict = add_dict or {}
        return self._call_rpc(
            "encode_from_tokens_scheduled",
            tokens,
            unprojected=unprojected,
            add_dict=add_dict,
            show_pbar=show_pbar,
        )

    def add_patches(
        self, patches: Any, strength_patch: float = 1.0, strength_model: float = 1.0
    ) -> Any:
        return self._call_rpc(
            "add_patches",
            patches,
            strength_patch=strength_patch,
            strength_model=strength_model,
        )

    def get_key_patches(self) -> Any:
        return self._call_rpc("get_key_patches")

    def load_sd(self, sd: dict, full_model: bool = False) -> Any:
        return self._call_rpc("load_sd", sd, full_model=full_model)

    def get_sd(self) -> Any:
        return self._call_rpc("get_sd")

    def clone(self) -> CLIPProxy:
        new_id = self._call_rpc("clone")
        return CLIPProxy(new_id, self._registry, manage_lifecycle=not IS_CHILD_PROCESS)


if not IS_CHILD_PROCESS:
    _CLIP_REGISTRY_SINGLETON = CLIPRegistry()
    _COND_STAGE_MODEL_REGISTRY_SINGLETON = CondStageModelRegistry()
    _TOKENIZER_REGISTRY_SINGLETON = TokenizerRegistry()
