# pylint: disable=import-outside-toplevel
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any

from comfy.isolation.proxies.base import (
    BaseProxy,
    BaseRegistry,
    detach_if_grad,
    get_thread_loop,
    run_coro_in_new_loop,
)

logger = logging.getLogger(__name__)


def _describe_value(obj: Any) -> str:
    try:
        import torch
    except Exception:
        torch = None
    try:
        if torch is not None and isinstance(obj, torch.Tensor):
            return (
                "Tensor(shape=%s,dtype=%s,device=%s,id=%s)"
                % (tuple(obj.shape), obj.dtype, obj.device, id(obj))
            )
    except Exception:
        pass
    return "%s(id=%s)" % (type(obj).__name__, id(obj))


def _prefer_device(*tensors: Any) -> Any:
    try:
        import torch
    except Exception:
        return None
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            return t.device
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return t.device
    return None


def _to_device(obj: Any, device: Any) -> Any:
    try:
        import torch
    except Exception:
        return obj
    if device is None:
        return obj
    if isinstance(obj, torch.Tensor):
        if obj.device != device:
            return obj.to(device)
        return obj
    if isinstance(obj, (list, tuple)):
        converted = [_to_device(x, device) for x in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def _to_cpu_for_rpc(obj: Any) -> Any:
    try:
        import torch
    except Exception:
        return obj
    if isinstance(obj, torch.Tensor):
        t = obj.detach() if obj.requires_grad else obj
        if t.is_cuda:
            return t.to("cpu")
        return t
    if isinstance(obj, (list, tuple)):
        converted = [_to_cpu_for_rpc(x) for x in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    if isinstance(obj, dict):
        return {k: _to_cpu_for_rpc(v) for k, v in obj.items()}
    return obj


class ModelSamplingRegistry(BaseRegistry[Any]):
    _type_prefix = "modelsampling"

    async def calculate_input(self, instance_id: str, sigma: Any, noise: Any) -> Any:
        sampling = self._get_instance(instance_id)
        return detach_if_grad(sampling.calculate_input(sigma, noise))

    async def calculate_denoised(
        self, instance_id: str, sigma: Any, model_output: Any, model_input: Any
    ) -> Any:
        sampling = self._get_instance(instance_id)
        return detach_if_grad(
            sampling.calculate_denoised(sigma, model_output, model_input)
        )

    async def noise_scaling(
        self,
        instance_id: str,
        sigma: Any,
        noise: Any,
        latent_image: Any,
        max_denoise: bool = False,
    ) -> Any:
        sampling = self._get_instance(instance_id)
        return detach_if_grad(
            sampling.noise_scaling(sigma, noise, latent_image, max_denoise=max_denoise)
        )

    async def inverse_noise_scaling(
        self, instance_id: str, sigma: Any, latent: Any
    ) -> Any:
        sampling = self._get_instance(instance_id)
        return detach_if_grad(sampling.inverse_noise_scaling(sigma, latent))

    async def timestep(self, instance_id: str, sigma: Any) -> Any:
        sampling = self._get_instance(instance_id)
        return sampling.timestep(sigma)

    async def sigma(self, instance_id: str, timestep: Any) -> Any:
        sampling = self._get_instance(instance_id)
        return sampling.sigma(timestep)

    async def percent_to_sigma(self, instance_id: str, percent: float) -> Any:
        sampling = self._get_instance(instance_id)
        return sampling.percent_to_sigma(percent)

    async def get_sigma_min(self, instance_id: str) -> Any:
        sampling = self._get_instance(instance_id)
        return detach_if_grad(sampling.sigma_min)

    async def get_sigma_max(self, instance_id: str) -> Any:
        sampling = self._get_instance(instance_id)
        return detach_if_grad(sampling.sigma_max)

    async def get_sigma_data(self, instance_id: str) -> Any:
        sampling = self._get_instance(instance_id)
        return detach_if_grad(sampling.sigma_data)

    async def get_sigmas(self, instance_id: str) -> Any:
        sampling = self._get_instance(instance_id)
        return detach_if_grad(sampling.sigmas)

    async def set_sigmas(self, instance_id: str, sigmas: Any) -> None:
        sampling = self._get_instance(instance_id)
        sampling.set_sigmas(sigmas)


class ModelSamplingProxy(BaseProxy[ModelSamplingRegistry]):
    _registry_class = ModelSamplingRegistry
    __module__ = "comfy.isolation.model_sampling_proxy"

    def _get_rpc(self) -> Any:
        if self._rpc_caller is None:
            from pyisolate._internal.rpc_protocol import get_child_rpc_instance

            rpc = get_child_rpc_instance()
            if rpc is not None:
                self._rpc_caller = rpc.create_caller(
                    ModelSamplingRegistry, ModelSamplingRegistry.get_remote_id()
                )
            else:
                registry = ModelSamplingRegistry()

                class _LocalCaller:
                    def calculate_input(
                        self, instance_id: str, sigma: Any, noise: Any
                    ) -> Any:
                        return registry.calculate_input(instance_id, sigma, noise)

                    def calculate_denoised(
                        self,
                        instance_id: str,
                        sigma: Any,
                        model_output: Any,
                        model_input: Any,
                    ) -> Any:
                        return registry.calculate_denoised(
                            instance_id, sigma, model_output, model_input
                        )

                    def noise_scaling(
                        self,
                        instance_id: str,
                        sigma: Any,
                        noise: Any,
                        latent_image: Any,
                        max_denoise: bool = False,
                    ) -> Any:
                        return registry.noise_scaling(
                            instance_id, sigma, noise, latent_image, max_denoise
                        )

                    def inverse_noise_scaling(
                        self, instance_id: str, sigma: Any, latent: Any
                    ) -> Any:
                        return registry.inverse_noise_scaling(
                            instance_id, sigma, latent
                        )

                    def timestep(self, instance_id: str, sigma: Any) -> Any:
                        return registry.timestep(instance_id, sigma)

                    def sigma(self, instance_id: str, timestep: Any) -> Any:
                        return registry.sigma(instance_id, timestep)

                    def percent_to_sigma(self, instance_id: str, percent: float) -> Any:
                        return registry.percent_to_sigma(instance_id, percent)

                    def get_sigma_min(self, instance_id: str) -> Any:
                        return registry.get_sigma_min(instance_id)

                    def get_sigma_max(self, instance_id: str) -> Any:
                        return registry.get_sigma_max(instance_id)

                    def get_sigma_data(self, instance_id: str) -> Any:
                        return registry.get_sigma_data(instance_id)

                    def get_sigmas(self, instance_id: str) -> Any:
                        return registry.get_sigmas(instance_id)

                    def set_sigmas(self, instance_id: str, sigmas: Any) -> None:
                        return registry.set_sigmas(instance_id, sigmas)

                self._rpc_caller = _LocalCaller()
        return self._rpc_caller

    def _call(self, method_name: str, *args: Any) -> Any:
        rpc = self._get_rpc()
        method = getattr(rpc, method_name)
        result = method(self._instance_id, *args)
        timeout_ms = self._rpc_timeout_ms()
        start_epoch = time.time()
        start_perf = time.perf_counter()
        thread_id = threading.get_ident()
        call_id = "%s:%s:%s:%.6f" % (
            self._instance_id,
            method_name,
            thread_id,
            start_perf,
        )
        logger.debug(
            "ISO:modelsampling_rpc_start method=%s instance_id=%s call_id=%s start_ts=%.6f thread=%s timeout_ms=%s",
            method_name,
            self._instance_id,
            call_id,
            start_epoch,
            thread_id,
            timeout_ms,
        )
        if asyncio.iscoroutine(result):
            result = asyncio.wait_for(result, timeout=timeout_ms / 1000.0)
            try:
                asyncio.get_running_loop()
                out = run_coro_in_new_loop(result)
            except RuntimeError:
                loop = get_thread_loop()
                out = loop.run_until_complete(result)
        else:
            out = result
        logger.debug(
            "ISO:modelsampling_rpc_after_await method=%s instance_id=%s call_id=%s out=%s",
            method_name,
            self._instance_id,
            call_id,
            _describe_value(out),
        )
        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
        logger.debug(
            "ISO:modelsampling_rpc_end method=%s instance_id=%s call_id=%s elapsed_ms=%.3f thread=%s",
            method_name,
            self._instance_id,
            call_id,
            elapsed_ms,
            thread_id,
        )
        logger.debug(
            "ISO:modelsampling_rpc_return method=%s instance_id=%s call_id=%s",
            method_name,
            self._instance_id,
            call_id,
        )
        return out

    @staticmethod
    def _rpc_timeout_ms() -> int:
        raw = os.environ.get(
            "COMFY_ISOLATION_MODEL_SAMPLING_RPC_TIMEOUT_MS",
            os.environ.get("COMFY_ISOLATION_LOAD_RPC_TIMEOUT_MS", "30000"),
        )
        try:
            timeout_ms = int(raw)
        except ValueError:
            timeout_ms = 30000
        return max(1, timeout_ms)

    @property
    def sigma_min(self) -> Any:
        return self._call("get_sigma_min")

    @property
    def sigma_max(self) -> Any:
        return self._call("get_sigma_max")

    @property
    def sigma_data(self) -> Any:
        return self._call("get_sigma_data")

    @property
    def sigmas(self) -> Any:
        return self._call("get_sigmas")

    def calculate_input(self, sigma: Any, noise: Any) -> Any:
        return self._call("calculate_input", sigma, noise)

    def calculate_denoised(
        self, sigma: Any, model_output: Any, model_input: Any
    ) -> Any:
        return self._call("calculate_denoised", sigma, model_output, model_input)

    def noise_scaling(
        self, sigma: Any, noise: Any, latent_image: Any, max_denoise: bool = False
    ) -> Any:
        preferred_device = _prefer_device(noise, latent_image)
        out = self._call(
            "noise_scaling",
            _to_cpu_for_rpc(sigma),
            _to_cpu_for_rpc(noise),
            _to_cpu_for_rpc(latent_image),
            max_denoise,
        )
        return _to_device(out, preferred_device)

    def inverse_noise_scaling(self, sigma: Any, latent: Any) -> Any:
        preferred_device = _prefer_device(latent)
        out = self._call(
            "inverse_noise_scaling",
            _to_cpu_for_rpc(sigma),
            _to_cpu_for_rpc(latent),
        )
        return _to_device(out, preferred_device)

    def timestep(self, sigma: Any) -> Any:
        return self._call("timestep", sigma)

    def sigma(self, timestep: Any) -> Any:
        return self._call("sigma", timestep)

    def percent_to_sigma(self, percent: float) -> Any:
        return self._call("percent_to_sigma", percent)

    def set_sigmas(self, sigmas: Any) -> None:
        return self._call("set_sigmas", sigmas)
