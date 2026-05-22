from __future__ import annotations
import copy
import queue
import threading
import torch
import logging

from collections import namedtuple
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
import comfy.utils
import comfy.patcher_extension
import comfy.model_management


class MultiGPUThreadPool:
    """Persistent thread pool for multi-GPU work distribution.

    Maintains one worker thread per extra GPU device. Each thread calls
    torch.cuda.set_device() once at startup so that compiled kernel caches
    (inductor/triton) stay warm across diffusion steps.
    """

    def __init__(self, devices: list[torch.device]):
        self._workers: list[threading.Thread] = []
        self._work_queues: dict[torch.device, queue.Queue] = {}
        self._result_queues: dict[torch.device, queue.Queue] = {}

        for device in devices:
            wq = queue.Queue()
            rq = queue.Queue()
            self._work_queues[device] = wq
            self._result_queues[device] = rq
            t = threading.Thread(target=self._worker_loop, args=(device, wq, rq), daemon=True)
            t.start()
            self._workers.append(t)

    def _worker_loop(self, device: torch.device, work_q: queue.Queue, result_q: queue.Queue):
        try:
            torch.cuda.set_device(device)
        except Exception as e:
            logging.error(f"MultiGPUThreadPool: failed to set device {device}: {e}")
            while True:
                item = work_q.get()
                if item is None:
                    return
                result_q.put((None, e))
            return
        while True:
            item = work_q.get()
            if item is None:
                break
            fn, args, kwargs = item
            try:
                result = fn(*args, **kwargs)
                result_q.put((result, None))
            except Exception as e:
                result_q.put((None, e))

    def submit(self, device: torch.device, fn, *args, **kwargs):
        self._work_queues[device].put((fn, args, kwargs))

    def get_result(self, device: torch.device):
        return self._result_queues[device].get()

    @property
    def devices(self) -> list[torch.device]:
        return list(self._work_queues.keys())

    def shutdown(self):
        for wq in self._work_queues.values():
            wq.put(None)  # sentinel
        for t in self._workers:
            t.join(timeout=5.0)


class GPUOptions:
    def __init__(self, device_index: int, relative_speed: float):
        self.device_index = device_index
        self.relative_speed = relative_speed

    def clone(self):
        return GPUOptions(self.device_index, self.relative_speed)

    def create_dict(self):
        return {
            "relative_speed": self.relative_speed
        }

class GPUOptionsGroup:
    def __init__(self):
        self.options: dict[int, GPUOptions] = {}

    def add(self, info: GPUOptions):
        self.options[info.device_index] = info

    def clone(self):
        c = GPUOptionsGroup()
        for opt in self.options.values():
            c.add(opt)
        return c

    def register(self, model: ModelPatcher):
        opts_dict = {}
        # get devices that are valid for this model
        devices: list[torch.device] = [model.load_device]
        for extra_model in model.get_additional_models_with_key("multigpu"):
            extra_model: ModelPatcher
            devices.append(extra_model.load_device)
        # create dictionary with actual device mapped to its GPUOptions
        device_opts_list: list[GPUOptions] = []
        for device in devices:
            device_opts = self.options.get(device.index, GPUOptions(device_index=device.index, relative_speed=1.0))
            opts_dict[device] = device_opts.create_dict()
            device_opts_list.append(device_opts)
        # make relative_speed relative to 1.0
        min_speed = min([x.relative_speed for x in device_opts_list])
        for value in opts_dict.values():
            value['relative_speed'] /= min_speed
        model.model_options['multigpu_options'] = opts_dict


def create_multigpu_deepclones(model: ModelPatcher, max_gpus: int, gpu_options: GPUOptionsGroup=None, reuse_loaded=False):
    'Prepare ModelPatcher to contain deepclones of its BaseModel and related properties.'
    model = model.clone()
    # check if multigpu is already prepared - get the load devices from them if possible to exclude
    skip_devices = set()
    multigpu_models = model.get_additional_models_with_key("multigpu")
    if len(multigpu_models) > 0:
        for mm in multigpu_models:
            skip_devices.add(mm.load_device)
    skip_devices = list(skip_devices)

    full_extra_devices = comfy.model_management.get_all_torch_devices(exclude_current=True)
    limit_extra_devices = full_extra_devices[:max_gpus-1]
    extra_devices = limit_extra_devices.copy()
    # exclude skipped devices
    for skip in skip_devices:
        if skip in extra_devices:
            extra_devices.remove(skip)
    # create new deepclones
    if len(extra_devices) > 0:
        for device in extra_devices:
            device_patcher = None
            if reuse_loaded:
                # check if there are any ModelPatchers currently loaded that could be referenced here after a clone
                loaded_models: list[ModelPatcher] = comfy.model_management.loaded_models()
                for lm in loaded_models:
                    if lm.model is not None and lm.clone_base_uuid == model.clone_base_uuid and lm.load_device == device:
                        device_patcher = lm.clone()
                        logging.info(f"Reusing loaded deepclone of {device_patcher.model.__class__.__name__} for {device}")
                        break
            if device_patcher is None:
                device_patcher = model.deepclone_multigpu(new_load_device=device)
                device_patcher.is_multigpu_base_clone = True
            multigpu_models = model.get_additional_models_with_key("multigpu")
            multigpu_models.append(device_patcher)
            model.set_additional_models("multigpu", multigpu_models)
        model.match_multigpu_clones()
        if gpu_options is None:
            gpu_options = GPUOptionsGroup()
        gpu_options.register(model)
    else:
        logging.info("No extra torch devices need initialization, skipping initializing MultiGPU Work Units.")
    # only keep model clones that don't go 'past' the intended max_gpu count;
    # this prunes any inherited multigpu clones whose load_device is no longer allowed
    # when max_gpus is lowered between runs.
    allowed_devices = set(limit_extra_devices)
    allowed_devices.add(model.load_device)
    multigpu_models = model.get_additional_models_with_key("multigpu")
    new_multigpu_models = [m for m in multigpu_models if m.load_device in allowed_devices]
    if len(new_multigpu_models) != len(multigpu_models):
        model.set_additional_models("multigpu", new_multigpu_models)
        model.match_multigpu_clones()
    return model


def create_upscale_model_multigpu_deepclones(upscale_model, max_gpus: int):
    """Return a shallow copy of ``upscale_model`` with a ``multigpu_clones`` dict of CPU-resident
    descriptor deepclones, one per extra CUDA device up to ``max_gpus``.
    """
    full_extra_devices = comfy.model_management.get_all_torch_devices(exclude_current=True)
    limit_extra_devices = full_extra_devices[:max_gpus - 1]
    cloned = copy.copy(upscale_model)
    existing = getattr(upscale_model, 'multigpu_clones', None)
    limit_extra_device_set = set(limit_extra_devices)
    clones: dict[torch.device, object] = {d: c for d, c in dict(existing).items() if d in limit_extra_device_set} if existing else {}
    if len(limit_extra_devices) == 0:
        logging.info("No extra torch devices need initialization, skipping initializing MultiGPU upscale clones.")
        if hasattr(cloned, 'multigpu_clones'):
            del cloned.multigpu_clones
        return cloned

    for device in limit_extra_devices:
        if device in clones:
            continue
        clone_source = copy.copy(upscale_model)
        if hasattr(clone_source, 'multigpu_clones'):
            del clone_source.multigpu_clones
        clone_desc = copy.deepcopy(clone_source)
        clone_desc.model.eval()
        for p in clone_desc.model.parameters():
            p.requires_grad_(False)
        clone_desc.to("cpu")
        clones[device] = clone_desc
        logging.info(f"Created CPU upscale_model descriptor deepclone for {device}")

    cloned.multigpu_clones = clones
    return cloned


def create_vae_multigpu_deepclones(vae, max_gpus: int):
    """Return a shallow copy of ``vae`` with a ``multigpu_clones`` dict of CPU-resident VAE
    deepclones, one per extra CUDA device up to ``max_gpus``.
    """
    vae.throw_exception_if_invalid()
    vae_device = torch.device(vae.device)
    cloned = copy.copy(vae)
    if hasattr(cloned, 'multigpu_clones'):
        del cloned.multigpu_clones
    if vae_device.type == "cpu":
        logging.info("CPU VAE selected, skipping initializing MultiGPU VAE clones.")
        return cloned

    full_extra_devices = comfy.model_management.get_all_torch_devices()

    def is_vae_device(device):
        return device.type == vae_device.type and device.index == vae_device.index

    limit_extra_devices = [d for d in full_extra_devices if not is_vae_device(d)][:max_gpus - 1]
    if len(limit_extra_devices) == 0:
        logging.info("No extra torch devices need initialization, skipping initializing MultiGPU VAE clones.")
        return cloned

    existing = getattr(vae, 'multigpu_clones', None)
    limit_extra_device_set = set(limit_extra_devices)
    clones: dict[torch.device, object] = {d: c for d, c in dict(existing).items() if d in limit_extra_device_set} if existing else {}

    for device in limit_extra_devices:
        if device in clones:
            continue
        cloned_patcher = vae.patcher.deepclone_multigpu(new_load_device=device)
        clone_vae = copy.copy(vae)
        if hasattr(clone_vae, 'multigpu_clones'):
            del clone_vae.multigpu_clones
        clone_vae.first_stage_model = cloned_patcher.model
        clone_vae.patcher = cloned_patcher
        clone_vae.first_stage_model.eval()
        for p in clone_vae.first_stage_model.parameters():
            p.requires_grad_(False)
        clone_vae.first_stage_model.to("cpu")
        clones[device] = clone_vae
        logging.info(f"Created CPU VAE deepclone for {device}")

    cloned.multigpu_clones = clones
    return cloned


LoadBalance = namedtuple('LoadBalance', ['work_per_device', 'idle_time'])
def load_balance_devices(model_options: dict[str], total_work: int, return_idle_time=False, work_normalized: int=None):
    'Optimize work assigned to different devices, accounting for their relative speeds and splittable work.'
    opts_dict = model_options['multigpu_options']
    devices = list(model_options['multigpu_clones'].keys())
    speed_per_device = []
    work_per_device = []
    # get sum of each device's relative_speed
    total_speed = 0.0
    for opts in opts_dict.values():
        total_speed += opts['relative_speed']
    # get relative work for each device;
    # obtained by w = (W*r)/R
    for device in devices:
        relative_speed = opts_dict[device]['relative_speed']
        relative_work = (total_work*relative_speed) / total_speed
        speed_per_device.append(relative_speed)
        work_per_device.append(relative_work)
    # relative work must be expressed in whole numbers, but likely is a decimal;
    # perform rounding while maintaining total sum equal to total work (sum of relative works)
    work_per_device = round_preserved(work_per_device)
    dict_work_per_device = {}
    for device, relative_work in zip(devices, work_per_device):
        dict_work_per_device[device] = relative_work
    if not return_idle_time:
        return LoadBalance(dict_work_per_device, None)
    # divide relative work by relative speed to get estimated completion time of said work by each device;
    # time here is relative and does not correspond to real-world units
    completion_time = [w/r for w,r in zip(work_per_device, speed_per_device)]
    # calculate relative time spent by the devices waiting on each other after their work is completed
    idle_time = abs(min(completion_time) - max(completion_time))
    # if need to compare work idle time, need to normalize to a common total work
    if work_normalized:
        idle_time *= (work_normalized/total_work)

    return LoadBalance(dict_work_per_device, idle_time)

def round_preserved(values: list[float]):
    'Round all values in a list, preserving the combined sum of values.'
    # get floor of values; casting to int does it too
    floored = [int(x) for x in values]
    total_floored = sum(floored)
    # get remainder to distribute
    remainder = round(sum(values)) - total_floored
    # pair values with fractional portions
    fractional = [(i, x-floored[i]) for i, x in enumerate(values)]
    # sort by fractional part in descending order
    fractional.sort(key=lambda x: x[1], reverse=True)
    # distribute the remainder
    for i in range(remainder):
        index = fractional[i][0]
        floored[index] += 1
    return floored
