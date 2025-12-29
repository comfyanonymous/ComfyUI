import importlib.resources
import json
import logging
import time
from importlib.abc import Traversable
from typing import Any, AsyncGenerator
import threading
import psutil

try:
    import pynvml
except ImportError:
    pynvml = None

import pytest

from comfy.api.components.schema.prompt import Prompt
from comfy.client.embedded_comfy_client import Comfy
from comfy.distributed.process_pool_executor import ProcessPoolExecutor
from comfy.model_downloader import add_known_models, KNOWN_LORAS
from comfy.model_downloader_types import CivitFile, HuggingFile
from comfy_extras.nodes.nodes_audio import TorchAudioNotFoundError
from . import workflows
import itertools
from comfy.cli_args import default_configuration
from comfy.cli_args_types import PerformanceFeature

logger = logging.getLogger(__name__)


class ResourceMonitor:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak_cpu_ram = 0
        self.peak_gpu_vram = 0
        self._stop_event = threading.Event()
        self._thread = None
        self._pynvml_available = False
        self._gpu_handles = []

    def _monitor(self):
        current_process = psutil.Process()
        while not self._stop_event.is_set():
            # Monitor CPU RAM (RSS) for process tree
            try:
                children = current_process.children(recursive=True)
                processes = [current_process] + children
                pids = {p.pid for p in processes}
                
                total_rss = 0
                for p in processes:
                    try:
                        total_rss += p.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.peak_cpu_ram = max(self.peak_cpu_ram, total_rss)

                # Monitor GPU VRAM if available
                if self._pynvml_available and self._gpu_handles:
                    total_vram = 0
                    try:
                        # Iterate over all detected GPUs
                        for handle in self._gpu_handles:
                            # Get all processes running on the GPU
                            try:
                                compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                                graphics_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
                                
                                # Filter for our process tree
                                for p in compute_procs + graphics_procs:
                                    if p.pid in pids:
                                        total_vram += p.usedGpuMemory
                            except Exception:
                                pass # Skip errors for specific GPU queries
                        
                        self.peak_gpu_vram = max(self.peak_gpu_vram, total_vram)
                    except Exception:
                        pass
            except Exception:
                pass
            
            time.sleep(self.interval)

    def __enter__(self):
        if pynvml:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                self._gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
                self._pynvml_available = True
            except Exception as e:
                logger.warning(f"Failed to initialize pynvml for VRAM monitoring: {e}")

        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        if self._pynvml_available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _generate_config_params():
    attn_keys = [
        "use_pytorch_cross_attention",
        # "use_split_cross_attention",
        # "use_quad_cross_attention",
        # "use_sage_attention",
        # "use_flash_attention"
    ]
    attn_options = [
        {k: (k == target_key) for k in attn_keys}
        for target_key in attn_keys
    ]

    async_options = [
        {"disable_async_offload": False},
        {"disable_async_offload": True},
    ]
    pinned_options = [
        {"disable_pinned_memory": False},
        {"disable_pinned_memory": True},
    ]
    fast_options = [
        {"fast": set()},
        # {"fast": {PerformanceFeature.Fp16Accumulation}},
        # {"fast": {PerformanceFeature.Fp8MatrixMultiplication}},
        {"fast": {PerformanceFeature.CublasOps}},
    ]

    for attn, asnc, pinned, fst in itertools.product(attn_options, async_options, pinned_options, fast_options):
        config_update = {}
        config_update.update(attn)
        config_update.update(asnc)
        config_update.update(pinned)
        config_update.update(fst)
        yield config_update


@pytest.fixture(scope="function", autouse=False, params=_generate_config_params(), ids=lambda p: ",".join(f"{k}={v}" for k, v in p.items()))
async def client(tmp_path_factory, request) -> AsyncGenerator[Any, Any]:
    config = default_configuration()
    # this should help things go a little faster
    config.disable_all_custom_nodes = True
    config.update(request.param)
    # use ProcessPoolExecutor to respect various config settings
    with ProcessPoolExecutor(max_workers=1) as executor:
        async with Comfy(configuration=config, executor=executor) as client:
            yield client


def _prepare_for_workflows() -> dict[str, Traversable]:

    add_known_models("loras", HuggingFile("artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5", "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors"))
    add_known_models("checkpoints", HuggingFile("autismanon/modeldump", "cardosAnime_v20.safetensors"))

    return {f.name: f for f in importlib.resources.files(workflows).iterdir() if f.is_file() and f.name.endswith(".json")}


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow_name, workflow_file", _prepare_for_workflows().items())
async def test_workflow(workflow_name: str, workflow_file: Traversable, has_gpu: bool, client: Comfy):
    if not has_gpu:
        pytest.skip("requires gpu")

    if "compile" in workflow_name:
        pytest.skip("compilation has regressed in 0.4.0 and later because upcast weights are now permitted to be compiled, causing OOM errors in most cases")
        return

    workflow = json.loads(workflow_file.read_text(encoding="utf8"))

    prompt = Prompt.validate(workflow)
    # todo: add all the models we want to test a bit m2ore elegantly
    outputs = {}
    
    start_time = time.time()
    monitor = ResourceMonitor()
    try:
        with monitor:
            outputs = await client.queue_prompt(prompt)
    except TorchAudioNotFoundError:
        pytest.skip("requires torchaudio")
    finally:
        end_time = time.time()
        duration = end_time - start_time
        ram_gb = monitor.peak_cpu_ram / (1024**3)
        vram_gb = monitor.peak_gpu_vram / (1024**3)
        logger.info(f"Test {workflow_name} with client {client} took {duration:.4f}s | Peak RAM: {ram_gb:.2f} GB | Peak VRAM: {vram_gb:.2f} GB")

    if any(v.class_type == "SaveImage" for v in prompt.values()):
        save_image_node_id = next(key for key in prompt if prompt[key].class_type == "SaveImage")
        assert outputs[save_image_node_id]["images"][0]["abs_path"] is not None
    elif any(v.class_type == "SaveAudio" for v in prompt.values()):
        save_audio_node_id = next(key for key in prompt if prompt[key].class_type == "SaveAudio")
        assert outputs[save_audio_node_id]["audio"][0]["filename"] is not None
    elif any(v.class_type == "SaveAnimatedWEBP" for v in prompt.values()):
        save_video_node_id = next(key for key in prompt if prompt[key].class_type == "SaveAnimatedWEBP")
        assert outputs[save_video_node_id]["images"][0]["filename"] is not None
    elif any(v.class_type == "PreviewString" for v in prompt.values()):
        save_image_node_id = next(key for key in prompt if prompt[key].class_type == "PreviewString")
        output_str = outputs[save_image_node_id]["string"][0]
        assert output_str is not None
        assert len(output_str) > 0
    else:
        assert len(outputs) > 0
        logger.warning(f"test {workflow_name} did not have a node that could be checked for output")
