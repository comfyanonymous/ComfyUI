import os
import math
import time
import torch
import asyncio
import threading
import torch.nn as nn
from pathlib import Path
import concurrent.futures
from einops import rearrange
import torch.nn.functional as F
from collections import OrderedDict
from safetensors import safe_open
from transformers.cache_utils import StaticCache
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Any, List, Dict
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.modules.diffusionmodules.openaimodel import ResBlock

INIT_MOE = torch.cuda.device_count() != 1
MOE_LAYER_SIZE = (1024**3) * 5.15 # approx

class HunyuanStaticCache(StaticCache):

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        cache_position = cache_kwargs.get("cache_position")
        if hasattr(self, "key_cache") and hasattr(self, "value_cache"):
            if self.key_cache[layer_idx].device != key_states.device:
                self.key_cache[layer_idx] = self.key_cache[layer_idx].to(key_states.device)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].to(value_states.device)
            k_out = self.key_cache[layer_idx]
            v_out = self.value_cache[layer_idx]
            key_states = key_states.to(k_out.dtype)
            value_states = value_states.to(v_out.dtype)
        else:
            if self.layers[layer_idx].keys is None:
                self.layers[layer_idx].lazy_initialization(key_states)
            k_out = self.layers[layer_idx].keys
            v_out = self.layers[layer_idx].values

        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            if cache_position.dim() == 1:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)

            else:
                assert cache_position.dim() == 2, f"multiple batch dims not yet {cache_position.shape=}"
                batch_size, _ = cache_position.shape
                for i in range(batch_size):
                    unbatched_dim = 1
                    k_out[i].index_copy_(unbatched_dim, cache_position[i], key_states[i])
                    v_out[i].index_copy_(unbatched_dim, cache_position[i], value_states[i])

        return k_out, v_out

def real_batched_index_select(t, dim, idx):
    return torch.stack([torch.index_select(t[i], dim - 1, idx[i]) for i in range(len(t))])

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding

class TimestepEmbedder(nn.Module):
    def __init__(self,
                 hidden_size,
                 act_layer=nn.GELU,
                 frequency_embedding_size=256,
                 max_period=10000,
                 out_size=None,
                 dtype=None,
                 device=None
                 ):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, **factory_kwargs),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb
    

def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    return x


def get_meshgrid_nd(start, *args, dim=2):

    start = _to_tuple(start, dim=dim)
    stop = _to_tuple(args[0], dim=dim)
    num = [stop[i] - start[i] for i in range(dim)]
    num_int = [int(x) for x in num]
    num = num_int

    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")
    grid = torch.stack(grid, dim=0)

    return grid

def build_2d_rope(
        seq_len: int, n_elem: int, image_infos: Optional[List[Tuple[slice, Tuple[int, int]]]] = None,
        device: Optional[torch.device] = None, base: int = 10000, base_rescale_factor: float = 1.0,
        return_all_pos: bool = False,
):

    assert n_elem % 4 == 0, f"n_elem must be divisible by 4, but got {n_elem}."

    # theta
    if base_rescale_factor != 1.0:
        base *= base_rescale_factor ** (n_elem / (n_elem - 2))
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    theta = theta.reshape(1, n_elem // 4, 2)    # [1, half_d, 2]

    # position indices
    if image_infos is None:
        image_infos = []

    image_infos_list = [image_infos]
    sample_seq_lens = [seq_len]

    # Prepare position indices for each sample
    x_sections = []
    y_sections = []
    for sample_id, sample_image_infos in enumerate(image_infos_list):
        last_pos = 0
        for sec_slice, (h, w) in sample_image_infos:
            L = sec_slice.start   # start from 0, so image_slice.start is just L
            # previous text
            if last_pos < L:
                y_sections.append(torch.arange(last_pos, L))
                x_sections.append(torch.arange(last_pos, L))
            elif h is None:
                # Interleave data has overlapped positions for <boi> <size> <ratio> <timestep> <eoi> tokens.
                y_sections.append(torch.arange(sec_slice.start, sec_slice.stop))
                x_sections.append(torch.arange(sec_slice.start, sec_slice.stop))
                continue
            else:
                # Interleave data has overlapped positions for noised image and the successive clean image,
                # leading to last_pos (= last text end L + noise w * h) > L (last text end L).
                pass
            # current image
            beta_y = L + (w * h - h) / 2
            beta_x = L + (w * h - w) / 2
            grid = get_meshgrid_nd((beta_y, beta_x), (beta_y + h, beta_x + w))  # [2, h, w]
            grid = grid.reshape(2, -1)  # (y, x)
            y_sections.append(grid[0])
            x_sections.append(grid[1])
            # step
            last_pos = L + w * h
        # final text
        y_sections.append(torch.arange(last_pos, sample_seq_lens[sample_id]))
        x_sections.append(torch.arange(last_pos, sample_seq_lens[sample_id]))

    x_pos = torch.cat(x_sections).long()
    y_pos = torch.cat(y_sections).long()
    # If there are overlap positions, we need to remove them.
    x_pos = x_pos[:seq_len]
    y_pos = y_pos[:seq_len]
    all_pos = torch.stack((y_pos, x_pos), dim=1).unsqueeze(1).to(device)    # [seq_len, 1, 2]

    # calc rope
    idx_theta = (all_pos * theta).reshape(all_pos.shape[0], n_elem // 2).repeat(1, 2)

    cos = torch.cos(idx_theta)
    sin = torch.sin(idx_theta)

    if return_all_pos:
        return cos, sin, all_pos

    return cos, sin


def build_batch_2d_rope(
        seq_len: int, n_elem: int, image_infos: Optional[List[List[Tuple[slice, Tuple[int, int]]]]] = None,
        device: Optional[torch.device] = None, base: int = 10000, base_rescale_factor: float = 1.0,
        return_all_pos: bool = False,
):
    cos_list, sin_list, all_pos_list = [], [], []
    if image_infos is None:
        image_infos = [None]
    for i, image_info in enumerate(image_infos):
        res = build_2d_rope(
            seq_len, n_elem, image_infos=image_info, device=device,
            base=base, base_rescale_factor=base_rescale_factor,
            return_all_pos=return_all_pos,
        )
        if return_all_pos:
            cos, sin, all_pos = res
        else:
            cos, sin = res
            all_pos = None
        cos_list.append(cos)
        sin_list.append(sin)
        all_pos_list.append(all_pos)

    stacked_cos = torch.stack(cos_list, dim=0)
    stacked_sin = torch.stack(sin_list, dim=0)

    if return_all_pos:
        return stacked_cos, stacked_sin, all_pos_list

    return stacked_cos, stacked_sin


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def default(val, d):
    return val if val is not None else d

def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)

def normalization(channels, **kwargs):
    return nn.GroupNorm(32, channels, **kwargs)

def topkgating(logits: torch.Tensor, topk: int):
    logits = logits.float()
    gates = F.softmax(logits, dim=1)

    num_experts = int(gates.shape[1])

    _, expert_index = torch.topk(gates, topk)
    expert_mask = F.one_hot(expert_index, num_experts)

    expert_index_flat = expert_index.flatten()
    tokens_per_expert = torch.bincount(expert_index_flat, minlength=num_experts)
    expert_capacity = torch.max(tokens_per_expert).item()

    gates_s = torch.clamp(
        torch.matmul(expert_mask.float(), gates.unsqueeze(-1)).sum(dim=1), min=torch.finfo(gates.dtype).eps
    )
    router_probs = gates / gates_s

    expert_index = torch.transpose(expert_index, 0, 1)
    expert_index = expert_index.reshape(-1)
    expert_mask = F.one_hot(expert_index, num_experts).to(torch.int32)

    token_priority = torch.cumsum(expert_mask, dim=0) * expert_mask - 1
    token_priority = token_priority.reshape((topk, -1, num_experts))
    token_priority = torch.transpose(token_priority, 0, 1)

    token_priority = torch.max(token_priority, dim=1)[0]

    valid_mask = torch.logical_and(token_priority >= 0, token_priority < expert_capacity)
    token_priority = torch.masked_fill(token_priority, ~valid_mask, 0)
    dispatch_mask = F.one_hot(token_priority, expert_capacity).to(torch.bool)
    valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, expert_capacity)
    dispatch_mask = torch.masked_fill(dispatch_mask, ~valid_mask, 0)

    combine_weights = torch.einsum("...te,...tec->...tec", router_probs, dispatch_mask)

    return combine_weights, dispatch_mask

class HunyuanRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class UNetDown(nn.Module):
    def __init__(self, patch_size, in_channels, emb_channels, hidden_channels, out_channels,
                 dropout=0.0, device=None, dtype=None):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]

        self.model = nn.ModuleList(
            [conv_nd(
                2,
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                **factory_kwargs
            )]
        )

        if self.patch_size == 1:
            self.model.append(ResBlock(
                channels=hidden_channels,
                emb_channels=emb_channels,
                out_channels=out_channels,
                use_scale_shift_norm = True,
                dropout=dropout,
                **factory_kwargs
            ))
        else:
            for i in range(self.patch_size // 2):
                self.model.append(ResBlock(
                    channels=hidden_channels,
                    emb_channels=emb_channels,
                    use_scale_shift_norm = True,
                    out_channels=hidden_channels if (i + 1) * 2 != self.patch_size else out_channels,
                    dropout=dropout,
                    down=True,
                    **factory_kwargs
                ))

    def forward(self, x, t):
        assert x.shape[2] % self.patch_size == 0 and x.shape[3] % self.patch_size == 0
        for module in self.model:
            if isinstance(module, ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        _, _, token_h, token_w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x, token_h, token_w


class UNetUp(nn.Module):

    def __init__(self, patch_size, in_channels, emb_channels, hidden_channels, out_channels,
                 dropout=0.0, device=None, dtype=None, operations = None, out_norm=False):
        operations = operations or nn
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]

        self.model = nn.ModuleList()

        if self.patch_size == 1:
            self.model.append(ResBlock(
                channels=in_channels,
                emb_channels=emb_channels,
                out_channels=hidden_channels,
                use_scale_shift_norm = True,
                dropout=dropout,
                **factory_kwargs
            ))
        else:
            for i in range(self.patch_size // 2):
                self.model.append(ResBlock(
                    channels=in_channels if i == 0 else hidden_channels,
                    emb_channels=emb_channels,
                    out_channels=hidden_channels,
                    use_scale_shift_norm = True,
                    dropout=dropout,
                    up=True,
                    **factory_kwargs
                ))

        if out_norm:
            self.model.append(nn.Sequential(
                normalization(hidden_channels, **factory_kwargs),
                nn.SiLU(),
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    **factory_kwargs
                ),
            ))
        else:
            self.model.append(nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                **factory_kwargs
            ))

    # batch_size, seq_len, model_dim
    def forward(self, x, t, token_h, token_w):
        x = rearrange(x, 'b (h w) c -> b c h w', h=token_h, w=token_w)
        for module in self.model:
            if isinstance(module, ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        return x
    
class HunyuanTopKGate(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.moe_topk = 8
        self.min_capacity = 8
        num_experts = 64
        self.wg = nn.Linear(config["hidden_size"], num_experts, bias=False, dtype=torch.float32)

    def forward(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)
        if self.wg.weight.dtype == torch.float32:
            hidden_states = hidden_states.float()
        logits = self.wg(hidden_states)
        gate_output = topkgating(logits, self.moe_topk)

        return gate_output
    
class HunyuanMLP(nn.Module):
    def __init__(self, config, layer_idx=None, is_shared_mlp=False, is_moe=False, device=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config["hidden_size"]

        self.intermediate_size = 3072

        self.act_fn = torch.nn.functional.silu
        self.intermediate_size *= 2  # SwiGLU
        self.gate_and_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, device=device)
        self.down_proj = nn.Linear(self.intermediate_size // 2, self.hidden_size, bias=False, device=device)
    def forward(self, x):
        self.gate_and_up_proj, self.down_proj = self.gate_and_up_proj.to(x.device), self.down_proj.to(x.device)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        gate_and_up_proj = self.gate_and_up_proj(x)
        x1, x2 = gate_and_up_proj.chunk(2, dim=2)
        down_proj = self.down_proj(x1 * self.act_fn(x2))
        return down_proj

class MoELRUCache(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpu_cache = OrderedDict()
        self.cpu_cache = OrderedDict()
        self.offload_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

        self.last_offload_event = None
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()

    async def _async_offload_to_cpu(self, layer_idx):
        # async offload from gpu (removed)

        num_experts = 64
        moe_group = [(layer_idx * num_experts + i, self.gpu_cache[layer_idx * num_experts + i])
                    for i in range(num_experts)
                    if (layer_idx * num_experts + i) in self.gpu_cache]
        event = torch.cuda.Event()
    
        with torch.cuda.stream(self.offload_stream):
            for index, moe in moe_group:
                moe_cpu = HunyuanMLP(moe.config).to("cpu", non_blocking=True)
                for (name, p_gpu), p_cpu in zip(moe.named_parameters(), moe_cpu.parameters()):
                    if p_gpu.device.type == "meta":
                        continue
                    with torch.no_grad():
                        p_cpu.data = torch.empty_like(p_gpu, device="cpu", pin_memory=True)
                        p_cpu.copy_(p_gpu, non_blocking=True)

                self.cpu_cache[index] = moe_cpu

            self.offload_stream.record_event(event)

        self.last_offload_event = event

        def finalize_offload_layer():
            event.synchronize()
            for index, moe in moe_group:
                moe.to("meta")
                self.gpu_cache.pop(index, None)
                del moe
            torch.cuda.empty_cache()

        threading.Thread(target=finalize_offload_layer, daemon=True).start()

    async def _async_load_to_gpu(self, index, moe):

        # if enough memory load, otherwise wait for offload
        while True:
            free_bytes, _ = torch.cuda.mem_get_info()
            if free_bytes > 2 * MOE_LAYER_SIZE:
                break

            self.last_offload_event.synchronize()
            torch.cuda.empty_cache()
            await asyncio.sleep(0.01)

        # async loading from cpu -> gpu
        with torch.cuda.stream(self.load_stream):
            moe_gpu = HunyuanMLP(moe.config).to("cuda", non_blocking=True)
            for (name, p_cpu), p_gpu in zip(moe.named_parameters(), moe_gpu.parameters()):
                with torch.no_grad():
                    p_gpu.data = torch.empty_like(p_cpu, device="cuda")
                    p_gpu.copy_(p_cpu, non_blocking=True)

        def finalize_load():
            self.gpu_cache[index] = moe_gpu
            self.cpu_cache.pop(index, None)

        threading.Thread(target=finalize_load, daemon=True).start()

    def add_cpu(self, moe, index):
        moe_cpu = moe.to("cpu")

        for _, p in moe_cpu.named_parameters():
            if not p.is_pinned():
                if p.device.type == "cpu":
                    p.data = p.data.pin_memory()
                else:
                    return

        self.cpu_cache[index] = moe_cpu
        self.cpu_cache.move_to_end(index)

def parse_layer_expert(key):
    parts = key.split(".")
    layer = int(parts[2])
    expert = int(parts[5])
    return layer, expert
        
class LazyMoELoader(nn.Module):
    def __init__(self, cache, config, max_workers = 16, max_concurrent_loads = 32):
        super().__init__()
        self.cache = cache
        self.config = config
        self._loop = cache._loop
        self.expert_key_index = self.index_safetensors()
        self._checkpoint = self.get_checkpoint()
        self._file = safe_open(self._checkpoint, framework="pt", device="cpu", mmap=True)
        self.expert_pool = self.build_meta_experts()

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._semaphore = asyncio.Semaphore(max_concurrent_loads)

    def build_meta_experts(self):
        pool = {}
        for layer, experts in self.expert_key_index.items():
            pool[layer] = {}
            for expert in experts:
                pool[layer][expert] = HunyuanMLP(
                    self.config,
                    layer_idx=layer,
                    device="meta",
                )
        return pool

    def get_checkpoint(self):
        comfyui_dir = Path.home() / "ComfyUI"
        checkpoint = comfyui_dir / "models" / "checkpoint" / "hunyuan_image_3.safetensors"
        checkpoint = checkpoint.resolve()
        if not os.path.exists(checkpoint):
            raise ValueError(f"Hunyuan Image 3 Checkpoint on one GPU should have the path: {checkpoint}")
        return checkpoint
    
    def index_safetensors(self):
        checkpoint = self.get_checkpoint()
        index = {}
        with safe_open(checkpoint, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "experts." in k:
                    layer, expert = parse_layer_expert(k)
                    index.setdefault(layer, {}).setdefault(expert, []).append(k)
        return index

    def lazy_init(self, layer_idx, expert_idx):
        keys = self.expert_key_index[layer_idx][expert_idx]
        model = self.expert_pool[layer_idx][expert_idx]

        def strip_expert_prefix(k):
            return k.split(f"experts.{expert_idx}.", 1)[1]

        sd = { strip_expert_prefix(k): self._file.get_tensor(k) for k in keys }

        for name, tensor in sd.items():
            getattr(model, name).data = tensor
        return model

    def _register_expert_sync(self, layer_idx, expert_idx, moe_cpu):
        self.cache.add_cpu(moe_cpu, (layer_idx * 64) + expert_idx)
        asyncio.run_coroutine_threadsafe(
            self.cache._async_load_to_gpu((layer_idx * 64) + expert_idx, moe_cpu),
            self.cache._loop
        )

    async def lazy_load_from_disk(self, layer_idx, expert_idx):
        loop = asyncio.get_event_loop()
        async with self._semaphore:
            moe_cpu = await loop.run_in_executor(None, self.lazy_init, layer_idx, expert_idx)
        self._loop.call_soon_threadsafe(self._register_expert_sync, layer_idx, expert_idx, moe_cpu)
        return moe_cpu

    async def schedule_layer_load_progressive(self, layer_idx, num_experts = 64):
        tasks = [asyncio.create_task(self.lazy_load_from_disk(layer_idx, i)) for i in range(num_experts)]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results

    def schedule_layer_load(self, layer_idx, num_experts = 64):
        fut = asyncio.run_coroutine_threadsafe(
            self.schedule_layer_load_progressive(layer_idx, num_experts),
            self._loop
        )
        return fut

    async def schedule_layer_load_(self, layer_idx):
        tasks = [self.lazy_load_from_disk(layer_idx, i) for i in range(64)]
        experts = await asyncio.gather(*tasks)
        return experts
    
def enough_vram(required_bytes):
    free, total = torch.cuda.mem_get_info()
    return free > required_bytes
    
class HunyuanMoE(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None, moe_lru=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.moe_topk = 8
        self.num_experts = 64
        self.shared_mlp = HunyuanMLP(config, layer_idx=layer_idx, is_shared_mlp=True)
        self.gate = HunyuanTopKGate(config, layer_idx=layer_idx)
        if INIT_MOE:
            self.experts = nn.ModuleList(
                [HunyuanMLP(config, layer_idx=layer_idx, is_shared_mlp=False, is_moe=True) for _ in range(self.num_experts)]
            )
        else:
            self.experts = []
            self.moe_lru = moe_lru

    def forward(self, hidden_states):
        if not INIT_MOE:
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(hidden_states.device.index)
        bsz, seq_len, hidden_size = hidden_states.shape

        hidden_states_mlp = self.shared_mlp(hidden_states)

        reshaped_input = hidden_states.reshape(-1, hidden_size)

        with torch.cuda.nvtx.range("MoE"):
            combine_weights, dispatch_mask = self.gate(hidden_states)

            dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.type_as(reshaped_input), reshaped_input)
            device = hidden_states.device
            dtype = reshaped_input.dtype
                
            used_mask = (dispatch_mask.sum(dim=(0, 2)) > 0)
            used_indices = used_mask.nonzero(as_tuple=False).squeeze(1).tolist()
        
            combined_output = torch.zeros_like(reshaped_input, device=device, dtype=dtype)
        
            if len(used_indices) == 0:
                pass
            else:
                tokens_padded = dispatched_input[used_indices]

                l1, l2 = [], []
                for i in used_indices:
                    expert = self.experts[i]
                    if isinstance(expert, (asyncio.Future, concurrent.futures.Future)):
                        expert = expert.result()
                    expert = expert.to(device)
                    l1.append(expert.gate_and_up_proj)
                    l2.append(expert.down_proj)

                compute_device = hidden_states.device

                l1 = [m.to(compute_device) for m in l1]
                W1 = torch.stack([m.weight for m in l1], dim=0)
                del l1
                W1_T = W1.transpose(1, 2)

                del W1
                x = torch.bmm(tokens_padded, W1_T)
                del W1_T, tokens_padded

                x1, x2 = x.chunk(2, dim=2)
                gated = x1 * F.silu(x2)

                l2 = [m.to(compute_device) for m in l2]
                W2 = torch.stack([m.weight for m in l2], dim=0)
                del l2
                W2_T = W2.transpose(1, 2)
                del W2
                out_padded = torch.bmm(gated, W2_T)
                del W2_T
        
                while not enough_vram(3*(1024 ** 3)):
                    event = self.moe_lru.last_offload_event
                    if event is not None and not event.query():
                        time.sleep(0.001)
        
        
                combine_weights_used = combine_weights[:, used_indices, :]
        
                combined_output = torch.einsum("suc,ucm->sm",
                    combine_weights_used.type_as(out_padded),
                    out_padded
                )
        
                del x, x1, x2, gated, out_padded

        combined_output = combined_output.reshape(bsz, seq_len, hidden_size)

        output = hidden_states_mlp + combined_output

        return output

class HunyuanImage3Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = 'self'

        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config["attention_head_dim"]
        self.num_key_value_heads = 8
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config["max_position_embeddings"]
        self.rope_theta = 10000.0
        self.is_causal = True
        self.hidden_size_q = self.head_dim * self.num_heads
        self.hidden_size_kv = self.head_dim * self.num_key_value_heads

        # define layers
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size_q + 2 * self.hidden_size_kv,
            bias=False
        )
        self.o_proj = nn.Linear(self.hidden_size_q, self.hidden_size, bias=False)

        self.query_layernorm = HunyuanRMSNorm(self.head_dim, eps=config["rms_norm_eps"])
        self.key_layernorm = HunyuanRMSNorm(self.head_dim, eps=config["rms_norm_eps"])

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            past_key_value,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
            **kwargs,
    ):

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.reshape(bsz, q_len, self.num_key_value_heads, self.num_key_value_groups + 2,
                                        self.head_dim)
        query_states, key_states, value_states = torch.split(qkv_states, [self.num_key_value_groups, 1, 1], dim=3)

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = custom_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            query_states = query_states.to(key_states.dtype)

        key_states = torch.repeat_interleave(key_states, dim=1, repeats = self.num_key_value_groups)
        value_states = torch.repeat_interleave(value_states, dim=1, repeats = self.num_key_value_groups)

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = optimized_attention(query_states, key_states, value_states, self.num_heads, mask = attention_mask, skip_reshape=True)

        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

class HunyuanImage3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, moe_lru=None):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.layer_idx = layer_idx
        self.self_attn = HunyuanImage3Attention(config, layer_idx=layer_idx)
        
        self.mlp = HunyuanMoE(config, layer_idx=layer_idx, moe_lru=moe_lru)

        self.input_layernorm = HunyuanRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_attention_layernorm = HunyuanRMSNorm(config["hidden_size"], eps=config['rms_norm_eps'])

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            use_cache: Optional[bool] = False,
            custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor | Any]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            custom_pos_emb=custom_pos_emb,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states, past_key_value)

        return outputs
    
class HunyuanImage3Model(nn.Module):
    def __init__(self, config, moe_lru=None):
        super().__init__()
        self.padding_idx = 128009
        self.vocab_size = 133120
        self.config = config
        self.layers = nn.ModuleList(
            [HunyuanImage3DecoderLayer(config, layer_idx, moe_lru = moe_lru) for layer_idx in range(config["num_hidden_layers"])]
        )
        
        self.ln_f = HunyuanRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

        self.shared_tensor = None
        self.moe_lru = moe_lru
        self.additional_layers_set = False
        self.moe_loader = LazyMoELoader(self.moe_lru, self.config)

    def forward(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache = True,
            custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
            mode: str = "gen_image",
            first_step: Optional[bool] = None,
            gen_timestep_scatter_index: Optional[torch.Tensor] = None,
    ):

        hidden_states = inputs_embeds

        next_decoder_cache = None
        next_layers = 0
        additional_layers = torch.cuda.mem_get_info()[0] // (MOE_LAYER_SIZE * 2)
        sparse_interval = max(1, len(self.layers) // additional_layers)

        if len(self.layers[0].mlp.experts) == 0:
            self.layers[0].mlp.experts = self.moe_loader.schedule_layer_load(0)

        for layer_idx, decoder_layer in enumerate(self.layers):
            
            if layer_idx + 1 < len(self.layers) and len(self.layers[layer_idx + 1].mlp.experts) == 0: # not loaded
                self.layers[layer_idx+1].mlp.experts = self.moe_loader.schedule_layer_load(layer_idx + 1)

            if layer_idx + 2 < len(self.layers) and len(self.layers[layer_idx + 2].mlp.experts) == 0: # load first and second layers
                self.layers[layer_idx+2].mlp.experts = self.moe_loader.schedule_layer_load(layer_idx + 2)
            
            if not self.additional_layers_set:
                if (layer_idx % sparse_interval == 0) and layer_idx >= sparse_interval:
                    self.layers[next_layers].mlp.experts = self.moe_loader.schedule_layer_load(next_layers)
                    next_layers += 1

            with torch.no_grad():
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    custom_pos_emb=custom_pos_emb,
                    mode=mode,
                    first_step=first_step,
                    gen_timestep_scatter_index=gen_timestep_scatter_index,
                )

            if layer_idx >= 0:
                if self.additional_layers_set and layer_idx <= self.additional_layers_set:
                    pass
                else:
                    asyncio.run_coroutine_threadsafe(
                        self.moe_lru._async_offload_to_cpu(layer_idx),
                        self.moe_lru._loop
                    )
                    self.layers[layer_idx].mlp.experts = []

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[1]

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache
        self.additional_layers_set = True
        return tuple(v for v in [hidden_states, next_cache] if v is not None)


class HunyuanImage3ForCausalMM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.timestep_emb = TimestepEmbedder(hidden_size=config["hidden_size"])
        self.patch_embed = UNetDown(
            patch_size=1,
            emb_channels=config["hidden_size"],
            in_channels=32,
            hidden_channels=1024,
            out_channels=config["hidden_size"],
        )
        self.time_embed = TimestepEmbedder(hidden_size=config["hidden_size"])

        self.final_layer = UNetUp(
            patch_size=1,
            emb_channels=config["hidden_size"],
            in_channels=config["hidden_size"],
            hidden_channels=1024,
            out_channels=32,
            out_norm=True,
        )
        self.time_embed_2 = TimestepEmbedder(hidden_size=config["hidden_size"])

        self.moe_lru = None
        if not INIT_MOE:
            self.moe_lru = MoELRUCache()

        self.model = HunyuanImage3Model(config, moe_lru=self.moe_lru)

        self.pad_id = 128009
        self.vocab_size = 133120

        self.lm_head = nn.Linear(config["hidden_size"], 133120, bias=False)
        self.first_step = True

        self.kv_cache = None
        self.token_dims = ()

    @staticmethod
    def get_pos_emb(custom_pos_emb, position_ids):
        cos, sin = custom_pos_emb
        cos = real_batched_index_select(cos, dim=1, idx=position_ids)
        sin = real_batched_index_select(sin, dim=1, idx=position_ids)
        return cos, sin

    def ragged_final_layer(self, x, image_mask, timestep, token_h, token_w, first_step):
        bsz, seq_len, n_embd = x.shape
        if first_step:
            image_output = x.masked_select(image_mask.unsqueeze(-1).bool()).reshape(bsz, -1, n_embd)
        else:
            image_output = x[:, 1:, :]
        timestep_emb = self.time_embed_2(timestep)
        pred = self.final_layer(image_output, timestep_emb, token_h, token_w)
        return pred

    def forward(self, x, condition, timestep, **kwargs):
        
        joint_image, cond_vae_image_mask, inputs_embeds, uncond_joint, uncond_vae_mask, uncond_inputs = condition.unbind()

        cond_exists = (joint_image[:, 0, :] != -100.0).any(dim=1).any()

        gen_timestep_scatter_index = 4
        
        if cond_exists:
            with torch.no_grad():
                joint_image[:, 2:3, :] = x[:, 2:3, :] # updates image ratio 

        if self.first_step:
            token_height, token_width = x[:, -2:, 0].tolist()[0]
            self.token_dims = (int(token_height), int(token_width))
            x = x[:, :-2, :]
        else:
            token_height, token_width = self.token_dims
        
        img_slices = []
        bsz, seq_len, n_embd = inputs_embeds.shape
        
        for i in range(x.size(0)):
            gen_offset = seq_len + x.size(1)
            if cond_exists:
                vae_mask_indices = (cond_vae_image_mask[i].squeeze(-1) == 1).nonzero(as_tuple=True)[0]
                vae_start, vae_end = vae_mask_indices[0].item(), vae_mask_indices[-1].item() + 1
                vae_start += gen_offset
                vae_end += gen_offset

                vit_start = vae_end + 1
                vit_end = joint_image.size(1) - 1 + gen_offset

                joint_slices_i = [
                    slice(vae_start, vae_end),
                    slice(vit_start, vit_end),
                ]
            else:
                joint_slices_i = []
            gen_slices_i = [slice(seq_len, gen_offset)]
            img_slices.append(gen_slices_i + joint_slices_i)

        img_s = img_slices[0]
        rope_img = [(img_s[0], (token_height, token_width))]
        rope_image_info = [rope_img if len(joint_slices_i) == 0 else rope_img + [(img_s[1], (384 // 16, 384 // 16)), (img_s[2], (256 // 16, 256 // 16))]]

        cond_timestep = torch.zeros(inputs_embeds.size(0))
        t_emb = self.time_embed(cond_timestep)


        if self.first_step:
            x[:, gen_timestep_scatter_index:gen_timestep_scatter_index+1, :] = self.timestep_emb(timestep.reshape(-1)).reshape(bsz, -1, n_embd)
        else:
            t_emb = self.time_embed(timestep)
            x[:, 3:-1], token_height, token_width = self.patch_embed(x[:, 3:-1], t_emb)
            timestep_emb = self.timestep_emb(timestep).reshape(bsz, -1, n_embd)
            inputs_embeds = torch.cat([timestep_emb, x], dim=1)

        input_args = [inputs_embeds, x] if self.first_step else [inputs_embeds]

        #/////////////
        # cond_vae_images

        # cond_timestep_scatter_index 
        if cond_exists:
            with torch.no_grad():
                joint_image[:, 3:4, :] = self.timestep_emb(timestep.reshape(-1)).reshape(bsz, -1, n_embd)

            inputs_embeds = torch.cat([*input_args, joint_image], dim = 1)
        else:
            inputs_embeds = torch.cat([*input_args, joint_image[:, 1:, :]], dim = 1) # joint_image == eos_token

        attention_mask = torch.ones(inputs_embeds.shape[1], inputs_embeds.shape[1], dtype=torch.bool).tril(diagonal=0).repeat(bsz, 1, 1)
        for i in range(bsz):
            for _, image_slice in enumerate(img_slices[i]):
                attention_mask[i, image_slice, image_slice] = True
        attention_mask = attention_mask.unsqueeze(1)

        # pos embed
        position_ids = torch.arange(0, inputs_embeds.shape[1], dtype=torch.long, device=x.device)[None].expand(x.size(0), -1)
        cos, sin = build_batch_2d_rope(
            image_infos=rope_image_info,
            seq_len=inputs_embeds.shape[1],
            n_elem=128, # head dim
            base=10000.0,
        )
        custom_pos_emb = (sin.to(position_ids.device), cos.to(position_ids.device))
        custom_pos_emb = self.get_pos_emb(custom_pos_emb, position_ids)

        if self.kv_cache is None:
            # TODO: should change when higgsv2 gets merged
            self.kv_cache = HunyuanStaticCache(
                config=self.config,
                batch_size=x.size(0) * 2,
                max_cache_len = inputs_embeds.shape[1],
                dtype=x.dtype,
            )

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=self.kv_cache,
            inputs_embeds=inputs_embeds,
            custom_pos_emb=custom_pos_emb,
            first_step=self.first_step,
            gen_timestep_scatter_index=gen_timestep_scatter_index,
        )
        hidden_states = outputs[0]

        # safety no-op
        past_key_value = outputs[1]
        if past_key_value is not None:
            self.kv_cache = past_key_value

        hidden_states = hidden_states.to(inputs_embeds.device)
        img_mask = torch.zeros(hidden_states.size(1))
        img_mask[seq_len + x.size(1)+4:] = 1; img_mask[-1] = 0

        diffusion_prediction = self.ragged_final_layer(
            hidden_states, img_mask, timestep, int(token_height), int(token_width), self.first_step)
        
        if self.first_step:
            self.first_step = False

        return diffusion_prediction
