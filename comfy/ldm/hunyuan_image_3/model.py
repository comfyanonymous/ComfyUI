import os
import gc
import math
import torch
import psutil
import torch.nn as nn
from pathlib import Path
from einops import rearrange
import torch.nn.functional as F
from collections import OrderedDict
from safetensors import safe_open
from contextlib import contextmanager
from transformers.cache_utils import StaticCache
from typing import Optional, Tuple, Any, List, Dict
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.modules.diffusionmodules.openaimodel import ResBlock

INIT_MOE = torch.cuda.device_count() != 1

if not INIT_MOE:
    MOE_LAYER_SIZE = (1024**3) * 2.65 # approx
    CPU_MOE_RATIO = None

    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    
    INIT_CUDA_MEM = (props.total_memory - torch.cuda.memory_reserved()) * 0.9
    ADDITIONAL_LAYERS_IN_GPU = math.floor(INIT_CUDA_MEM / MOE_LAYER_SIZE)

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

def topkgating(
        logits: torch.Tensor,
        topk: int,
        norm_topk_prob: bool = True,
):
    logits = logits.float()
    gates = F.softmax(logits, dim=1)

    extra = ADDITIONAL_LAYERS_IN_GPU

    values_all, indices_all = torch.topk(gates, topk + extra, dim=1)
    expert_weight = values_all[:, :topk]
    expert_index = indices_all[:, :topk]

    _, cpu_expert_index = torch.topk(gates, int(CPU_MOE_RATIO * 64), dim = 1)
    cpu_expert_index = cpu_expert_index[:, (8 + ADDITIONAL_LAYERS_IN_GPU):]

    if norm_topk_prob and topk > 1:
        denom = expert_weight.sum(dim=1, keepdim=True).clamp_min(torch.finfo(gates.dtype).eps)
        expert_weight = expert_weight / denom

    return expert_weight, expert_index, cpu_expert_index, indices_all

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
                dropout=dropout,
                **factory_kwargs
            ))
        else:
            for i in range(self.patch_size // 2):
                self.model.append(ResBlock(
                    channels=hidden_channels,
                    emb_channels=emb_channels,
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
                dropout=dropout,
                **factory_kwargs
            ))
        else:
            for i in range(self.patch_size // 2):
                self.model.append(ResBlock(
                    channels=in_channels if i == 0 else hidden_channels,
                    emb_channels=emb_channels,
                    out_channels=hidden_channels,
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

        self.norm_topk_prob = True

    def forward(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)
        if self.wg.weight.dtype == torch.float32:
            hidden_states = hidden_states.float()
        logits = self.wg(hidden_states)
        gate_output = topkgating(logits, self.moe_topk, norm_topk_prob=self.norm_topk_prob,)

        return gate_output
    
class HunyuanMLP(nn.Module):
    def __init__(self, config, layer_idx=None, is_shared_mlp=False, is_moe=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config["hidden_size"]

        self.intermediate_size = 3072

        self.act_fn = torch.nn.functional.silu
        self.intermediate_size *= 2  # SwiGLU
        self.gate_and_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size // 2, self.hidden_size, bias=False)
    def forward(self, x):
        self.gate_and_up_proj, self.down_proj = self.gate_and_up_proj.to(x.device), self.down_proj.to(x.device)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        gate_and_up_proj = self.gate_and_up_proj(x)
        x1, x2 = gate_and_up_proj.chunk(2, dim=2)
        down_proj = self.down_proj(x1 * self.act_fn(x2))
        return down_proj

class MoELRUCache(nn.Module):
    def __init__(self, cpu_mem: int = 50, safety_buffer_bytes = 3*(1024**3), max_gpu_eviction_attempts = 8):
        super().__init__()
        global CPU_MOE_RATIO

        _, total = torch.cuda.mem_get_info()
        max_gpu_mem_gb = max((total - 2 * safety_buffer_bytes) / (1024**3), 1)

        self.MAX_GPU_MEM = int(max_gpu_mem_gb * 1024**3)
        self.MAX_CPU_MEM = int(cpu_mem * 1024**3)
        self.gpu_cache = OrderedDict()
        self.cpu_cache = OrderedDict()

        self.gpu_mem_usage = 0
        self.cpu_mem_usage = 0
        # 50% for system and headroom
        try:
            self.MAX_CPU_MEM = int((os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')) 
                                    - psutil.Process(os.getpid()).memory_info().rss 
                                    - safety_buffer_bytes) * 0.55
        except:
            self.MAX_CPU_MEM = int(cpu_mem * (1024**3) * 0.5) # TODO

        ADDITIONAL_LAYERS_IN_CPU = math.floor((50 * (1024**3)) / MOE_LAYER_SIZE)
        CPU_MOE_RATIO = (min(64 - ADDITIONAL_LAYERS_IN_GPU, ADDITIONAL_LAYERS_IN_CPU)) / 64

        self.MAX_GPU_MEM = int(max_gpu_mem_gb * 1024**3)
        self.SAFETY_BUFFER = int(safety_buffer_bytes)
        self.MAX_GPU_EVICT_ATTEMPTS = max_gpu_eviction_attempts
    
    def _gpu_free_bytes(self):
        free, total = torch.cuda.mem_get_info()
        return int(free)
    
    def _estimate_size(self, moe):
        # include parameters + buffers
        size = 0
        for p in moe.parameters():
            size += p.numel() * p.element_size()
        for b in moe.buffers():
            size += b.numel() * b.element_size()
        return int(size)

    def _evict_until_free(self, required_bytes, max_attempts=16):
        attempts = 0
        while self._gpu_free_bytes() < required_bytes and attempts < max_attempts:
            evicted = self._evict_from_gpu()
            if not evicted:
                break
            attempts += 1
        return self._gpu_free_bytes() >= required_bytes

    @contextmanager
    def ensure_headroom(self, required_bytes):

        safety = getattr(self, "SAFETY_BUFFER", 0)
        target_free = int(required_bytes + safety)

        if getattr(self, "_headroom", None) is not None:
            try:
                del self._headroom
            except Exception:
                pass
            self._headroom = None

        ok = self._evict_until_free(target_free)
        if not ok and self._gpu_free_bytes() < target_free:
            # last ditch
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        try:
            yield
        finally:
            if getattr(self, "_headroom", None) is None:
                try:
                    self._headroom = torch.empty((self._headroom_bytes,), dtype=torch.uint8, device="cuda:0")
                except Exception:
                    self._headroom = None

    def add_gpu(self, moe, index, allowed_retries=3):
        size = self._estimate_size(moe)

        while self.gpu_mem_usage + size > self.MAX_GPU_MEM:
            if not self._evict_from_gpu():
                break

        attempts = 0
        while self._gpu_free_bytes() < size + self.SAFETY_BUFFER and attempts < self.MAX_GPU_EVICT_ATTEMPTS:
            if not self._evict_from_gpu():
                break
            attempts += 1

        for _ in range(allowed_retries):
            try:
                moe_cuda = moe.to("cuda:0")
                break
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                evicted = self._evict_from_gpu()
                if not evicted: # can't evict
                    raise
        else:
            raise RuntimeError("Failed to move expert to GPU after evictions")

        self.gpu_cache[index] = moe_cuda
        self.gpu_cache.move_to_end(index)
        self.gpu_mem_usage += size

        return

    def add_cpu(self, moe, index):
        size = self._estimate_size(moe)
        while self.cpu_mem_usage + size > self.MAX_CPU_MEM:
            if not self._evict_from_cpu():
                break
        moe_cpu = moe.to("cpu")
        self.cpu_cache[index] = moe_cpu
        self.cpu_cache.move_to_end(index)
        self.cpu_mem_usage += size
    
    def get_from_device(self, index):
        if index in self.gpu_cache:
            moe = self.gpu_cache[index]
            self.gpu_cache.move_to_end(index)
            return moe
        if index in self.cpu_cache:
            moe = self.cpu_cache.pop(index)
            self.cpu_mem_usage = max(0, self.cpu_mem_usage - self._estimate_size(moe))
            try:
                self.add_gpu(moe, index)
                return self.gpu_cache[index]
            except RuntimeError:
                self.cpu_cache[index] = moe
                self.cpu_cache.move_to_end(index)
                self.cpu_mem_usage += self._estimate_size(moe)
                raise

        return None # load from disk

    def _evict_from_gpu(self):
        if not self.gpu_cache:
            return False

        idx, moe = self.gpu_cache.popitem(last=False)
        size = self._estimate_size(moe)
        self.gpu_mem_usage = max(0, self.gpu_mem_usage - size)

        if self.cpu_mem_usage + size <= self.MAX_CPU_MEM:
            try:
                moe_cpu = moe.to("cpu")
            except Exception:
                # drop the model if cpu is full
                del moe
                return True
            self.cpu_cache[idx] = moe_cpu
            self.cpu_cache.move_to_end(idx)
            self.cpu_mem_usage += size
            return True
        else:
            del moe
            return True
        
    def _evict_from_cpu(self):
        if not self.cpu_cache:
            return False
        _, moe = self.cpu_cache.popitem(last=False)
        size = self._estimate_size(moe)
        self.cpu_mem_usage = max(0, self.cpu_mem_usage - size)
        del moe
        gc.collect()
        return True
        
class LazyMoELoader(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def lazy_init(self, config, layer_idx, expert_idx):
        comfyui_dir = Path.home() / "ComfyUI"
        checkpoint = comfyui_dir / "models" / "checkpoint" / "hunyuan_image_3.safetensors"
        checkpoint = checkpoint.resolve()
        if not os.path.exists(checkpoint):
            raise ValueError(f"Hunyuan Image 3 Checkpoint on one GPU should have the path: {checkpoint}")
        
        prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}."
        additional_prefix = f"model.layers.{layer_idx}.mlp.gate_and_up_proj.weight"
        sd = {}
    
        with safe_open(checkpoint, framework="pt", device=self.device) as f:
            for k in f.keys():
                if k.startswith(prefix) or k.startswith(additional_prefix):
                    new_k = k.split(f"experts.{expert_idx}.", 1)[1]
                    sd[new_k] = f.get_tensor(k)

        return HunyuanMLP(config, layer_idx=layer_idx, is_shared_mlp=False, is_moe=True).load_state_dict(sd).to(self.deivce)
    
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
            self.experts = None
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
            expert_weight, expert_index, cpu_expert_index, indices_all = self.gate(hidden_states)
            if not INIT_MOE:
                if ADDITIONAL_LAYERS_IN_GPU > 0:
                    additional_expert_index = indices_all[:, expert_index.size(1): expert_index.size(1) + ADDITIONAL_LAYERS_IN_GPU]
 
                    flat = additional_expert_index.reshape(-1).to("cpu")
                    counts = torch.bincount(flat, minlength=self.num_experts)
                    top_extra = torch.topk(counts, k=min(ADDITIONAL_LAYERS_IN_GPU, (counts>0).sum().item())).indices.tolist()
                        
                    for expert_id in top_extra:
                        if self.moe_lru.get_from_device(expert_id + self.layer_idx) is None:
                            expert_cpu = LazyMoELoader(device="cpu").lazy_init(self.config, self.layer_idx, expert_id)
                            self.moe_lru.add_gpu(expert_cpu, expert_id + self.layer_idx)
            
                if cpu_expert_index is not None and cpu_expert_index.numel() > 0:
                    for expert_id in torch.unique(cpu_expert_index).cpu().tolist():
                        if self.moe_lru.get_from_device(expert_id + self.layer_idx) is None:
                            expert_cpu = LazyMoELoader(device="cpu").lazy_init(self.config, self.layer_idx, expert_id)
                            self.moe_lru.add_cpu(expert_cpu, expert_id + self.layer_idx)
            
            combined_output = torch.zeros_like(reshaped_input)
            experts_list = []
            for e in range(self.num_experts):
                token_mask = (expert_index == e)
                if not token_mask.any():
                    continue
                expert = self.moe_lru.get_from_device(e + self.layer_idx)
                if expert is None:
                    expert = LazyMoELoader()
                    expert = expert.lazy_init(self.config, self.layer_idx, e)
                    self.moe_lru.add_gpu(expert, e + self.layer_idx)
                    experts_list.append((e, expert))

            per_pos, per_tokens, per_weights = [], [], []
            for e, _ in experts_list:
                token_mask = (expert_index == e)

                token_ids = token_mask.nonzero(as_tuple=False)
                token_positions = token_ids[:, 0]
                topk_slot = token_ids[:, 1]

                tokens = reshaped_input[token_positions]
                weights = expert_weight[token_positions, topk_slot]

                per_pos.append(token_positions)
                per_tokens.append(tokens)
                per_weights.append(weights)

            lengths = [t.shape[0] for t in per_tokens]
            E = len(per_tokens)
            L = max(lengths)
            tokens_padded = torch.zeros((E, L, hidden_size), device=hidden_states.device, dtype=reshaped_input.dtype)
            weights_padded = torch.zeros((E, L), device=hidden_states.device, dtype=per_weights[0].dtype)
            for i, t in enumerate(per_tokens):
                tokens_padded[i, : t.shape[0]] = t
                weights_padded[i, : t.shape[0]] = per_weights[i]

            l1, l2 = [], []
            for _, expert in experts_list:
                l1.append(expert.gate_and_up_proj)
                l2.append(expert.down_proj)

            W1 = torch.stack([l.weight for l in l1]).to(hidden_states.device)
            W2 = torch.stack([l.weight for l in l2]).to(hidden_states.device)

            W1_T = W1.transpose(1, 2)
            W2_T = W2.transpose(1, 2)

            x = torch.bmm(tokens_padded, W1_T)
            x = F.silu(x)

            out_padded = torch.bmm(x, W2_T)

            out_padded = out_padded * weights_padded.unsqueeze(-1)

            for i, token_positions in enumerate(per_pos):
                Ni = lengths[i]
                out_i = out_padded[i, :Ni]
                combined_output.index_add_(0, token_positions.to(hidden_states.device), out_i)

            #dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.type_as(hidden_states), reshaped_input)
            #chunks = dispatched_input.chunk(self.num_experts, dim=0)
            #expert_outputs = []
            #for chunk, expert in zip(chunks, self.experts):
            #    expert_outputs.append(expert(chunk))

            #expert_output = torch.cat(expert_outputs, dim=0)
            #combined_output = torch.einsum("sec,ecm->sm", combine_weights.type_as(hidden_states), expert_output)

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
        self.head_dim = self.hidden_size // self.num_heads
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
        self.wte = nn.Embedding(133120, config["hidden_size"], self.padding_idx)
        self.layers = nn.ModuleList(
            [HunyuanImage3DecoderLayer(config, layer_idx, moe_lru = moe_lru) for layer_idx in range(config["num_hidden_layers"])]
        )
        
        self.ln_f = HunyuanRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

        self.shared_tensor = None

    def forward(
            self,
            input_ids: torch.LongTensor = None,
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

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        next_decoder_cache = None
        for layer_idx, decoder_layer in enumerate(self.layers):

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

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[1]

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache

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

        if self.kv_cache is None:
            # TODO: should change when higgsv2 gets merged
            self.kv_cache = HunyuanStaticCache(
                config=self.config,
                batch_size=x.size(0) * 2,
                max_cache_len = inputs_embeds.shape[1],
                dtype=x.dtype,
            )

        image_mask = torch.ones(x.size(1), device=x.device)
        image_mask[:3] = torch.zeros(3); image_mask[-1] = torch.zeros(1)
        gen_timestep_scatter_index = 4

        with torch.no_grad():
            joint_image[:, 2, :] = x[:, 2, :] # updates image ratio 

        position_ids = torch.arange(0, inputs_embeds.shape[1], dtype=torch.long, device=x.device)[None].expand(x.size(0), -1)
        height, width = x.shape[2] * 16, x.shape[3] * 16
        token_height = height // (16 * 16)
        token_width = width // (16 * 16)
        
        batch_image_slices = []
        for i in range(x.size(0)):
            # slice the vae and vit parts + slice the latent from x
            joint_slices_i = [slice(3, cond_vae_image_mask[i].size(0) + 3), slice(cond_vae_image_mask[i].size(0) + 4, joint_image.size(1) - 1)]
            gen_slices_i = [slice(3, x[i].size(1) - 1)]
            batch_image_slices.append(joint_slices_i + gen_slices_i)

        rope_image_info = [
            [(s, (token_height, token_width)) for s in slices_i]
            for slices_i in batch_image_slices
        ]
        seq_len = inputs_embeds.shape[1]
        cos, sin = build_batch_2d_rope(
            image_infos=rope_image_info,
            seq_len=seq_len,
            n_elem=self.config["hidden_size"] // self.config["num_attention_heads"],
            base=10000.0,
        )
        custom_pos_emb = (sin.to(position_ids.device), cos.to(position_ids.device))

        custom_pos_emb = self.get_pos_emb(custom_pos_emb, position_ids)

        cond_timestep = torch.zeros(inputs_embeds.size(0))
        t_emb = self.time_embed(cond_timestep)

        bsz, seq_len, n_embd = inputs_embeds.shape

        # FIXME: token_h and token_w for the first step
        if self.first_step:
            x[:, gen_timestep_scatter_index:gen_timestep_scatter_index+1, :] = self.timestep_emb(timestep.reshape(-1)).reshape(bsz, -1, n_embd)
        else:
            t_emb = self.time_embed(timestep)
            x[:, 3:-1], token_h, token_w = self.patch_embed(x[:, 3:-1], t_emb)
            timestep_emb = self.timestep_emb(timestep).reshape(bsz, -1, n_embd)
            x = torch.cat([timestep_emb, x], dim=1)

        inputs_embeds = torch.cat([inputs_embeds, x], dim = 1)

        #/////////////
        # cond_vae_images

        # cond_timestep_scatter_index 
        joint_image[:, 3] = self.timestep_emb(timestep.reshape(-1)).reshape(bsz, -1, n_embd)

        inputs_embeds = torch.cat([inputs_embeds, joint_image], dim = 1)

        attention_mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril(diagonal=0).repeat(bsz, 1, 1)
        for i in range(bsz):
            for _, image_slice in enumerate(batch_image_slices[i]):
                attention_mask[i, image_slice, image_slice] = True
        attention_mask = attention_mask.unsqueeze(1)

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
        diffusion_prediction = self.ragged_final_layer(
            hidden_states, image_mask, timestep, token_h, token_w, self.first_step)
        
        if self.first_step:
            self.first_step = False

        return diffusion_prediction
