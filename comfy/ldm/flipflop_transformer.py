from __future__ import annotations
import torch
import copy

import comfy.model_management


class FlipFlopModule(torch.nn.Module):
    def __init__(self, block_types: tuple[str, ...], enable_flipflop: bool = True):
        super().__init__()
        self.block_types = block_types
        self.enable_flipflop = enable_flipflop
        self.flipflop: dict[str, FlipFlopHolder] = {}
        self.block_info: dict[str, tuple[int, int]] = {}
        self.flipflop_prefixes: list[str] = []

    def setup_flipflop_holders(self, block_info: dict[str, tuple[int, int]], flipflop_prefixes: list[str], load_device: torch.device, offload_device: torch.device):
        for block_type, (flipflop_blocks, total_blocks) in block_info.items():
            if block_type in self.flipflop:
                continue
            self.flipflop[block_type] = FlipFlopHolder(getattr(self, block_type)[total_blocks-flipflop_blocks:], flipflop_blocks, total_blocks, load_device, offload_device)
            self.block_info[block_type] = (flipflop_blocks, total_blocks)
        self.flipflop_prefixes = flipflop_prefixes.copy()

    def init_flipflop_block_copies(self, device: torch.device) -> int:
        memory_freed = 0
        for holder in self.flipflop.values():
            memory_freed += holder.init_flipflop_block_copies(device)
        return memory_freed

    def clean_flipflop_holders(self):
        memory_freed = 0
        for block_type in list(self.flipflop.keys()):
            memory_freed += self.flipflop[block_type].clean_flipflop_blocks()
            del self.flipflop[block_type]
        self.block_info = {}
        self.flipflop_prefixes = []
        return memory_freed

    def get_all_blocks(self, block_type: str) -> list[torch.nn.Module]:
        return getattr(self, block_type)

    def get_blocks(self, block_type: str) -> torch.nn.ModuleList:
        if block_type not in self.block_types:
            raise ValueError(f"Block type {block_type} not found in {self.block_types}")
        if block_type in self.flipflop:
            return getattr(self, block_type)[:self.flipflop[block_type].i_offset]
        return getattr(self, block_type)

    def get_all_block_module_sizes(self, reverse_sort_by_size: bool = False) -> list[tuple[str, int]]:
        '''
        Returns a list of (block_type, size) sorted by size.
        If reverse_sort_by_size is True, the list is sorted by size in reverse order.
        '''
        sizes = [(block_type, self.get_block_module_size(block_type)) for block_type in self.block_types]
        sizes.sort(key=lambda x: x[1], reverse=reverse_sort_by_size)
        return sizes

    def get_block_module_size(self, block_type: str) -> int:
        return comfy.model_management.module_size(getattr(self, block_type)[0])

    def execute_blocks(self, block_type: str, func, out: torch.Tensor | tuple[torch.Tensor,...], *args, **kwargs):
        # execute blocks, supporting both single and double (or higher) block types
        if isinstance(out, torch.Tensor):
            out = (out,)
        for i, block in enumerate(self.get_blocks(block_type)):
            out = func(i, block, *out, *args, **kwargs)
            if isinstance(out, torch.Tensor):
                out = (out,)
        if block_type in self.flipflop:
            holder = self.flipflop[block_type]
            with holder.context() as ctx:
                for i, block in enumerate(holder.blocks):
                    out = ctx(func, i, block, *out, *args, **kwargs)
                    if isinstance(out, torch.Tensor):
                        out = (out,)
        if len(out) == 1:
            out = out[0]
        return out


class FlipFlopContext:
    def __init__(self, holder: FlipFlopHolder):
        # NOTE: there is a bug when there are an odd number of blocks to flipflop.
        # Worked around right now by always making sure it will be even, but need to resolve.
        self.holder = holder
        self.reset()

    def reset(self):
        self.num_blocks = len(self.holder.blocks)
        self.first_flip = True
        self.first_flop = True
        self.last_flip = False
        self.last_flop = False

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.holder.compute_stream.record_event(self.holder.cpy_end_event)

    def do_flip(self, func, i: int, _, *args, **kwargs):
        # flip
        self.holder.compute_stream.wait_event(self.holder.cpy_end_event)
        with torch.cuda.stream(self.holder.compute_stream):
            out = func(i+self.holder.i_offset, self.holder.flip, *args, **kwargs)
        self.holder.event_flip.record(self.holder.compute_stream)
        # while flip executes, queue flop to copy to its next block
        next_flop_i = i + 1
        if next_flop_i >= self.num_blocks:
            next_flop_i = next_flop_i - self.num_blocks
            self.last_flip = True
        if not self.first_flip:
            self.holder._copy_state_dict(self.holder.flop.state_dict(), self.holder.blocks[next_flop_i].state_dict(), self.holder.event_flop, self.holder.cpy_end_event)
        if self.last_flip:
            self.holder._copy_state_dict(self.holder.flip.state_dict(), self.holder.blocks[0].state_dict(), cpy_start_event=self.holder.event_flip)
        self.first_flip = False
        return out

    def do_flop(self, func, i: int, _, *args, **kwargs):
        # flop
        if not self.first_flop:
            self.holder.compute_stream.wait_event(self.holder.cpy_end_event)
        with torch.cuda.stream(self.holder.compute_stream):
            out = func(i+self.holder.i_offset, self.holder.flop, *args, **kwargs)
        self.holder.event_flop.record(self.holder.compute_stream)
        # while flop executes, queue flip to copy to its next block
        next_flip_i = i + 1
        if next_flip_i >= self.num_blocks:
            next_flip_i = next_flip_i - self.num_blocks
            self.last_flop = True
        self.holder._copy_state_dict(self.holder.flip.state_dict(), self.holder.blocks[next_flip_i].state_dict(), self.holder.event_flip, self.holder.cpy_end_event)
        if self.last_flop:
            self.holder._copy_state_dict(self.holder.flop.state_dict(), self.holder.blocks[1].state_dict(), cpy_start_event=self.holder.event_flop)
        self.first_flop = False
        return out

    @torch.no_grad()
    def __call__(self, func, i: int, block: torch.nn.Module, *args, **kwargs):
        # flips are even indexes, flops are odd indexes
        if i % 2 == 0:
            return self.do_flip(func, i, block, *args, **kwargs)
        else:
            return self.do_flop(func, i, block, *args, **kwargs)


class FlipFlopHolder:
    def __init__(self, blocks: list[torch.nn.Module], flip_amount: int, total_amount: int, load_device: torch.device, offload_device: torch.device):
        self.load_device = load_device
        self.offload_device = offload_device
        self.blocks = blocks
        self.flip_amount = flip_amount
        self.total_amount = total_amount
        # NOTE: used to make sure block indexes passed into block functions match expected patch indexes
        self.i_offset = total_amount - flip_amount

        self.block_module_size = 0
        if len(self.blocks) > 0:
            self.block_module_size = comfy.model_management.module_size(self.blocks[0])

        self.flip: torch.nn.Module = None
        self.flop: torch.nn.Module = None

        self.compute_stream = torch.cuda.default_stream(self.load_device)
        self.cpy_stream = torch.cuda.Stream(self.load_device)

        self.event_flip = torch.cuda.Event(enable_timing=False)
        self.event_flop = torch.cuda.Event(enable_timing=False)
        self.cpy_end_event = torch.cuda.Event(enable_timing=False)
        # INIT - is this actually needed?
        self.compute_stream.record_event(self.cpy_end_event)

    def _copy_state_dict(self, dst, src, cpy_start_event: torch.cuda.Event=None, cpy_end_event: torch.cuda.Event=None):
        if cpy_start_event:
            self.cpy_stream.wait_event(cpy_start_event)

        with torch.cuda.stream(self.cpy_stream):
            for k, v in src.items():
                dst[k].copy_(v, non_blocking=True)
        if cpy_end_event:
            cpy_end_event.record(self.cpy_stream)

    def context(self):
        return FlipFlopContext(self)

    def init_flipflop_block_copies(self, load_device: torch.device) -> int:
        self.flip = copy.deepcopy(self.blocks[0]).to(device=load_device)
        self.flop = copy.deepcopy(self.blocks[1]).to(device=load_device)
        return comfy.model_management.module_size(self.flip) + comfy.model_management.module_size(self.flop)

    def clean_flipflop_blocks(self) -> int:
        memory_freed = 0
        memory_freed += comfy.model_management.module_size(self.flip)
        memory_freed += comfy.model_management.module_size(self.flop)
        del self.flip
        del self.flop
        self.flip = None
        self.flop = None
        return memory_freed
