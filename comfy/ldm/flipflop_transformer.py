from __future__ import annotations
import torch
import torch.cuda as cuda
import copy
from typing import List, Tuple

import comfy.model_management


class FlipFlopContext:
    def __init__(self, holder: FlipFlopHolder):
        self.holder = holder
        self.reset()

    def reset(self):
        self.num_blocks = len(self.holder.blocks)
        self.first_flip = True
        self.first_flop = True
        self.last_flip = False
        self.last_flop = False
        # TODO: the 'i' that's passed into func needs to be properly offset to do patches correctly

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.holder.compute_stream.record_event(self.holder.cpy_end_event)

    def do_flip(self, func, i: int, _, *args, **kwargs):
        # flip
        self.holder.compute_stream.wait_event(self.holder.cpy_end_event)
        with torch.cuda.stream(self.holder.compute_stream):
            out = func(i, self.holder.flip, *args, **kwargs)
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
            out = func(i, self.holder.flop, *args, **kwargs)
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
    def __init__(self, blocks: List[torch.nn.Module], flip_amount: int, load_device="cuda", offload_device="cpu"):
        self.load_device = torch.device(load_device)
        self.offload_device = torch.device(offload_device)
        self.blocks = blocks
        self.flip_amount = flip_amount

        self.block_module_size = 0
        if len(self.blocks) > 0:
            self.block_module_size = comfy.model_management.module_size(self.blocks[0])

        self.flip: torch.nn.Module = None
        self.flop: torch.nn.Module = None
        # TODO: make initialization happen in model management code/model patcher, not here
        self.init_flipflop_blocks(self.load_device)

        self.compute_stream = cuda.default_stream(self.load_device)
        self.cpy_stream = cuda.Stream(self.load_device)

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

    def init_flipflop_blocks(self, load_device: torch.device):
        self.flip = copy.deepcopy(self.blocks[0]).to(device=load_device)
        self.flop = copy.deepcopy(self.blocks[1]).to(device=load_device)

    def clean_flipflop_blocks(self):
        del self.flip
        del self.flop
        self.flip = None
        self.flop = None


class FlopFlopModule(torch.nn.Module):
    def __init__(self, block_types: tuple[str, ...]):
        super().__init__()
        self.block_types = block_types
        self.flipflop: dict[str, FlipFlopHolder] = {}

    def setup_flipflop_holders(self, block_percentage: float):
        for block_type in self.block_types:
            if block_type in self.flipflop:
                continue
            num_blocks = int(len(self.transformer_blocks) * (1.0-block_percentage))
            self.flipflop["transformer_blocks"] = FlipFlopHolder(self.transformer_blocks[num_blocks:], num_blocks)

    def clean_flipflop_holders(self):
        for block_type in self.flipflop.keys():
            self.flipflop[block_type].clean_flipflop_blocks()
            del self.flipflop[block_type]

    def get_blocks(self, block_type: str) -> torch.nn.ModuleList:
        if block_type not in self.block_types:
            raise ValueError(f"Block type {block_type} not found in {self.block_types}")
        if block_type in self.flipflop:
            return getattr(self, block_type)[:self.flipflop[block_type].flip_amount]
        return getattr(self, block_type)

    def get_all_block_module_sizes(self, sort_by_size: bool = False) -> list[tuple[str, int]]:
        '''
        Returns a list of (block_type, size).
        If sort_by_size is True, the list is sorted by size.
        '''
        sizes = [(block_type, self.get_block_module_size(block_type)) for block_type in self.block_types]
        if sort_by_size:
            sizes.sort(key=lambda x: x[1])
        return sizes

    def get_block_module_size(self, block_type: str) -> int:
        return comfy.model_management.module_size(getattr(self, block_type)[0])


# Below is the implementation from contentis' prototype flip flop
class FlipFlopTransformer:
    def __init__(self, transformer_blocks: List[torch.nn.Module], block_wrap_fn, out_names: Tuple[str], pinned_staging: bool = False, inference_device="cuda", offloading_device="cpu"):
        self.transformer_blocks = transformer_blocks
        self.offloading_device = torch.device(offloading_device)
        self.inference_device = torch.device(inference_device)
        self.staging = pinned_staging

        self.flip = copy.deepcopy(self.transformer_blocks[0]).to(device=self.inference_device)
        self.flop = copy.deepcopy(self.transformer_blocks[1]).to(device=self.inference_device)

        self._cpy_fn = self._copy_state_dict
        if self.staging:
            self.staging_buffer = self._pin_module(self.transformer_blocks[0]).state_dict()
            self._cpy_fn = self._copy_state_dict_with_staging

        self.compute_stream = cuda.default_stream(self.inference_device)
        self.cpy_stream = cuda.Stream(self.inference_device)

        self.event_flip = torch.cuda.Event(enable_timing=False)
        self.event_flop = torch.cuda.Event(enable_timing=False)
        self.cpy_end_event = torch.cuda.Event(enable_timing=False)

        self.block_wrap_fn = block_wrap_fn
        self.out_names = out_names

        self.num_blocks = len(self.transformer_blocks)
        self.extra_run = self.num_blocks % 2

        # INIT
        self.compute_stream.record_event(self.cpy_end_event)

    def _copy_state_dict(self, dst, src, cpy_start_event=None, cpy_end_event=None):
        if cpy_start_event:
            self.cpy_stream.wait_event(cpy_start_event)

        with torch.cuda.stream(self.cpy_stream):
            for k, v in src.items():
                dst[k].copy_(v, non_blocking=True)
        if cpy_end_event:
            cpy_end_event.record(self.cpy_stream)

    def _copy_state_dict_with_staging(self, dst, src, cpy_start_event=None, cpy_end_event=None):
        if cpy_start_event:
            self.cpy_stream.wait_event(cpy_start_event)

        with torch.cuda.stream(self.cpy_stream):
            for k, v in src.items():
                self.staging_buffer[k].copy_(v, non_blocking=True)
                dst[k].copy_(self.staging_buffer[k], non_blocking=True)
        if cpy_end_event:
            cpy_end_event.record(self.cpy_stream)

    def _pin_module(self, module):
        pinned_module = copy.deepcopy(module)
        for param in pinned_module.parameters():
            param.data = param.data.pin_memory()
        # Pin all buffers (if any)
        for buffer in pinned_module.buffers():
            buffer.data = buffer.data.pin_memory()
        return pinned_module

    def _reset(self):
        if self.extra_run:
            self._copy_state_dict(self.flop.state_dict(), self.transformer_blocks[1].state_dict(), cpy_start_event=self.event_flop)
            self._copy_state_dict(self.flip.state_dict(), self.transformer_blocks[0].state_dict(), cpy_start_event=self.event_flip)
        else:
            self._copy_state_dict(self.flip.state_dict(), self.transformer_blocks[0].state_dict(), cpy_start_event=self.event_flip)
            self._copy_state_dict(self.flop.state_dict(), self.transformer_blocks[1].state_dict(), cpy_start_event=self.event_flop)

        self.compute_stream.record_event(self.cpy_end_event)

    @torch.no_grad()
    def __call__(self, **feed_dict):
        '''
        Flip accounts for even blocks (0 is first block), flop accounts for odd blocks.
        '''
        # separated flip flop refactor
        num_blocks = len(self.transformer_blocks)
        first_flip = True
        first_flop = True
        last_flip = False
        last_flop = False
        for i, block in enumerate(self.transformer_blocks):
            is_flip = i % 2 == 0
            if is_flip:
                # flip
                self.compute_stream.wait_event(self.cpy_end_event)
                with torch.cuda.stream(self.compute_stream):
                    feed_dict = self.block_wrap_fn(self.flip, **feed_dict)
                self.event_flip.record(self.compute_stream)
                # while flip executes, queue flop to copy to its next block
                next_flop_i = i + 1
                if next_flop_i >= num_blocks:
                    next_flop_i = next_flop_i - num_blocks
                    last_flip = True
                if not first_flip:
                    self._copy_state_dict(self.flop.state_dict(), self.transformer_blocks[next_flop_i].state_dict(), self.event_flop, self.cpy_end_event)
                if last_flip:
                    self._copy_state_dict(self.flip.state_dict(), self.transformer_blocks[0].state_dict(), cpy_start_event=self.event_flip)
                first_flip = False
            else:
                # flop
                if not first_flop:
                    self.compute_stream.wait_event(self.cpy_end_event)
                with torch.cuda.stream(self.compute_stream):
                    feed_dict = self.block_wrap_fn(self.flop, **feed_dict)
                self.event_flop.record(self.compute_stream)
                # while flop executes, queue flip to copy to its next block
                next_flip_i = i + 1
                if next_flip_i >= num_blocks:
                    next_flip_i = next_flip_i - num_blocks
                    last_flop = True
                self._copy_state_dict(self.flip.state_dict(), self.transformer_blocks[next_flip_i].state_dict(), self.event_flip, self.cpy_end_event)
                if last_flop:
                    self._copy_state_dict(self.flop.state_dict(), self.transformer_blocks[1].state_dict(), cpy_start_event=self.event_flop)
                first_flop = False

        self.compute_stream.record_event(self.cpy_end_event)

        outputs = [feed_dict[name] for name in self.out_names]
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    @torch.no_grad()
    def __call__old(self, **feed_dict):
        # contentis' prototype flip flop
        # Wait for reset
        self.compute_stream.wait_event(self.cpy_end_event)

        with torch.cuda.stream(self.compute_stream):
            feed_dict = self.block_wrap_fn(self.flip, **feed_dict)
        self.event_flip.record(self.compute_stream)

        for i in range(self.num_blocks // 2 - 1):

            with torch.cuda.stream(self.compute_stream):
                feed_dict = self.block_wrap_fn(self.flop, **feed_dict)
            self.event_flop.record(self.compute_stream)

            self._cpy_fn(self.flip.state_dict(), self.transformer_blocks[(i + 1) * 2].state_dict(), self.event_flip,
                            self.cpy_end_event)

            self.compute_stream.wait_event(self.cpy_end_event)

            with torch.cuda.stream(self.compute_stream):
                feed_dict = self.block_wrap_fn(self.flip, **feed_dict)
            self.event_flip.record(self.compute_stream)

            self._cpy_fn(self.flop.state_dict(), self.transformer_blocks[(i + 1) * 2 + 1].state_dict(), self.event_flop,
                            self.cpy_end_event)
            self.compute_stream.wait_event(self.cpy_end_event)


        with torch.cuda.stream(self.compute_stream):
            feed_dict = self.block_wrap_fn(self.flop, **feed_dict)
        self.event_flop.record(self.compute_stream)

        if self.extra_run:
            self._cpy_fn(self.flip.state_dict(), self.transformer_blocks[-1].state_dict(), self.event_flip,
                            self.cpy_end_event)
            self.compute_stream.wait_event(self.cpy_end_event)
            with torch.cuda.stream(self.compute_stream):
                feed_dict = self.block_wrap_fn(self.flip, **feed_dict)
            self.event_flip.record(self.compute_stream)

        self._reset()

        outputs = [feed_dict[name] for name in self.out_names]
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

# @register("Flux")
# class Flux:
#     @staticmethod
#     def double_block_wrap(block, **kwargs):
#         kwargs["img"], kwargs["txt"] = block(img=kwargs["img"],
#                                              txt=kwargs["txt"],
#                                              vec=kwargs["vec"],
#                                              pe=kwargs["pe"],
#                                              attn_mask=kwargs.get("attn_mask"))
#         return kwargs

#     @staticmethod
#     def single_block_wrap(block, **kwargs):
#         kwargs["img"] = block(kwargs["img"],
#                               vec=kwargs["vec"],
#                               pe=kwargs["pe"],
#                               attn_mask=kwargs.get("attn_mask"))
#         return kwargs

#     double_config = FlipFlopConfig(block_name="double_blocks",
#                                    block_wrap_fn=double_block_wrap,
#                                    out_names=("img", "txt"),
#                                    overwrite_forward="double_transformer_fwd",
#                                    pinned_staging=False)

#     single_config = FlipFlopConfig(block_name="single_blocks",
#                                    block_wrap_fn=single_block_wrap,
#                                    out_names=("img",),
#                                    overwrite_forward="single_transformer_fwd",
#                                    pinned_staging=False)
#     @staticmethod
#     def patch(model):
#         patch_model_from_config(model, Flux.double_config)
#         patch_model_from_config(model, Flux.single_config)
#         return model


# @register("WanModel")
# class Wan:
#     @staticmethod
#     def wan_blocks_wrap(block, **kwargs):
#         kwargs["x"] = block(x=kwargs["x"],
#                             context=kwargs["context"],
#                             e=kwargs["e"],
#                             freqs=kwargs["freqs"],
#                             context_img_len=kwargs.get("context_img_len"))
#         return kwargs

#     blocks_config = FlipFlopConfig(block_name="blocks",
#                                    block_wrap_fn=wan_blocks_wrap,
#                                    out_names=("x",),
#                                    overwrite_forward="block_fwd",
#                                    pinned_staging=False)


#     @staticmethod
#     def patch(model):
#         patch_model_from_config(model, Wan.blocks_config)
#         return model

# @register("QwenImageTransformer2DModel")
# class QwenImage:
#     @staticmethod
#     def qwen_blocks_wrap(block, **kwargs):
#         kwargs["encoder_hidden_states"], kwargs["hidden_states"] = block(hidden_states=kwargs["hidden_states"],
#                                                                          encoder_hidden_states=kwargs["encoder_hidden_states"],
#                                                                          encoder_hidden_states_mask=kwargs["encoder_hidden_states_mask"],
#                                                                          temb=kwargs["temb"],
#                                                                          image_rotary_emb=kwargs["image_rotary_emb"],
#                                                                          transformer_options=kwargs["transformer_options"])
#         return kwargs

#     blocks_config = FlipFlopConfig(block_name="transformer_blocks",
#                                    block_wrap_fn=qwen_blocks_wrap,
#                                    out_names=("encoder_hidden_states", "hidden_states"),
#                                    overwrite_forward="blocks_fwd",
#                                    pinned_staging=False)


#     @staticmethod
#     def patch(model):
#         patch_model_from_config(model, QwenImage.blocks_config)
#         return model
