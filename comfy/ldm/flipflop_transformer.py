import torch
import torch.cuda as cuda
import copy
from typing import List, Tuple
from dataclasses import dataclass

FLIPFLOP_REGISTRY = {}

def register(name):
    def decorator(cls):
        FLIPFLOP_REGISTRY[name] = cls
        return cls
    return decorator


@dataclass
class FlipFlopConfig:
    block_name: str
    block_wrap_fn: callable
    out_names: Tuple[str]
    overwrite_forward: str
    pinned_staging: bool = False
    inference_device: str = "cuda"
    offloading_device: str = "cpu"


def patch_model_from_config(model, config: FlipFlopConfig):
    block_list = getattr(model, config.block_name)
    flip_flop_transformer = FlipFlopTransformer(block_list,
                                                block_wrap_fn=config.block_wrap_fn,
                                                out_names=config.out_names,
                                                offloading_device=config.offloading_device,
                                                inference_device=config.inference_device,
                                                pinned_staging=config.pinned_staging)
    delattr(model, config.block_name)
    setattr(model, config.block_name, flip_flop_transformer)
    setattr(model, config.overwrite_forward, flip_flop_transformer.__call__)


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


@register("QwenImageTransformer2DModel")
class QwenImage:
    @staticmethod
    def qwen_blocks_wrap(block, **kwargs):
        kwargs["encoder_hidden_states"], kwargs["hidden_states"] = block(hidden_states=kwargs["hidden_states"],
                                                                         encoder_hidden_states=kwargs["encoder_hidden_states"],
                                                                         encoder_hidden_states_mask=kwargs["encoder_hidden_states_mask"],
                                                                         temb=kwargs["temb"],
                                                                         image_rotary_emb=kwargs["image_rotary_emb"])
        return kwargs

    blocks_config = FlipFlopConfig(block_name="transformer_blocks",
                                   block_wrap_fn=qwen_blocks_wrap,
                                   out_names=("encoder_hidden_states", "hidden_states"),
                                   overwrite_forward="block_fwd",
                                   pinned_staging=False)


    @staticmethod
    def patch(model):
        patch_model_from_config(model, QwenImage.blocks_config)
        return model

