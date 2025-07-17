"""
Tiny AutoEncoder for Hunyuan Video
(DNN for encoding / decoding videos to Hunyuan Video's latent space)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


import comfy.utils
import comfy.ops

DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))

def conv(n_in, n_out, **kwargs):
    return comfy.ops.disable_weight_init.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(inplace=True), conv(n_out, n_out), nn.ReLU(inplace=True), conv(n_out, n_out))
        self.skip = comfy.ops.disable_weight_init.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))

class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = comfy.ops.disable_weight_init.Conv2d(n_f*stride, n_f, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))

class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = comfy.ops.disable_weight_init.Conv2d(n_f, n_f*stride, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)

def apply_model_with_memblocks(model, x, parallel, show_progress_bar):
    """
    Apply a sequential model with memblocks to the given input.
    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - parallel: if True, parallelize over timesteps (fast but uses O(T) memory)
        if False, each timestep will be processed sequentially (slow but uses O(1) memory)
    - show_progress_bar: if True, enables tqdm progressbar display
    Returns NTCHW tensor of output data.
    """
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    print('Received tensor of shape:',x.shape)
    if parallel:
        x = x.reshape(N*T, C, H, W)
        # parallel over input timesteps, iterate over blocks
        for b in model:
            if isinstance(b, MemBlock):
                NT, C, H, W = x.shape
                T = NT // N
                _x = x.reshape(N, T, C, H, W)
                mem = F.pad(_x, (0,0,0,0,0,0,1,0), value=0)[:,:T].reshape(x.shape)
                print('Intermediate shape:',x.shape)
                x = b(x, mem)
            else:
                print('Intermediate shape:',x.shape)
                x = b(x)
        NT, C, H, W = x.shape
        T = NT // N
        x = x.view(N, T, C, H, W)
    else:
        # TODO(oboerbohan): at least on macos this still gradually uses more memory during decode...
        # need to fix :(
        out = []
        # iterate over input timesteps and also iterate over blocks.
        # because of the cursed TPool/TGrow blocks, this is not a nested loop,
        # it's actually a ***graph traversal*** problem! so let's make a queue
        work_queue = [TWorkItem(xt, 0) for t, xt in enumerate(x.reshape(N, T * C, H, W).chunk(T, dim=1))]
        # we'll also need a separate addressable memory per node as well
        mem = [None] * len(model)
        while work_queue:
            xt, i = work_queue.pop(0)
            print('Intermediate shape:', xt.shape)
            if i == len(model):
                # reached end of the graph, append result to output list
                out.append(xt)
            else:
                # fetch the block to process
                b = model[i]
                if isinstance(b, MemBlock):
                    # mem blocks are simple since we're visiting the graph in causal order
                    if mem[i] is None:
                        xt_new = b(xt, xt * 0)
                        mem[i] = xt
                    else:
                        xt_new = b(xt, mem[i])
                        mem[i].copy_(xt) # inplace might reduce mysterious pytorch memory allocations? doesn't help though
                    # add successor to work queue
                    work_queue.insert(0, TWorkItem(xt_new, i+1))
                elif isinstance(b, TPool):
                    # pool blocks are miserable
                    if mem[i] is None:
                        mem[i] = [] # pool memory is itself a queue of inputs to pool
                    mem[i].append(xt)
                    if len(mem[i]) > b.stride:
                        # pool mem is in invalid state, we should have pooled before this
                        raise ValueError("???")
                    elif len(mem[i]) < b.stride:
                        # pool mem is not yet full, go back to processing the work queue
                        pass
                    else:
                        # pool mem is ready, run the pool block
                        N, C, H, W = xt.shape 
                        xt = b(torch.cat(mem[i], 1).view(N*b.stride, C, H, W))
                        # reset the pool mem
                        mem[i] = []
                        # add successor to work queue
                        work_queue.insert(0, TWorkItem(xt, i+1))
                elif isinstance(b, TGrow):
                    xt = b(xt)
                    NT, C, H, W = xt.shape
                    # each tgrow has multiple successor nodes
                    for xt_next in reversed(xt.view(N, b.stride*C, H, W).chunk(b.stride, 1)):
                        # add successor to work queue
                        work_queue.insert(0, TWorkItem(xt_next, i+1))
                else:
                    # normal block with no funny business
                    xt = b(xt)
                    # add successor to work queue
                    work_queue.insert(0, TWorkItem(xt, i+1))
        x = torch.stack(out, 1)
    return x

class TAEHV(nn.Module):
    latent_channels = 16
    image_channels = 3
    def __init__(self, checkpoint_path=None, decoder_time_upscale=(True, True), decoder_space_upscale=(True, True, True)):
        """Initialize pretrained TAEHV from the given checkpoint.
        Arg:
            checkpoint_path: path to weight file to load. taehv.pth for Hunyuan, taew2_1.pth for Wan 2.1.
            decoder_time_upscale: whether temporal upsampling is enabled for each block. upsampling can be disabled for a cheaper preview.
            decoder_space_upscale: whether spatial upsampling is enabled for each block. upsampling can be disabled for a cheaper preview.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            conv(TAEHV.image_channels, 64), nn.ReLU(inplace=True),
            TPool(64, 2), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            conv(64, TAEHV.latent_channels),
        )
        n_f = [256, 128, 64, 64]
        self.frames_to_trim = 2**sum(decoder_time_upscale) - 1
        self.decoder = nn.Sequential(
            Clamp(), conv(TAEHV.latent_channels, n_f[0]), nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1), TGrow(n_f[0], 1), conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1), conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1), conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True), conv(n_f[3], TAEHV.image_channels),
        )
        if checkpoint_path is not None:
            self.load_state_dict(comfy.utils.load_torch_file(checkpoint_path, safe_load=True))

    
    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(self.patch_tgrow_layers(state_dict), strict=strict)
    
    
    @staticmethod
    def from_comfy_state_dict(state_dict):
        """Create TAEHV model from ComfyUI-formatted state dict.
        
        Args:
            state_dict: State dict with taehv_decoder.* and taehv_encoder.* keys
            
        Returns:
            TAEHV model with loaded weights
        """
        # Create model without loading checkpoint
        model = TAEHV(checkpoint_path=None)

        # Convert ComfyUI state dict format back to TAEHV format
        taehv_sd = {}
        for key, value in state_dict.items():
            if key.startswith("taehv_decoder."):
                new_key = key.replace("taehv_decoder.", "decoder.")
                taehv_sd[new_key] = value
            elif key.startswith("taehv_encoder."):
                new_key = key.replace("taehv_encoder.", "encoder.")
                taehv_sd[new_key] = value

        # Load the converted state dict
        if taehv_sd:
            model.load_state_dict(taehv_sd, strict=False)

        return model

    def patch_tgrow_layers(self, sd):
        """Patch TGrow layers to use a smaller kernel if needed.
        Args:
            sd: state dict to patch
        """
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if sd[key].shape[0] > new_sd[key].shape[0]:
                    # take the last-timestep output channels
                    sd[key] = sd[key][-new_sd[key].shape[0]:]
        return sd

    def encode_video(self, x, parallel=False, show_progress_bar=False):
        """Encode a sequence of frames.
        Args:
            x: input NTCHW RGB (C=3) tensor with values in [0, 1].
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTCHW latent tensor with ~Gaussian values.
        """
        return apply_model_with_memblocks(self.encoder, x, parallel, show_progress_bar)

    def decode_video(self, x, parallel=False, show_progress_bar=False):
        """Decode a sequence of frames.
        Args:
            x: input NCTHW latent (C=16) tensor with ~Gaussian values.
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTCHW RGB tensor with ~[0, 1] values.
        """
        #converting NCTHW to NTCHW
        x = x.permute(0,2,1,3,4)
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)

        
        x = x[:, self.frames_to_trim:] # trim the time dimension
        
        #converting NTCHW to NCTHW
        x = x.permute(0,2,1,3,4)

        return x

    def decode(self, x):
        """Decode a single frame or batch of frames for preview."""
        if x.ndim == 4:
            # Add temporal dimension for single frame
            x = x.unsqueeze(1)

        # For preview, we'll just take the first frame after decoding
        decoded = self.decode_video(x, parallel=False, show_progress_bar=False)
        print(decoded.shape)

        #converting

        return decoded
        # Return single frame for preview
        # if decoded.shape[1] > 0:
        #     return decoded[:, 0]
        # else:
        #     return decoded.squeeze(1)

    def encode(self, x):
        """Encode a single frame or batch of frames."""
        if x.ndim == 4:
            # Add temporal dimension for single frame
            x = x.unsqueeze(1)

        encoded = self.encode_video(x, parallel=False, show_progress_bar=False)

        # Return single frame
        return encoded.squeeze(1)

    def forward(self, x):
        return self.decode(x)