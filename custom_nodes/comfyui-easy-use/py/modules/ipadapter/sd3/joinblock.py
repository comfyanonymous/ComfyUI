import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.modules.diffusionmodules.mmdit import (RMSNorm, JointBlock,)


class AdaLayerNorm(nn.Module):
    """
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, time_embedding_dim=None, mode="normal"):
        super().__init__()

        self.silu = nn.SiLU()
        num_params_dict = dict(
            zero=6,
            normal=2,
        )
        num_params = num_params_dict[mode]
        self.linear = nn.Linear(
            time_embedding_dim or embedding_dim, num_params * embedding_dim, bias=True
        )
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        self.mode = mode

    def forward(
        self,
        x,
        hidden_dtype=None,
        emb=None,
    ):
        emb = self.linear(self.silu(emb))
        if self.mode == "normal":
            shift_msa, scale_msa = emb.chunk(2, dim=1)
            x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
            return x

        elif self.mode == "zero":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(
                6, dim=1
            )
            x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class IPAttnProcessor(nn.Module):

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        ip_hidden_states_dim=None,
        ip_encoder_hidden_states_dim=None,
        head_dim=None,
        timesteps_emb_dim=1280,
    ):
        super().__init__()

        self.norm_ip = AdaLayerNorm(
            ip_hidden_states_dim, time_embedding_dim=timesteps_emb_dim
        )
        self.to_k_ip = nn.Linear(ip_hidden_states_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(ip_hidden_states_dim, hidden_size, bias=False)
        self.norm_q = RMSNorm(head_dim, 1e-6)
        self.norm_k = RMSNorm(head_dim, 1e-6)
        self.norm_ip_k = RMSNorm(head_dim, 1e-6)

    def forward(
        self,
        ip_hidden_states,
        img_query,
        img_key=None,
        img_value=None,
        t_emb=None,
        n_heads=1,
    ):
        if ip_hidden_states is None:
            return None

        if not hasattr(self, "to_k_ip") or not hasattr(self, "to_v_ip"):
            return None

        # norm ip input
        norm_ip_hidden_states = self.norm_ip(ip_hidden_states, emb=t_emb)

        # to k and v
        ip_key = self.to_k_ip(norm_ip_hidden_states)
        ip_value = self.to_v_ip(norm_ip_hidden_states)

        # reshape
        img_query = rearrange(img_query, "b l (h d) -> b h l d", h=n_heads)
        img_key = rearrange(img_key, "b l (h d) -> b h l d", h=n_heads)
        # note that the image is in a different shape: b l h d
        # so we transpose to b h l d
        # or do we have to transpose here?
        img_value = torch.transpose(img_value, 1, 2)
        ip_key = rearrange(ip_key, "b l (h d) -> b h l d", h=n_heads)
        ip_value = rearrange(ip_value, "b l (h d) -> b h l d", h=n_heads)

        # norm
        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        ip_key = self.norm_ip_k(ip_key)

        # cat img
        key = torch.cat([img_key, ip_key], dim=2)
        value = torch.cat([img_value, ip_value], dim=2)

        #
        ip_hidden_states = F.scaled_dot_product_attention(
            img_query, key, value, dropout_p=0.0, is_causal=False
        )
        ip_hidden_states = rearrange(ip_hidden_states, "b h l d -> b l (h d)")
        ip_hidden_states = ip_hidden_states.to(img_query.dtype)
        return ip_hidden_states


class JointBlockIPWrapper:
    """To be used as a patch_replace with Comfy"""

    def __init__(
        self,
        original_block: JointBlock,
        adapter: IPAttnProcessor,
        ip_options=None,
    ):
        self.original_block = original_block
        self.adapter = adapter
        if ip_options is None:
            ip_options = {}
        self.ip_options = ip_options

    def block_mixing(self, context, x, context_block, x_block, c):
        """
        Comes from mmdit.py. Modified to add ipadapter attention.
        """
        context_qkv, context_intermediates = context_block.pre_attention(context, c)

        if x_block.x_block_self_attn:
            x_qkv, x_qkv2, x_intermediates = x_block.pre_attention_x(x, c)
        else:
            x_qkv, x_intermediates = x_block.pre_attention(x, c)

        qkv = tuple(torch.cat((context_qkv[j], x_qkv[j]), dim=1) for j in range(3))

        attn = optimized_attention(
            qkv[0],
            qkv[1],
            qkv[2],
            heads=x_block.attn.num_heads,
        )
        context_attn, x_attn = (
            attn[:, : context_qkv[0].shape[1]],
            attn[:, context_qkv[0].shape[1] :],
        )
        # if the current timestep is not in the ipadapter enabling range, then the resampler wasn't run
        # and the hidden states will be None
        if (
            self.ip_options["hidden_states"] is not None
            and self.ip_options["t_emb"] is not None
        ):
            # IP-Adapter
            ip_attn = self.adapter(
                self.ip_options["hidden_states"],
                *x_qkv,
                self.ip_options["t_emb"],
                x_block.attn.num_heads,
            )
            x_attn = x_attn + ip_attn * self.ip_options["weight"]

        # Everything else is unchanged
        if not context_block.pre_only:
            context = context_block.post_attention(context_attn, *context_intermediates)

        else:
            context = None
        if x_block.x_block_self_attn:
            attn2 = optimized_attention(
                x_qkv2[0],
                x_qkv2[1],
                x_qkv2[2],
                heads=x_block.attn2.num_heads,
            )
            x = x_block.post_attention_x(x_attn, attn2, *x_intermediates)
        else:
            x = x_block.post_attention(x_attn, *x_intermediates)
        return context, x

    def __call__(self, args, _):
        # Code from mmdit.py:
        # in this case, we're blocks_replace[("double_block", i)]
        # note that although we're passed the original block,
        # we can't actually get it from inside its wrapper
        # (which would simplify the whole code...)
        #   ```
        #   def block_wrap(args):
        #       out = {}
        #       out["txt"], out["img"] = self.joint_blocks[i](args["txt"], args["img"], c=args["vec"])
        #       return out
        #   out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": c_mod}, {"original_block": block_wrap})
        #   context = out["txt"]
        #   x = out["img"]
        #   ```
        c, x = self.block_mixing(
            args["txt"],
            args["img"],
            self.original_block.context_block,
            self.original_block.x_block,
            c=args["vec"],
        )
        return {"txt": c, "img": x}