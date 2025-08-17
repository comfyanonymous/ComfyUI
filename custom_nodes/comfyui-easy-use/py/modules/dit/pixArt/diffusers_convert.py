# For using the diffusers format weights
#  Based on the original ComfyUI function +
#  https://github.com/PixArt-alpha/PixArt-alpha/blob/master/tools/convert_pixart_alpha_to_diffusers.py
import torch

conversion_map_ms = [  # for multi_scale_train (MS)
    # Resolution
    ("csize_embedder.mlp.0.weight", "adaln_single.emb.resolution_embedder.linear_1.weight"),
    ("csize_embedder.mlp.0.bias", "adaln_single.emb.resolution_embedder.linear_1.bias"),
    ("csize_embedder.mlp.2.weight", "adaln_single.emb.resolution_embedder.linear_2.weight"),
    ("csize_embedder.mlp.2.bias", "adaln_single.emb.resolution_embedder.linear_2.bias"),
    # Aspect ratio
    ("ar_embedder.mlp.0.weight", "adaln_single.emb.aspect_ratio_embedder.linear_1.weight"),
    ("ar_embedder.mlp.0.bias", "adaln_single.emb.aspect_ratio_embedder.linear_1.bias"),
    ("ar_embedder.mlp.2.weight", "adaln_single.emb.aspect_ratio_embedder.linear_2.weight"),
    ("ar_embedder.mlp.2.bias", "adaln_single.emb.aspect_ratio_embedder.linear_2.bias"),
]


def get_depth(state_dict):
    return sum(key.endswith('.attn1.to_k.bias') for key in state_dict.keys())


def get_lora_depth(state_dict):
    return sum(key.endswith('.attn1.to_k.lora_A.weight') for key in state_dict.keys())


def get_conversion_map(state_dict):
    conversion_map = [  # main SD conversion map (PixArt reference, HF Diffusers)
        # Patch embeddings
        ("x_embedder.proj.weight", "pos_embed.proj.weight"),
        ("x_embedder.proj.bias", "pos_embed.proj.bias"),
        # Caption projection
        ("y_embedder.y_embedding", "caption_projection.y_embedding"),
        ("y_embedder.y_proj.fc1.weight", "caption_projection.linear_1.weight"),
        ("y_embedder.y_proj.fc1.bias", "caption_projection.linear_1.bias"),
        ("y_embedder.y_proj.fc2.weight", "caption_projection.linear_2.weight"),
        ("y_embedder.y_proj.fc2.bias", "caption_projection.linear_2.bias"),
        # AdaLN-single LN
        ("t_embedder.mlp.0.weight", "adaln_single.emb.timestep_embedder.linear_1.weight"),
        ("t_embedder.mlp.0.bias", "adaln_single.emb.timestep_embedder.linear_1.bias"),
        ("t_embedder.mlp.2.weight", "adaln_single.emb.timestep_embedder.linear_2.weight"),
        ("t_embedder.mlp.2.bias", "adaln_single.emb.timestep_embedder.linear_2.bias"),
        # Shared norm
        ("t_block.1.weight", "adaln_single.linear.weight"),
        ("t_block.1.bias", "adaln_single.linear.bias"),
        # Final block
        ("final_layer.linear.weight", "proj_out.weight"),
        ("final_layer.linear.bias", "proj_out.bias"),
        ("final_layer.scale_shift_table", "scale_shift_table"),
    ]

    # Add actual transformer blocks
    for depth in range(get_depth(state_dict)):
        # Transformer blocks
        conversion_map += [
            (f"blocks.{depth}.scale_shift_table", f"transformer_blocks.{depth}.scale_shift_table"),
            # Projection
            (f"blocks.{depth}.attn.proj.weight", f"transformer_blocks.{depth}.attn1.to_out.0.weight"),
            (f"blocks.{depth}.attn.proj.bias", f"transformer_blocks.{depth}.attn1.to_out.0.bias"),
            # Feed-forward
            (f"blocks.{depth}.mlp.fc1.weight", f"transformer_blocks.{depth}.ff.net.0.proj.weight"),
            (f"blocks.{depth}.mlp.fc1.bias", f"transformer_blocks.{depth}.ff.net.0.proj.bias"),
            (f"blocks.{depth}.mlp.fc2.weight", f"transformer_blocks.{depth}.ff.net.2.weight"),
            (f"blocks.{depth}.mlp.fc2.bias", f"transformer_blocks.{depth}.ff.net.2.bias"),
            # Cross-attention (proj)
            (f"blocks.{depth}.cross_attn.proj.weight", f"transformer_blocks.{depth}.attn2.to_out.0.weight"),
            (f"blocks.{depth}.cross_attn.proj.bias", f"transformer_blocks.{depth}.attn2.to_out.0.bias"),
        ]
    return conversion_map


def find_prefix(state_dict, target_key):
    prefix = ""
    for k in state_dict.keys():
        if k.endswith(target_key):
            prefix = k.split(target_key)[0]
            break
    return prefix


def convert_state_dict(state_dict):
    if "adaln_single.emb.resolution_embedder.linear_1.weight" in state_dict.keys():
        cmap = get_conversion_map(state_dict) + conversion_map_ms
    else:
        cmap = get_conversion_map(state_dict)

    missing = [k for k, v in cmap if v not in state_dict]
    new_state_dict = {k: state_dict[v] for k, v in cmap if k not in missing}
    matched = list(v for k, v in cmap if v in state_dict.keys())

    for depth in range(get_depth(state_dict)):
        for wb in ["weight", "bias"]:
            # Self Attention
            key = lambda a: f"transformer_blocks.{depth}.attn1.to_{a}.{wb}"
            new_state_dict[f"blocks.{depth}.attn.qkv.{wb}"] = torch.cat((
                state_dict[key('q')], state_dict[key('k')], state_dict[key('v')]
            ), dim=0)
            matched += [key('q'), key('k'), key('v')]

            # Cross-attention (linear)
            key = lambda a: f"transformer_blocks.{depth}.attn2.to_{a}.{wb}"
            new_state_dict[f"blocks.{depth}.cross_attn.q_linear.{wb}"] = state_dict[key('q')]
            new_state_dict[f"blocks.{depth}.cross_attn.kv_linear.{wb}"] = torch.cat((
                state_dict[key('k')], state_dict[key('v')]
            ), dim=0)
            matched += [key('q'), key('k'), key('v')]

    if len(matched) < len(state_dict):
        print(f"PixArt: UNET conversion has leftover keys! ({len(matched)} vs {len(state_dict)})")
        print(list(set(state_dict.keys()) - set(matched)))

    if len(missing) > 0:
        print(f"PixArt: UNET conversion has missing keys!")
        print(missing)

    return new_state_dict


# Same as above but for LoRA weights:
def convert_lora_state_dict(state_dict, peft=True):
    # koyha
    rep_ak = lambda x: x.replace(".weight", ".lora_down.weight")
    rep_bk = lambda x: x.replace(".weight", ".lora_up.weight")
    rep_pk = lambda x: x.replace(".weight", ".alpha")
    if peft:  # peft
        rep_ap = lambda x: x.replace(".weight", ".lora_A.weight")
        rep_bp = lambda x: x.replace(".weight", ".lora_B.weight")
        rep_pp = lambda x: x.replace(".weight", ".alpha")

        prefix = find_prefix(state_dict, "adaln_single.linear.lora_A.weight")
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
    else:  # OneTrainer
        rep_ap = lambda x: x.replace(".", "_")[:-7] + ".lora_down.weight"
        rep_bp = lambda x: x.replace(".", "_")[:-7] + ".lora_up.weight"
        rep_pp = lambda x: x.replace(".", "_")[:-7] + ".alpha"

        prefix = "lora_transformer_"
        t5_marker = "lora_te_encoder"
        t5_keys = []
        for key in list(state_dict.keys()):
            if key.startswith(prefix):
                state_dict[key[len(prefix):]] = state_dict.pop(key)
            elif t5_marker in key:
                t5_keys.append(state_dict.pop(key))
        if len(t5_keys) > 0:
            print(f"Text Encoder not supported for PixArt LoRA, ignoring {len(t5_keys)} keys")

    cmap = []
    cmap_unet = get_conversion_map(state_dict) + conversion_map_ms  # todo: 512 model
    for k, v in cmap_unet:
        if v.endswith(".weight"):
            cmap.append((rep_ak(k), rep_ap(v)))
            cmap.append((rep_bk(k), rep_bp(v)))
            if not peft:
                cmap.append((rep_pk(k), rep_pp(v)))

    missing = [k for k, v in cmap if v not in state_dict]
    new_state_dict = {k: state_dict[v] for k, v in cmap if k not in missing}
    matched = list(v for k, v in cmap if v in state_dict.keys())

    lora_depth = get_lora_depth(state_dict)
    for fp, fk in ((rep_ap, rep_ak), (rep_bp, rep_bk)):
        for depth in range(lora_depth):
            # Self Attention
            key = lambda a: fp(f"transformer_blocks.{depth}.attn1.to_{a}.weight")
            new_state_dict[fk(f"blocks.{depth}.attn.qkv.weight")] = torch.cat((
                state_dict[key('q')], state_dict[key('k')], state_dict[key('v')]
            ), dim=0)

            matched += [key('q'), key('k'), key('v')]
            if not peft:
                akey = lambda a: rep_pp(f"transformer_blocks.{depth}.attn1.to_{a}.weight")
                new_state_dict[rep_pk((f"blocks.{depth}.attn.qkv.weight"))] = state_dict[akey("q")]
                matched += [akey('q'), akey('k'), akey('v')]

            # Self Attention projection?
            key = lambda a: fp(f"transformer_blocks.{depth}.attn1.to_{a}.weight")
            new_state_dict[fk(f"blocks.{depth}.attn.proj.weight")] = state_dict[key('out.0')]
            matched += [key('out.0')]

            # Cross-attention (linear)
            key = lambda a: fp(f"transformer_blocks.{depth}.attn2.to_{a}.weight")
            new_state_dict[fk(f"blocks.{depth}.cross_attn.q_linear.weight")] = state_dict[key('q')]
            new_state_dict[fk(f"blocks.{depth}.cross_attn.kv_linear.weight")] = torch.cat((
                state_dict[key('k')], state_dict[key('v')]
            ), dim=0)
            matched += [key('q'), key('k'), key('v')]
            if not peft:
                akey = lambda a: rep_pp(f"transformer_blocks.{depth}.attn2.to_{a}.weight")
                new_state_dict[rep_pk((f"blocks.{depth}.cross_attn.q_linear.weight"))] = state_dict[akey("q")]
                new_state_dict[rep_pk((f"blocks.{depth}.cross_attn.kv_linear.weight"))] = state_dict[akey("k")]
                matched += [akey('q'), akey('k'), akey('v')]

            # Cross Attention projection?
            key = lambda a: fp(f"transformer_blocks.{depth}.attn2.to_{a}.weight")
            new_state_dict[fk(f"blocks.{depth}.cross_attn.proj.weight")] = state_dict[key('out.0')]
            matched += [key('out.0')]

            key = fp(f"transformer_blocks.{depth}.ff.net.0.proj.weight")
            new_state_dict[fk(f"blocks.{depth}.mlp.fc1.weight")] = state_dict[key]
            matched += [key]

            key = fp(f"transformer_blocks.{depth}.ff.net.2.weight")
            new_state_dict[fk(f"blocks.{depth}.mlp.fc2.weight")] = state_dict[key]
            matched += [key]

    if len(matched) < len(state_dict):
        print(f"PixArt: LoRA conversion has leftover keys! ({len(matched)} vs {len(state_dict)})")
        print(list(set(state_dict.keys()) - set(matched)))

    if len(missing) > 0:
        print(f"PixArt: LoRA conversion has missing keys! (probably)")
        print(missing)

    return new_state_dict