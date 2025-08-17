import torch
import numpy as np
import re
import itertools

from comfy import model_management
from comfy.sdxl_clip import SDXLClipModel, SDXLRefinerClipModel, SDXLClipG
try:
    from comfy.text_encoders.sd3_clip import SD3ClipModel, T5XXLModel
except ImportError:
    from comfy.sd3_clip import SD3ClipModel, T5XXLModel

from nodes import NODE_CLASS_MAPPINGS, ConditioningConcat, ConditioningZeroOut, ConditioningSetTimestepRange, ConditioningCombine

def _grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def _norm_mag(w, n):
    d = w - 1
    return 1 + np.sign(d) * np.sqrt(np.abs(d) ** 2 / n)
    # return  np.sign(w) * np.sqrt(np.abs(w)**2 / n)


def divide_length(word_ids, weights):
    sums = dict(zip(*np.unique(word_ids, return_counts=True)))
    sums[0] = 1
    weights = [[_norm_mag(w, sums[id]) if id != 0 else 1.0
                for w, id in zip(x, y)] for x, y in zip(weights, word_ids)]
    return weights


def shift_mean_weight(word_ids, weights):
    delta = 1 - np.mean([w for x, y in zip(weights, word_ids) for w, id in zip(x, y) if id != 0])
    weights = [[w if id == 0 else w + delta
                for w, id in zip(x, y)] for x, y in zip(weights, word_ids)]
    return weights


def scale_to_norm(weights, word_ids, w_max):
    top = np.max(weights)
    w_max = min(top, w_max)
    weights = [[w_max if id == 0 else (w / top) * w_max
                for w, id in zip(x, y)] for x, y in zip(weights, word_ids)]
    return weights


def from_zero(weights, base_emb):
    weight_tensor = torch.tensor(weights, dtype=base_emb.dtype, device=base_emb.device)
    weight_tensor = weight_tensor.reshape(1, -1, 1).expand(base_emb.shape)
    return base_emb * weight_tensor


def mask_word_id(tokens, word_ids, target_id, mask_token):
    new_tokens = [[mask_token if wid == target_id else t
                   for t, wid in zip(x, y)] for x, y in zip(tokens, word_ids)]
    mask = np.array(word_ids) == target_id
    return (new_tokens, mask)


def batched_clip_encode(tokens, length, encode_func, num_chunks):
    embs = []
    for e in _grouper(32, tokens):
        enc, pooled = encode_func(e)
        enc = enc.reshape((len(e), length, -1))
        embs.append(enc)
    embs = torch.cat(embs)
    embs = embs.reshape((len(tokens) // num_chunks, length * num_chunks, -1))
    return embs


def from_masked(tokens, weights, word_ids, base_emb, length, encode_func, m_token=266):
    pooled_base = base_emb[0, length - 1:length, :]
    wids, inds = np.unique(np.array(word_ids).reshape(-1), return_index=True)
    weight_dict = dict((id, w)
                       for id, w in zip(wids, np.array(weights).reshape(-1)[inds])
                       if w != 1.0)

    if len(weight_dict) == 0:
        return torch.zeros_like(base_emb), base_emb[0, length - 1:length, :]

    weight_tensor = torch.tensor(weights, dtype=base_emb.dtype, device=base_emb.device)
    weight_tensor = weight_tensor.reshape(1, -1, 1).expand(base_emb.shape)

    # m_token = (clip.tokenizer.end_token, 1.0) if  clip.tokenizer.pad_with_end else (0,1.0)
    # TODO: find most suitable masking token here
    m_token = (m_token, 1.0)

    ws = []
    masked_tokens = []
    masks = []

    # create prompts
    for id, w in weight_dict.items():
        masked, m = mask_word_id(tokens, word_ids, id, m_token)
        masked_tokens.extend(masked)

        m = torch.tensor(m, dtype=base_emb.dtype, device=base_emb.device)
        m = m.reshape(1, -1, 1).expand(base_emb.shape)
        masks.append(m)

        ws.append(w)

    # batch process prompts
    embs = batched_clip_encode(masked_tokens, length, encode_func, len(tokens))
    masks = torch.cat(masks)

    embs = (base_emb.expand(embs.shape) - embs)
    pooled = embs[0, length - 1:length, :]

    embs *= masks
    embs = embs.sum(axis=0, keepdim=True)

    pooled_start = pooled_base.expand(len(ws), -1)
    ws = torch.tensor(ws).reshape(-1, 1).expand(pooled_start.shape)
    pooled = (pooled - pooled_start) * (ws - 1)
    pooled = pooled.mean(axis=0, keepdim=True)

    return ((weight_tensor - 1) * embs), pooled_base + pooled


def mask_inds(tokens, inds, mask_token):
    clip_len = len(tokens[0])
    inds_set = set(inds)
    new_tokens = [[mask_token if i * clip_len + j in inds_set else t
                   for j, t in enumerate(x)] for i, x in enumerate(tokens)]
    return new_tokens


def down_weight(tokens, weights, word_ids, base_emb, length, encode_func, m_token=266):
    w, w_inv = np.unique(weights, return_inverse=True)

    if np.sum(w < 1) == 0:
        return base_emb, tokens, base_emb[0, length - 1:length, :]
    # m_token = (clip.tokenizer.end_token, 1.0) if  clip.tokenizer.pad_with_end else (0,1.0)
    # using the comma token as a masking token seems to work better than aos tokens for SD 1.x
    m_token = (m_token, 1.0)

    masked_tokens = []

    masked_current = tokens
    for i in range(len(w)):
        if w[i] >= 1:
            continue
        masked_current = mask_inds(masked_current, np.where(w_inv == i)[0], m_token)
        masked_tokens.extend(masked_current)

    embs = batched_clip_encode(masked_tokens, length, encode_func, len(tokens))
    embs = torch.cat([base_emb, embs])
    w = w[w <= 1.0]
    w_mix = np.diff([0] + w.tolist())
    w_mix = torch.tensor(w_mix, dtype=embs.dtype, device=embs.device).reshape((-1, 1, 1))

    weighted_emb = (w_mix * embs).sum(axis=0, keepdim=True)
    return weighted_emb, masked_current, weighted_emb[0, length - 1:length, :]


def scale_emb_to_mag(base_emb, weighted_emb):
    norm_base = torch.linalg.norm(base_emb)
    norm_weighted = torch.linalg.norm(weighted_emb)
    embeddings_final = (norm_base / norm_weighted) * weighted_emb
    return embeddings_final


def recover_dist(base_emb, weighted_emb):
    fixed_std = (base_emb.std() / weighted_emb.std()) * (weighted_emb - weighted_emb.mean())
    embeddings_final = fixed_std + (base_emb.mean() - fixed_std.mean())
    return embeddings_final


def A1111_renorm(base_emb, weighted_emb):
    embeddings_final = (base_emb.mean() / weighted_emb.mean()) * weighted_emb
    return embeddings_final


def advanced_encode_from_tokens(tokenized, token_normalization, weight_interpretation, encode_func, m_token=266,
                                length=77, w_max=1.0, return_pooled=False, apply_to_pooled=False):
    tokens = [[t for t, _, _ in x] for x in tokenized]
    weights = [[w for _, w, _ in x] for x in tokenized]
    word_ids = [[wid for _, _, wid in x] for x in tokenized]

    # weight normalization
    # ====================

    # distribute down/up weights over word lengths
    if token_normalization.startswith("length"):
        weights = divide_length(word_ids, weights)

    # make mean of word tokens 1
    if token_normalization.endswith("mean"):
        weights = shift_mean_weight(word_ids, weights)

        # weight interpretation
    # =====================
    pooled = None

    if weight_interpretation == "comfy":
        weighted_tokens = [[(t, w) for t, w in zip(x, y)] for x, y in zip(tokens, weights)]
        weighted_emb, pooled_base = encode_func(weighted_tokens)
        pooled = pooled_base
    else:
        unweighted_tokens = [[(t, 1.0) for t, _, _ in x] for x in tokenized]
        base_emb, pooled_base = encode_func(unweighted_tokens)

    if weight_interpretation == "A1111":
        weighted_emb = from_zero(weights, base_emb)
        weighted_emb = A1111_renorm(base_emb, weighted_emb)
        pooled = pooled_base

    if weight_interpretation == "compel":
        pos_tokens = [[(t, w) if w >= 1.0 else (t, 1.0) for t, w in zip(x, y)] for x, y in zip(tokens, weights)]
        weighted_emb, _ = encode_func(pos_tokens)
        weighted_emb, _, pooled = down_weight(pos_tokens, weights, word_ids, weighted_emb, length, encode_func)

    if weight_interpretation == "comfy++":
        weighted_emb, tokens_down, _ = down_weight(unweighted_tokens, weights, word_ids, base_emb, length, encode_func)
        weights = [[w if w > 1.0 else 1.0 for w in x] for x in weights]
        # unweighted_tokens = [[(t,1.0) for t, _,_ in x] for x in tokens_down]
        embs, pooled = from_masked(unweighted_tokens, weights, word_ids, base_emb, length, encode_func)
        weighted_emb += embs

    if weight_interpretation == "down_weight":
        weights = scale_to_norm(weights, word_ids, w_max)
        weighted_emb, _, pooled = down_weight(unweighted_tokens, weights, word_ids, base_emb, length, encode_func)

    if return_pooled:
        if apply_to_pooled:
            return weighted_emb, pooled
        else:
            return weighted_emb, pooled_base
    return weighted_emb, None


def encode_token_weights_g(model, token_weight_pairs):
    return model.clip_g.encode_token_weights(token_weight_pairs)


def encode_token_weights_l(model, token_weight_pairs):
    l_out, pooled = model.clip_l.encode_token_weights(token_weight_pairs)
    return l_out, pooled

def encode_token_weights_t5(model, token_weight_pairs):
    return model.t5xxl.encode_token_weights(token_weight_pairs)


def encode_token_weights(model, token_weight_pairs, encode_func):
    if model.layer_idx is not None:
        # 2016 [c2cb8e88] 及以上版本去除了sdxl clip的clip_layer方法
        # if compare_revision(2016):
        model.cond_stage_model.set_clip_options({'layer': model.layer_idx})
        # else:
        # model.cond_stage_model.clip_layer(model.layer_idx)

    model_management.load_model_gpu(model.patcher)
    return encode_func(model.cond_stage_model, token_weight_pairs)

def prepareXL(embs_l, embs_g, pooled, clip_balance):
    l_w = 1 - max(0, clip_balance - .5) * 2
    g_w = 1 - max(0, .5 - clip_balance) * 2
    if embs_l is not None:
        return torch.cat([embs_l * l_w, embs_g * g_w], dim=-1), pooled
    else:
        return embs_g, pooled

def prepareSD3(out, pooled, clip_balance):
    lg_w = 1 - max(0, clip_balance - .5) * 2
    t5_w = 1 - max(0, .5 - clip_balance) * 2
    if out.shape[0] > 1:
        return torch.cat([out[0] * lg_w, out[1] * t5_w], dim=-1), pooled
    else:
        return out, pooled

def advanced_encode(clip, text, token_normalization, weight_interpretation, w_max=1.0, clip_balance=.5,
                    apply_to_pooled=True, width=1024, height=1024, crop_w=0, crop_h=0, target_width=1024, target_height=1024, a1111_prompt_style=False, steps=1):

    # Use clip text encode by smzNodes like same as a1111, when if you need installed the smzNodes
    if a1111_prompt_style:
        if "smZ CLIPTextEncode" in NODE_CLASS_MAPPINGS:
            cls = NODE_CLASS_MAPPINGS['smZ CLIPTextEncode']
            embeddings_final, = cls().encode(clip, text, weight_interpretation, True, True, False, False, 6, 1024, 1024, 0, 0, 1024, 1024, '', '', steps)
            return embeddings_final
        else:
            raise Exception(f"[smzNodes Not Found] you need to install 'ComfyUI-smzNodes'")

    time_start = 0
    time_end = 1
    match = re.search(r'TIMESTEP.*$', text)
    if match:
        timestep = match.group()
        timestep = timestep.split(' ')
        timestep = timestep[0]
        text = text.replace(timestep, '')
        value = timestep.split(':')
        if len(value) >= 3:
            time_start = float(value[1])
            time_end = float(value[2])
        elif len(value) == 2:
            time_start = float(value[1])
            time_end = 1
        elif len(value) == 1:
            time_start = 0.1
            time_end = 1

    pass3 = [x.strip() for x in text.split("BREAK")]
    pass3 = [x for x in pass3 if x != '']

    if len(pass3) == 0:
        pass3 = ['']

    # pass3_str = [f'[{x}]' for x in pass3]
    # print(f"CLIP: {str.join(' + ', pass3_str)}")

    conditioning = None

    for text in pass3:
        tokenized = clip.tokenize(text, return_word_ids=True)
        if SD3ClipModel and isinstance(clip.cond_stage_model, SD3ClipModel):
            lg_out = None
            pooled = None
            out = None

            if len(tokenized['l']) > 0 or len(tokenized['g']) > 0:
                if clip.cond_stage_model.clip_l is not None:
                    lg_out, l_pooled = advanced_encode_from_tokens(tokenized['l'],
                                                                           token_normalization,
                                                                           weight_interpretation,
                                                                           lambda x: encode_token_weights(clip, x, encode_token_weights_l),
                                                                           w_max=w_max, return_pooled=True,)
                else:
                    l_pooled = torch.zeros((1, 768), device=model_management.intermediate_device())

                if clip.cond_stage_model.clip_g is not None:
                    g_out, g_pooled = advanced_encode_from_tokens(tokenized['g'],
                               token_normalization,
                               weight_interpretation,
                               lambda x: encode_token_weights(clip, x, encode_token_weights_g),
                               w_max=w_max, return_pooled=True)
                    if lg_out is not None:
                        lg_out = torch.cat([lg_out, g_out], dim=-1)
                    else:
                        lg_out = torch.nn.functional.pad(g_out, (768, 0))
                else:
                    g_out = None
                    g_pooled = torch.zeros((1, 1280), device=model_management.intermediate_device())

                if lg_out is not None:
                    lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
                    out = lg_out
                pooled = torch.cat((l_pooled, g_pooled), dim=-1)

            # t5xxl
            if 't5xxl' in tokenized:
                t5_out, t5_pooled = advanced_encode_from_tokens(tokenized['t5xxl'],
                               token_normalization,
                               weight_interpretation,
                               lambda x: encode_token_weights(clip, x, encode_token_weights_t5),
                               w_max=w_max, return_pooled=True)
                if lg_out is not None:
                    out = torch.cat([lg_out, t5_out], dim=-2)
                else:
                    out = t5_out

            if out is None:
                out = torch.zeros((1, 77, 4096), device=model_management.intermediate_device())

            if pooled is None:
                pooled = torch.zeros((1, 768 + 1280), device=model_management.intermediate_device())

            embeddings_final, pooled = prepareSD3(out, pooled, clip_balance)
            cond = [[embeddings_final, {"pooled_output": pooled}]]

        elif isinstance(clip.cond_stage_model, (SDXLClipModel, SDXLRefinerClipModel, SDXLClipG)):
            embs_l = None
            embs_g = None
            pooled = None
            if 'l' in tokenized and isinstance(clip.cond_stage_model, SDXLClipModel):
                embs_l, _ = advanced_encode_from_tokens(tokenized['l'],
                                                        token_normalization,
                                                        weight_interpretation,
                                                        lambda x: encode_token_weights(clip, x, encode_token_weights_l),
                                                        w_max=w_max,
                                                        return_pooled=False)
            if 'g' in tokenized:
                embs_g, pooled = advanced_encode_from_tokens(tokenized['g'],
                                                             token_normalization,
                                                             weight_interpretation,
                                                             lambda x: encode_token_weights(clip, x,
                                                                                            encode_token_weights_g),
                                                             w_max=w_max,
                                                             return_pooled=True,
                                                             apply_to_pooled=apply_to_pooled)

            embeddings_final, pooled = prepareXL(embs_l, embs_g, pooled, clip_balance)

            cond = [[embeddings_final, {"pooled_output": pooled}]]
            # cond = [[embeddings_final,
            #                  {"pooled_output": pooled, "width": width, "height": height, "crop_w": crop_w,
            #                   "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]]
        else:
            embeddings_final, pooled = advanced_encode_from_tokens(tokenized['l'],
                                                                   token_normalization,
                                                                   weight_interpretation,
                                                                   lambda x: encode_token_weights(clip, x, encode_token_weights_l),
                                                                   w_max=w_max,return_pooled=True,)
            cond = [[embeddings_final, {"pooled_output": pooled}]]

        if conditioning is not None:
            conditioning = ConditioningConcat().concat(conditioning, cond)[0]
        else:
            conditioning = cond

    # setTimeStepRange
    if time_start > 0 or time_end < 1:
        conditioning_2, = ConditioningSetTimestepRange().set_range(conditioning, 0, time_start)
        conditioning_1, = ConditioningZeroOut().zero_out(conditioning)
        conditioning_1, = ConditioningSetTimestepRange().set_range(conditioning_1, time_start, time_end)
        conditioning, = ConditioningCombine().combine(conditioning_1, conditioning_2)

    return conditioning



