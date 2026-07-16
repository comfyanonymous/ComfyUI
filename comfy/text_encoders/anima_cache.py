import contextvars
import logging
import numbers
import weakref

import torch


_cache_scope = contextvars.ContextVar("anima_prefix_cache", default=None)


def begin_cache_scope(enabled=True):
    cache = weakref.WeakKeyDictionary() if enabled else None
    return cache, _cache_scope.set(cache)


def end_cache_scope(scope):
    cache, token = scope
    try:
        if cache is not None:
            cache.clear()
    finally:
        _cache_scope.reset(token)


def set_owner(model, owner):
    object.__setattr__(model, "_anima_cache_owner", weakref.ref(owner))


def get_owner(model):
    owner = getattr(model, "_anima_cache_owner", None)
    return owner() if owner is not None else None


def _token_ids(tokens):
    if len(tokens) != 1:
        return None
    sequence = tokens[0]
    if torch.is_tensor(sequence):
        sequence = sequence.tolist()
    token_ids = []
    for token in sequence:
        if isinstance(token, numbers.Integral):
            token_ids.append(int(token))
        elif isinstance(token, (tuple, list)) and len(token) == 2 and isinstance(token[0], numbers.Integral) and isinstance(token[1], numbers.Real) and token[1] == 1:
            token_ids.append(int(token[0]))
        else:
            return None
    return tuple(token_ids)


def _common_prefix(first, second):
    length = min(len(first), len(second))
    for index in range(length):
        if first[index] != second[index]:
            return index
    return length


def _copy_key_values(key_values, length):
    return [(key[:, :, :length].clone(), value[:, :, :length].clone(), length) for key, value, _ in key_values]


def forward(transformer, cache_owner, tokens, attention_mask, embeds, num_tokens, intermediate_output, final_layer_norm_intermediate, dtype, embeds_info):
    prefix_cache = _cache_scope.get()
    token_ids = _token_ids(tokens)
    weight_uuid = getattr(cache_owner, "current_weight_patches_uuid", None)
    if prefix_cache is None or cache_owner is None or weight_uuid is None or token_ids is None or embeds_info or intermediate_output is not None or attention_mask is not None and not bool(torch.all(attention_mask)):
        return transformer(None, attention_mask, embeds=embeds, num_tokens=num_tokens, intermediate_output=intermediate_output, final_layer_norm_intermediate=final_layer_norm_intermediate, dtype=dtype, embeds_info=embeds_info)

    cached = prefix_cache.get(cache_owner)
    if cached is not None and (cached[0]() is not transformer or cached[1] != weight_uuid or cached[3].device != embeds.device or cached[3].dtype != embeds.dtype):
        cached = None
    common = _common_prefix(cached[2], token_ids) if cached is not None else 0
    if common > 0:
        logging.debug("Anima Qwen cache reused %d prefix tokens", common)
    if cached is not None and common == len(token_ids) == len(cached[2]):
        return cached[3].clone(), None

    if cached is not None and common == len(token_ids):
        hidden = cached[3][:, :common].clone()
        prefix_cache[cache_owner] = (weakref.ref(transformer), weight_uuid, token_ids, hidden.clone(), _copy_key_values(cached[4], common))
        return hidden, None

    past_key_values = []
    prefix_hidden = None
    if cached is not None and common > 0:
        prefix_hidden = cached[3][:, :common].clone()
        past_key_values = _copy_key_values(cached[4], common)

    suffix_embeds = embeds[:, common:]
    if common > 0 and suffix_embeds.shape[1] > 1:
        suffix_outputs = []
        next_key_values = past_key_values
        for index in range(suffix_embeds.shape[1]):
            output = transformer(
                None,
                None,
                embeds=suffix_embeds[:, index:index + 1],
                num_tokens=[1],
                intermediate_output=intermediate_output,
                final_layer_norm_intermediate=final_layer_norm_intermediate,
                dtype=dtype,
                embeds_info=embeds_info,
                past_key_values=next_key_values,
            )
            suffix_outputs.append(output[0])
            next_key_values = output[2]
        suffix_hidden = torch.cat(suffix_outputs, dim=1)
    else:
        output = transformer(
            None,
            attention_mask,
            embeds=suffix_embeds,
            num_tokens=[1] if common > 0 else num_tokens,
            intermediate_output=intermediate_output,
            final_layer_norm_intermediate=final_layer_norm_intermediate,
            dtype=dtype,
            embeds_info=embeds_info,
            past_key_values=past_key_values,
        )
        suffix_hidden = output[0]
        next_key_values = output[2]
    hidden = suffix_hidden if prefix_hidden is None else torch.cat((prefix_hidden, suffix_hidden), dim=1)
    prefix_cache[cache_owner] = (weakref.ref(transformer), weight_uuid, token_ids, hidden.clone(), _copy_key_values(next_key_values, len(token_ids)))
    return hidden, None
