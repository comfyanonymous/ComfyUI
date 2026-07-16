import asyncio

import pytest
import torch

from comfy.text_encoders import anima_cache


def tokens(*ids):
    return [list(ids)]


class TinyCausalTransformer:
    def __init__(self):
        self.calls = []
        self.num_tokens = []

    def __call__(self, _, attention_mask, embeds, num_tokens, intermediate_output, final_layer_norm_intermediate, dtype, embeds_info, past_key_values=None):
        self.calls.append((embeds.shape[1], past_key_values is not None))
        self.num_tokens.append(tuple(num_tokens))
        if past_key_values and embeds.shape[1] > 1:
            raise RuntimeError("cached multi-token suffix would use an invalid causal mask")
        prefix = 0
        if past_key_values:
            prefix = past_key_values[0][2]
            previous = past_key_values[0][0][:, :, :prefix].reshape(embeds.shape[0], prefix, 1)
        else:
            previous = embeds[:, :0]
        sequence = torch.cat((previous, embeds), dim=1)
        hidden = sequence.cumsum(dim=1)[:, prefix:]
        key = sequence.reshape(sequence.shape[0], 1, sequence.shape[1], 1).clone()
        value = (sequence * 2).reshape(sequence.shape[0], 1, sequence.shape[1], 1).clone()
        if past_key_values is None:
            return hidden, None
        return hidden, None, [(key, value, sequence.shape[1])]


class CacheOwner:
    def __init__(self, weight_uuid="weights-a"):
        self.current_weight_patches_uuid = weight_uuid


def cached_forward(transformer, token_ids, dtype=torch.float32, attention_mask=None, intermediate_output=None, embeds_info=None, owner=None):
    embeds = torch.tensor(token_ids, dtype=dtype).reshape(1, -1, 1)
    owner = owner or CacheOwner()
    return anima_cache.forward(transformer, owner, tokens(*token_ids), attention_mask, embeds, [len(token_ids)], intermediate_output, False, torch.float32, embeds_info or [])


@pytest.fixture
def prefix_cache():
    scope = anima_cache.begin_cache_scope()
    try:
        yield scope[0]
    finally:
        anima_cache.end_cache_scope(scope)


def test_token_ids_accept_runtime_lists_tensors_and_unit_weight_legacy_pairs():
    assert anima_cache._token_ids([[1, 2, 3]]) == (1, 2, 3)
    assert anima_cache._token_ids(torch.tensor([[1, 2, 3]])) == (1, 2, 3)
    assert anima_cache._token_ids([[(1, 1.0), (2, 1)]]) == (1, 2)
    assert anima_cache._token_ids([[(1, 0.5), (2, 1.0)]]) is None
    assert anima_cache._token_ids([[(1, 1.0, "custom")]]) is None
    assert anima_cache._token_ids([[{"type": "embedding"}]]) is None


def test_non_unit_legacy_weight_bypasses_cache(prefix_cache):
    transformer = TinyCausalTransformer()
    owner = CacheOwner()
    embeds = torch.tensor([1.0, 2.0]).reshape(1, -1, 1)
    weighted_tokens = [[(1, 0.5), (2, 1.0)]]

    anima_cache.forward(transformer, owner, weighted_tokens, None, embeds, [2], None, False, torch.float32, [])
    anima_cache.forward(transformer, owner, weighted_tokens, None, embeds, [2], None, False, torch.float32, [])

    assert transformer.calls == [(2, False), (2, False)]
    assert len(prefix_cache) == 0


def test_reuses_prefix_for_extension_exactly(caplog, prefix_cache):
    transformer = TinyCausalTransformer()
    owner = CacheOwner()
    cached_forward(transformer, [1, 2, 3], owner=owner)
    with caplog.at_level("DEBUG"):
        output, _ = cached_forward(transformer, [1, 2, 3, 4], owner=owner)

    assert torch.equal(output, torch.tensor([[[1.0], [3.0], [6.0], [10.0]]]))
    assert transformer.calls == [(3, True), (1, True)]
    assert transformer.num_tokens == [(3,), (1,)]
    assert "Anima Qwen cache reused 3 prefix tokens" in caplog.messages


def test_exact_hit_does_not_call_transformer(prefix_cache):
    transformer = TinyCausalTransformer()
    owner = CacheOwner()
    expected, _ = cached_forward(transformer, [1, 2, 3], owner=owner)
    call_count = len(transformer.calls)

    output, _ = cached_forward(transformer, [1, 2, 3], owner=owner)

    assert torch.equal(output, expected)
    assert len(transformer.calls) == call_count


def test_strict_prefix_of_cached_prompt_is_copied_without_transformer_call(prefix_cache):
    transformer = TinyCausalTransformer()
    owner = CacheOwner()
    cached_forward(transformer, [1, 2, 3, 4], owner=owner)
    output, _ = cached_forward(transformer, [1, 2], owner=owner)

    assert torch.equal(output, torch.tensor([[[1.0], [3.0]]]))
    assert transformer.calls == [(4, True)]
    cached = prefix_cache[owner]
    assert cached[2] == (1, 2)
    assert cached[3]._base is None
    assert all(key.shape[2] == value.shape[2] == index == 2 for key, value, index in cached[4])
    assert all(key._base is None and value._base is None for key, value, _ in cached[4])


def test_diverging_suffix_matches_full_causal_forward(prefix_cache):
    transformer = TinyCausalTransformer()
    owner = CacheOwner()
    cached_forward(transformer, [1, 2, 9], owner=owner)
    output, _ = cached_forward(transformer, [1, 2, 4, 5], owner=owner)

    assert torch.equal(output, torch.tensor([[[1.0], [3.0], [7.0], [12.0]]]))
    assert transformer.calls == [(3, True), (1, True), (1, True)]
    assert transformer.num_tokens == [(3,), (1,), (1,)]


def test_cache_is_isolated_by_owner_and_transformer_identity(prefix_cache):
    shared_transformer = TinyCausalTransformer()
    first_owner = CacheOwner("same-uuid")
    second_owner = CacheOwner("same-uuid")
    cached_forward(shared_transformer, [1, 2], owner=first_owner)
    cached_forward(shared_transformer, [1, 2, 3], owner=second_owner)
    assert shared_transformer.calls == [(2, True), (3, True)]

    replacement = TinyCausalTransformer()
    cached_forward(replacement, [1, 2, 3], owner=first_owner)
    assert replacement.calls == [(3, True)]


def test_scope_end_clears_cached_prefixes_and_disables_reuse():
    transformer = TinyCausalTransformer()
    owner = CacheOwner()
    scope = anima_cache.begin_cache_scope()
    cache = scope[0]
    try:
        cached_forward(transformer, [1, 2], owner=owner)
        assert len(cache) == 1
    finally:
        anima_cache.end_cache_scope(scope)

    assert len(cache) == 0
    cached_forward(transformer, [1, 2, 3], owner=owner)
    assert transformer.calls[-1] == (3, False)
    assert len(cache) == 0


def test_child_tasks_share_the_same_cache_scope():
    transformer = TinyCausalTransformer()
    owner = CacheOwner()

    async def run():
        scope = anima_cache.begin_cache_scope()
        primed = asyncio.Event()

        async def prime():
            cached_forward(transformer, [1, 2], owner=owner)
            primed.set()

        async def reuse():
            await primed.wait()
            return cached_forward(transformer, [1, 2, 3], owner=owner)

        try:
            first = asyncio.create_task(prime())
            second = asyncio.create_task(reuse())
            await first
            return await second
        finally:
            anima_cache.end_cache_scope(scope)

    output, _ = asyncio.run(run())

    assert torch.equal(output, torch.tensor([[[1.0], [3.0], [6.0]]]))
    assert transformer.calls == [(2, True), (1, True)]


def test_nested_cache_scopes_do_not_share_prefixes():
    transformer = TinyCausalTransformer()
    owner = CacheOwner()
    outer_scope = anima_cache.begin_cache_scope()
    outer_cache = outer_scope[0]
    try:
        cached_forward(transformer, [1, 2], owner=owner)
        inner_scope = anima_cache.begin_cache_scope()
        inner_cache = inner_scope[0]
        try:
            cached_forward(transformer, [1, 2, 3], owner=owner)
            assert len(inner_cache) == 1
        finally:
            anima_cache.end_cache_scope(inner_scope)

        assert len(inner_cache) == 0
        cached_forward(transformer, [1, 2, 4], owner=owner)
        assert len(outer_cache) == 1
    finally:
        anima_cache.end_cache_scope(outer_scope)

    assert transformer.calls == [(2, True), (3, True), (1, True)]


def test_disabled_scope_hides_an_outer_cache_scope():
    transformer = TinyCausalTransformer()
    owner = CacheOwner()
    outer_scope = anima_cache.begin_cache_scope()
    try:
        cached_forward(transformer, [1, 2], owner=owner)
        disabled_scope = anima_cache.begin_cache_scope(False)
        try:
            cached_forward(transformer, [1, 2], owner=owner)
        finally:
            anima_cache.end_cache_scope(disabled_scope)

        call_count = len(transformer.calls)
        cached_forward(transformer, [1, 2], owner=owner)
        assert len(transformer.calls) == call_count
    finally:
        anima_cache.end_cache_scope(outer_scope)

    assert transformer.calls == [(2, True), (2, False)]


def test_unsafe_inputs_and_changed_model_state_do_not_reuse_cache(prefix_cache):
    transformer = TinyCausalTransformer()
    owner = CacheOwner()
    cached_forward(transformer, [1, 2], attention_mask=torch.tensor([[True, False]]), owner=owner)
    cached_forward(transformer, [1, 2], intermediate_output=0, owner=owner)
    cached_forward(transformer, [1, 2], embeds_info=[{"custom": True}], owner=owner)
    assert transformer.calls == [(2, False), (2, False), (2, False)]
    assert len(prefix_cache) == 0

    cached_forward(transformer, [1, 2], dtype=torch.float32, owner=owner)
    cached_forward(transformer, [1, 2, 3], dtype=torch.float64, owner=owner)
    owner.current_weight_patches_uuid = "weights-b"
    cached_forward(transformer, [1, 2, 3, 4], dtype=torch.float64, owner=owner)
    assert transformer.calls[-3:] == [(2, True), (3, True), (4, True)]


def test_cache_owner_reference_is_not_registered_as_child_module():
    owner = torch.nn.Module()
    child = torch.nn.Module()
    owner.add_module("child", child)

    anima_cache.set_owner(child, owner)

    assert anima_cache.get_owner(child) is owner
    assert "_anima_cache_owner" not in child._modules
    assert list(owner.state_dict()) == []
