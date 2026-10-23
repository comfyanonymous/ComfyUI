"""OBL-089 spike: the LLM prefix cache with images in the prompt (cache ids from image content, MRoPE positions)."""
import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ops  # noqa: E402
from comfy.text_encoders import llm_prefix_cache  # noqa: E402
from comfy.text_encoders.qwen_vl import qwen2vl_mrope_position_ids  # noqa: E402

HIDDEN = 64
IMAGE_SIZE = 16  # positions per image span: an 8x8 grid merged 2x2


def _image(seed):
    return {"type": "image", "data": torch.rand(1, 32, 32, 3, generator=torch.Generator().manual_seed(seed))}


def _info(index):
    return {"type": "image", "index": index, "size": IMAGE_SIZE, "extra": torch.tensor([[1, 8, 8]])}


def test_prompt_ids_hash_each_image_position():
    a, b = _image(1), _image(2)
    tokens = [5, 6, a, 7, b, 8]
    info = [_info(2), _info(3 + IMAGE_SIZE)]
    ids = llm_prefix_cache.prompt_ids(tokens, info, 4 + 2 * IMAGE_SIZE)
    assert ids[:2] == [5, 6] and ids[2 + IMAGE_SIZE] == 7 and ids[-1] == 8
    span_a, span_b = ids[2:2 + IMAGE_SIZE], ids[3 + IMAGE_SIZE:3 + 2 * IMAGE_SIZE]
    assert len(set(span_a)) == IMAGE_SIZE and span_a[0][0] != span_b[0][0]
    # the same image content (a fresh tensor) gives the same ids; a changed pixel does not
    again = llm_prefix_cache.prompt_ids([5, 6, {"type": "image", "data": a["data"].clone()}, 7, b, 8], info, 4 + 2 * IMAGE_SIZE)
    assert again == ids
    changed = a["data"].clone()
    changed[0, 0, 0, 0] += 0.5
    assert llm_prefix_cache.prompt_ids([5, 6, {"type": "image", "data": changed}, 7, b, 8], info, 4 + 2 * IMAGE_SIZE)[2] != ids[2]


@pytest.mark.parametrize("case", ["video", "misaligned", "short", "extra-span"])
def test_prompt_ids_refuse_what_they_cannot_key(case):
    a = _image(1)
    tokens, info, seq = [5, a, 6], [_info(1)], 2 + IMAGE_SIZE
    if case == "video":
        tokens = [5, {"type": "video", "data": a["data"]}, 6]
    elif case == "misaligned":
        info = [_info(2)]
    elif case == "short":
        seq += 1
    elif case == "extra-span":
        info = [_info(1), _info(3 + IMAGE_SIZE)]
    assert llm_prefix_cache.prompt_ids(tokens, info, seq) is None


def _tiny_model(monkeypatch, layers=4):
    import comfy.text_encoders.qwen35 as qwen35
    monkeypatch.setitem(qwen35.QWEN35_MODELS, "tiny", dict(vision=dict(hidden_size=32, num_heads=2, intermediate_size=64, depth=1)))

    class Tiny(qwen35.Qwen35):
        model_type = "tiny"
    config = dict(vocab_size=97, hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=layers, layer_types=qwen35._qwen35_layer_types(layers),
                  num_attention_heads=2, num_key_value_heads=1, head_dim=64, linear_num_key_heads=2, linear_num_value_heads=2,
                  linear_key_head_dim=16, linear_value_head_dim=16, mtp=True, lm_head=True, stop_tokens=[],
                  rope_theta=100.0, mrope_section=[3, 3, 2])  # 16 rotary dims, fast enough to tell positions apart
    torch.manual_seed(0)
    model = Tiny(config, torch.float32, "cpu", comfy.ops.manual_cast)
    model.requires_grad_(False)
    for name, p in model.named_parameters():
        p.data = torch.randn_like(p) * (0.05 if "norm" in name else 0.3)
    return model


MSG_END = [90, 91]  # stands in for "<|im_end|>\n", outside the tiny vocab


def _cache_on(monkeypatch, chunk=0):
    monkeypatch.setattr(llm_prefix_cache, "MODE", "vram")
    monkeypatch.setattr(llm_prefix_cache, "MSG_END", tuple(MSG_END))
    monkeypatch.setattr(llm_prefix_cache, "CHUNK", chunk)
    monkeypatch.setattr(llm_prefix_cache, "HEADROOM", 64)
    llm_prefix_cache.drop()


def _prompt(model, parts):
    """parts: token lists and image seeds. Returns (embeds, cache ids, MRoPE positions) as Qwen35ClipModel builds them."""
    tokens, info, chunks = [], [], []
    for part in parts:
        if isinstance(part, int):  # an image
            image = _image(part)
            info.append(_info(len(tokens) + sum(i["size"] - 1 for i in info)))
            tokens.append(image)
            chunks.append(torch.randn(1, IMAGE_SIZE, HIDDEN, generator=torch.Generator().manual_seed(1000 + part)))
        else:
            tokens.extend(part)
            chunks.append(model.model.embed_tokens(torch.tensor([part])).float())
    embeds = torch.cat(chunks, dim=1)
    ids = llm_prefix_cache.prompt_ids(tokens, info, embeds.shape[1])
    return embeds, ids, qwen2vl_mrope_position_ids(info, embeds.shape[1], "cpu")


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("chunk", [0, 8])
def test_cached_image_prompt_matches_uncached_greedy(monkeypatch, mtp, chunk):
    model = _tiny_model(monkeypatch)
    _cache_on(monkeypatch, chunk)
    system = [1, 2, 3, 4, 5, 6, 7, 8] + MSG_END
    turn1 = [system, [9, 10, 11], 1, [12, 13] + MSG_END, [20, 21, 22, 23, 24, 25, 26, 27]]

    # every multi-token pass (prefill segments, cached or not) gets MRoPE positions for its own length
    real_forward, seen = model.model.forward, []

    def spy(x, embeds=None, position_ids=None, **kw):
        if embeds is not None and embeds.shape[1] > 6:
            seen.append(position_ids is not None and tuple(position_ids.shape) == (3, embeds.shape[1]))
        return real_forward(x, embeds=embeds, position_ids=position_ids, **kw)
    monkeypatch.setattr(model.model, "forward", spy)

    def run(parts, cached):
        embeds, ids, positions = _prompt(model, parts)
        kw = {"cache_ids": ids} if cached else {}
        return model.generate(embeds, do_sample=False, max_length=12, mtp=mtp, position_ids=positions, **kw)

    first = run(turn1, cached=True)
    assert llm_prefix_cache.stats["image_positions"] == IMAGE_SIZE
    assert first == run(turn1, cached=False)
    # the next turn: the reply, then a second image and more text
    turn2 = turn1 + [first + MSG_END, [30, 31, 32], 2, [33, 34] + MSG_END, [40, 41, 42, 43, 44, 45, 46, 47]]
    llm_prefix_cache.drop()
    run(turn1, cached=True)
    second = run(turn2, cached=True)
    stats = dict(llm_prefix_cache.stats)
    assert stats["hit"] and stats["prefix_len"] > len(system) + 3 + IMAGE_SIZE  # resumed past the first image
    assert stats["image_positions"] == 2 * IMAGE_SIZE
    assert second == run(turn2, cached=False)
    assert seen and all(seen)


def test_changed_image_misses_at_the_image(monkeypatch):
    model = _tiny_model(monkeypatch)
    _cache_on(monkeypatch)
    system = [1, 2, 3, 4, 5, 6, 7, 8] + MSG_END
    tail = [[12, 13] + MSG_END, [20, 21, 22, 23, 24, 25, 26, 27]]
    for seed in (1, 2):
        embeds, ids, positions = _prompt(model, [system, [9, 10, 11], seed] + tail)
        model.generate(embeds, do_sample=False, max_length=4, mtp=False, position_ids=positions, cache_ids=ids)
    # the second prompt shares only the text before the image: it resumes at the system message checkpoint
    assert llm_prefix_cache.stats["common_prefix"] == len(system) + 3
    assert llm_prefix_cache.stats["prefix_len"] == len(system)


def test_cached_prefill_uses_image_positions(monkeypatch):
    # the last hidden state after resuming a cached image prefix equals one uncached pass with MRoPE positions,
    # and differs from one with plain sequential positions (so the comparison can see a position mix-up)
    model = _tiny_model(monkeypatch)
    _cache_on(monkeypatch, chunk=8)
    system = [1, 2, 3, 4, 5, 6, 7, 8] + MSG_END
    turn1 = [system, [9, 10, 11], 1, [12, 13] + MSG_END, [20, 21, 22, 23, 24, 25, 26, 27]]
    turn2 = turn1 + [[14, 15] + MSG_END, [30, 31, 32], 2, [33, 34] + MSG_END, [40, 41, 42, 43, 44, 45, 46, 47]]

    def forward(e, kv, p):
        return model.model.forward(None, embeds=e, attention_mask=None, past_key_values=kv, position_ids=p)[0]

    def full(embeds, positions):
        kv = model.init_kv_cache(1, embeds.shape[1] + 8, "cpu", torch.float32)
        return forward(embeds.clone(), kv, positions)[:, -1]

    for parts in (turn1, turn2):
        embeds, ids, positions = _prompt(model, parts)  # the forward updates its input in place: prefill a copy
        _, x = llm_prefix_cache.prefill(model, embeds.clone(), ids, embeds.shape[1] + 16, forward, position_ids=positions)
    assert llm_prefix_cache.stats["prefix_len"] > len(system) + 3 + IMAGE_SIZE
    assert torch.allclose(x[:, -1], full(embeds, positions), atol=1e-4)
    assert not torch.allclose(x[:, -1], full(embeds, None), atol=1e-2)
