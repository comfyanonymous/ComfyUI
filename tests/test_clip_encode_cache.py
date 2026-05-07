"""Unit tests for the CLIPTextEncode LRU cache (nodes.py).

Mocks a CLIP-like object so the test doesn't need a real text encoder.
Covers: cache hit/miss, key isolation by id/layer_idx/text, deep-copy on
retrieval (downstream dict mutation safe), bounded eviction order,
hooked-clip bypass, env-var disable.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


# nodes.py imports comfy.* heavily — set PYTHONPATH so the test runner
# resolves them. When run from the project root via pytest, this is set
# already; this guard makes the test runnable standalone too.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nodes  # noqa: E402


class _FakeCLIP:
    """Minimal CLIP-shaped object: tokenize returns a placeholder, and
    encode_from_tokens_scheduled returns a CONDITIONING-shaped list."""

    def __init__(self, *, layer_idx=None, has_hooks=False, use_schedule=False):
        self.layer_idx = layer_idx
        self.use_clip_schedule = use_schedule
        self.calls = 0
        if has_hooks:
            self.patcher = mock.SimpleNamespace(forced_hooks=object())
        else:
            self.patcher = None

    def tokenize(self, text):
        return {"tokens": text}

    def encode_from_tokens_scheduled(self, tokens):
        # Return a CONDITIONING: list of [tensor-stand-in, dict] pairs.
        # Use a unique tensor stand-in (a fresh list) per call so we can
        # tell whether we got a cached value or a freshly-encoded one.
        self.calls += 1
        marker = [self.calls, tokens["tokens"]]
        return [[marker, {"original": True}]]


class CLIPEncodeCacheTests(unittest.TestCase):
    def setUp(self):
        # Reset cache between tests
        nodes._CLIP_ENCODE_CACHE.clear()
        nodes._CLIP_ENCODE_CACHE_ENABLED = True

    def test_cache_hit_skips_second_encode(self):
        clip = _FakeCLIP()
        a = nodes._clip_encode_cached(clip, "hello world")
        b = nodes._clip_encode_cached(clip, "hello world")
        self.assertEqual(clip.calls, 1, "second call should hit cache")
        # Returned conditioning has the same tensor (shared)
        self.assertIs(a[0][0], b[0][0])

    def test_cache_returns_fresh_outer_dict(self):
        # Downstream nodes mutate the per-cond dict; the cache must not
        # leak those mutations back to the next consumer.
        clip = _FakeCLIP()
        a = nodes._clip_encode_cached(clip, "x")
        a[0][1]["mutated"] = True
        b = nodes._clip_encode_cached(clip, "x")
        self.assertNotIn("mutated", b[0][1],
                          "dict mutation must not leak across cache hits")

    def test_layer_idx_in_key(self):
        clip = _FakeCLIP(layer_idx=-1)
        nodes._clip_encode_cached(clip, "x")
        self.assertEqual(clip.calls, 1)
        # Mutating layer_idx (e.g. CLIPSetLastLayer) shouldn't reuse cache
        clip.layer_idx = -2
        nodes._clip_encode_cached(clip, "x")
        self.assertEqual(clip.calls, 2,
                          "layer_idx change must invalidate cache key")

    def test_different_text_misses_cache(self):
        clip = _FakeCLIP()
        nodes._clip_encode_cached(clip, "alpha")
        nodes._clip_encode_cached(clip, "beta")
        self.assertEqual(clip.calls, 2)

    def test_different_clip_instance_misses_cache(self):
        c1 = _FakeCLIP()
        c2 = _FakeCLIP()
        nodes._clip_encode_cached(c1, "x")
        nodes._clip_encode_cached(c2, "x")
        # Each clip has its own counter — both should have encoded once
        self.assertEqual(c1.calls, 1)
        self.assertEqual(c2.calls, 1)

    def test_hooked_scheduled_clip_bypasses_cache(self):
        clip = _FakeCLIP(has_hooks=True, use_schedule=True)
        nodes._clip_encode_cached(clip, "x")
        nodes._clip_encode_cached(clip, "x")
        self.assertEqual(clip.calls, 2,
                          "hooked + scheduled CLIPs must always re-encode")
        self.assertEqual(len(nodes._CLIP_ENCODE_CACHE), 0,
                          "hooked path must not pollute the cache")

    def test_hooked_unscheduled_clip_caches(self):
        # Hooks present but use_clip_schedule=False — fast path still applies
        clip = _FakeCLIP(has_hooks=True, use_schedule=False)
        nodes._clip_encode_cached(clip, "x")
        nodes._clip_encode_cached(clip, "x")
        self.assertEqual(clip.calls, 1,
                          "non-scheduled hooked CLIPs still cache (fast path)")

    def test_disabled_via_env_skips_cache(self):
        nodes._CLIP_ENCODE_CACHE_ENABLED = False
        clip = _FakeCLIP()
        nodes._clip_encode_cached(clip, "x")
        nodes._clip_encode_cached(clip, "x")
        self.assertEqual(clip.calls, 2)
        self.assertEqual(len(nodes._CLIP_ENCODE_CACHE), 0)

    def test_lru_eviction(self):
        # Stuff the cache past the cap and verify oldest entry evicted
        original_max = nodes._CLIP_ENCODE_CACHE_MAX
        try:
            nodes._CLIP_ENCODE_CACHE_MAX = 3
            clips = [_FakeCLIP() for _ in range(5)]
            for c in clips:
                nodes._clip_encode_cached(c, "x")
            self.assertEqual(len(nodes._CLIP_ENCODE_CACHE), 3)
            # The oldest two clips' entries should be gone
            keys = list(nodes._CLIP_ENCODE_CACHE.keys())
            kept_clip_ids = {k[0] for k in keys}
            self.assertNotIn(id(clips[0]), kept_clip_ids)
            self.assertNotIn(id(clips[1]), kept_clip_ids)
            self.assertIn(id(clips[4]), kept_clip_ids,
                          "most-recent must always be kept")
        finally:
            nodes._CLIP_ENCODE_CACHE_MAX = original_max

    def test_lru_recency_on_hit(self):
        # Re-hitting a key promotes it; oldest gets evicted instead.
        original_max = nodes._CLIP_ENCODE_CACHE_MAX
        try:
            nodes._CLIP_ENCODE_CACHE_MAX = 3
            c1, c2, c3 = _FakeCLIP(), _FakeCLIP(), _FakeCLIP()
            nodes._clip_encode_cached(c1, "x")
            nodes._clip_encode_cached(c2, "x")
            nodes._clip_encode_cached(c3, "x")
            # Touch c1 so it's most-recent
            nodes._clip_encode_cached(c1, "x")
            # New entry should evict c2 (now oldest), not c1
            c4 = _FakeCLIP()
            nodes._clip_encode_cached(c4, "x")
            keys = {k[0] for k in nodes._CLIP_ENCODE_CACHE.keys()}
            self.assertIn(id(c1), keys, "recently-touched c1 must survive eviction")
            self.assertNotIn(id(c2), keys, "oldest c2 must have been evicted")
        finally:
            nodes._CLIP_ENCODE_CACHE_MAX = original_max


if __name__ == "__main__":
    unittest.main()
