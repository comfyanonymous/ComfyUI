# LoD attention - implementation specification (as built)

This is the algorithm-level specification of the LoD sparse attention read as
implemented in this llama.cpp port, written to be sufficient for an independent
(e.g. naive PyTorch) re-implementation and reproduction study. Engineering notes
about kernels, graphs and caches live in `lod-attention.md`; this file is the math
and the state machine only.

Relation to the original LoD1 spec (`lod1_spec.md` in RWKVInside4): this port
implements the **all-pages branch** (page -> leaf, no region tier) with a retained
dense KV cache and segment-shared selection. Divergences are listed in section 8.

## 1. State

Per attention layer with full (non-SWA, non-linear) attention, per sequence:

- dense KV cache: `K[t] in R^(Hkv x Dk)`, `V[t] in R^(Hkv x Dv)` for every past
  token position `t` in `[0, T)`. Nothing is ever evicted or compressed away.
  (The cache may be quantized for storage; that is orthogonal.)
- page sums, F32, accumulated from the **raw pre-quantization** K/V:

      Ks[p, h] = sum_{t in page p} K[t, h]      p = 0 .. ceil(T/ps) - 1
      Vs[p, h] = sum_{t in page p} V[t, h]

  where a page is the fixed position bucket `page(t) = floor(t / ps)`,
  `ps` = page_size (default 64, valid 16..64 here). Sums exist at KV-head
  resolution. Extra state is therefore `2 * D * Hkv * T/ps` floats (~1/ps of
  the cache, ~1.5% at ps=64) - LoD here spends a little MORE memory than dense,
  never less; the savings are all in read volume.

Derived quantities used below, for a read at position `T0` (number of already
processed tokens) with `n` new tokens:

    P     = floor(T0 / ps)          # complete pages, the only summarized ones
    tail0 = P * ps                  # first token NOT covered by a complete page
    tail  = [tail0, T0 + n)         # partial page + the new tokens, read exactly

Invariant: cell index == position (append-only, single stream per sequence).

## 2. Selection

Selection picks `kP = top_pages` (default 32) complete pages to be read at leaf
(exact) detail. Scoring uses the raw K sums (ranking is invariant to the 1/ps
normalization and to the softmax scale):

    score(p, h, q) = q_h . Ks[p, kv(h)]         # kv(h) = h / (Hq/Hkv)

Pooling (this port; the original spec is per-query-head):

- decode (n == 1): `s(p) = max_h score(p, h, q)` - one shared set per layer.
  Optional `sel = head`: one set per KV head, `s_kv(p) = max_{h in group kv}`.
- prefill segment (n > 1): additionally max over the segment's queries:
  `s(p) = max_{h, q in segment} score(p, h, q)` - ONE selection per layer per
  segment (see section 4; this is a deliberate coarsening of the spec).

    sel = argtop_kP s(p),  ties -> lower page id

If fewer than kP complete pages exist, all of them are selected.

## 3. The read: one softmax over three tiers

For a query `q` (head h, position t, scale `c` = 1/sqrt(Dk) unless the model
overrides), the attention output is a single softmax over the union of:

    tier L (leaves)    : logit_j = c * q . K[j]          j in page p, p in sel
                         value_j = V[j]
    tier E (exact tail): logit_j = c * q . K[j]          tail0 <= j <= t
                         value_j = V[j]
    tier S (summaries) : logit_p = c * q . (Ks[p]/ps) + log(ps)
                                                         p complete, p NOT in sel
                         value_p = Vs[p] / ps

    out = sum_i softmax(all logits)_i * value_i

Notes that make this *refinement* rather than pruning:

- a selected page contributes its leaves INSTEAD of its summary (the summary is
  silenced); an unselected page contributes exactly one summary term. Every past
  token therefore contributes to the denominator exactly once, either exactly or
  through its page's count-weighted mean.
- the summary term `exp(c q.mean_k + log ps) = ps * exp(c q.mean_k)` is a
  first-order (Jensen) estimate of `sum_{j in p} exp(c q.k_j)`; by Jensen it is
  an underestimate, so approximation error biases weight toward the exactly-read
  tokens (benign direction).
- **full expansion invariant**: if kP >= P, tier S is empty and the read equals
  dense attention EXACTLY (bitwise in the composed f32 path). This is the primary
  correctness anchor for any re-implementation.
- causality: tier E applies the causal condition `j <= t`; tiers L and S only
  contain complete pages, which lie entirely in the past for every query of the
  current step/segment (pages overlapping the segment are in the tail by
  construction).

## 4. Chunked prefill (segments)

The prompt is processed in segments of `n_seg` tokens (the ubatch, 2048
recommended). For each segment starting at T0:

1. fold the segment's raw K/V into the page sums (append-only; pages that
   complete during this segment become summarizable for LATER segments).
2. compute ONE shared selection from the segment's queries (section 2). Note
   selection sees the sums as of T0 (pages completed by this very segment are
   still in the tail below).
3. every query t in the segment reads: leaves of sel + exact tail
   `[tail0, t]` (causal within the segment) + summaries of unselected complete
   pages. One softmax per query as in section 3.

Segment size is independent of ps. Larger segments widen the exact tail (up to
ps + n_seg tokens) and amortize selection; with n_seg = 2048 prefill quality
matched dense in our measurements (PPL parity).

## 5. Decode

For each generated token (or small speculative batch n <= 8):

1. fold the new token(s) K/V into their page's sum (running, append-only).
2. select per token/batch (section 2) over pages complete as of the CURRENT
   position - selection never lags.
3. read as in section 3 with the causal tail `[tail0, t]`.

Read volume per token: `(P - kP) + kP*ps + |tail|` terms instead of `T` -
asymptotically `T/ps` for the summary walk plus a constant `kP*ps + O(ps)`.

## 6. State maintenance under mutation

The dense cache is ground truth; sums are a derived, append-only accumulator
with a validity watermark `sums_pos` (tokens folded so far):

- append (prefill/decode): fold, advance `sums_pos`.
- suffix removal at p0 (prompt-cache reuse, speculative rollback): rewind
  `sums_pos` to the PAGE FLOOR `floor(p0/ps)*ps` and ZERO the page-sum rows in
  `[floor(p0/ps), ceil(old_pos/ps))`. Folds are read-modify-write, so rewound
  rows must be physically zeroed - a bare watermark rewind leaves stale mass
  that later folds would double-count (this was a real bug: corrupted summaries
  after cache reuse and after MTP rollbacks).
- catch-up: whenever `sums_pos < T0` at read time (after a rewind, or after
  another sequence's dense-path batches), refold `[sums_pos, T0)` from the
  cache before selecting. Exactness is restored because the zeroed rows plus
  refold reproduce the append-only history.
- full clear: zero all sums rows + watermark.
- multi-sequence serving: sums and watermark are per sequence (per KV stream).
  A batch mixing several sequences may fall back to a fully dense read for that
  batch - always coherent, because the cache holds exactly what dense would
  hold; the sums catch up lazily afterwards.

## 7. Parameters and defaults

    page_size  ps    = 64      (16..64; smaller = finer selection, more summary terms)
    top_pages  kP    = 32      (raise to >=128 beyond ~100k context, see 9.)
    selection        = layer   (shared set; `head` = per-KV-head sets)
    segment    n_seg = 2048    (prefill ubatch)

## 8. Divergences from the original LoD1 spec (the "compromises")

1. dense KV retained; leaves read from it. No packed/quantized index, no memory
   saving (original: ~0.3x). Bought: bitwise dense anchor, free interop with
   existing KV quantization, coherent dense fallback, trivial lossless "merge"
   (coarse sums = sum of child sums, pure metadata).
2. no region tier: index is 2-level (page -> token). The read has three TERMS
   (S/L/E) but only one summary LEVEL.
3. selection granularity: layer-shared (optionally per KV head) and per-SEGMENT
   in prefill; the spec is per-query-head with larger budgets (top 64/head).
4. fixed budgets: no adaptive mass criterion, no per-layer budget overrides.
5. page size 64 vs spec default 64/region 512; no >=512k handling.

## 9. Measured reproduction anchors (gemma-4-31B q4_0, qwen3.6-27B)

- full expansion == dense: bitwise (f32 composed path), ~1e-14 nmse graph tests.
- perplexity (c=4096, kP=32): gemma 91.50 (dense 91.92); qwen 4.269 (4.298).
- needle 6k ctx: retrieved at kP=4..8 of ~95 pages.
- needle 100k ctx (~1560 pages): MISSED at kP=32 and 64, retrieved at kP=128
  (layer-shared selection; per-KV-head at 64 was borderline). This is the
  selection-budget wall the region tier / finer pages should address.
- prefill throughput is depth-flat (pages beyond budget cost one dot each);
  decode ~93-100% of dense on hybrid models at 0-48k depth.

## 10. Naive PyTorch sketch

    def lod_read(q, K, V, Ks, Vs, T0, ps, kP, scale):
        # q: [Hq, Dk] one query at position T0 (decode case)
        P     = T0 // ps
        tail0 = P * ps
        g     = Hq // Hkv

        s = torch.einsum('hd,phd->ph', q, Ks[:P, kv_of_heads])   # raw-sum scores
        s = s.max(dim=-1).values                                 # pool over heads
        sel = s.topk(min(kP, P)).indices                         # shared set

        logits, values = [], []
        for p in sel:                                            # tier L
            logits.append(scale * q @ K[p*ps:(p+1)*ps].mT)
            values.append(V[p*ps:(p+1)*ps])
        logits.append(scale * q @ K[tail0:T0+1].mT)              # tier E
        values.append(V[tail0:T0+1])
        uns = mask_out(range(P), sel)                            # tier S
        logits.append(scale * (q @ (Ks[uns]/ps).mT) + math.log(ps))
        values.append(Vs[uns] / ps)

        w = torch.softmax(torch.cat(logits, -1), -1)             # ONE softmax
        return w @ torch.cat(values, -2)

    # prefill segment: same, plus (a) fold segment K/V into Ks/Vs first,
    # (b) one selection max-pooled over the segment's queries and heads,
    # (c) causal tail slice [tail0 : t+1] per query t.

Verification order for a re-implementation: (1) kP >= P equals dense to float
tolerance; (2) PPL parity at kP=32; (3) needle at 6k/kP=8; (4) the 100k budget
wall reproduces (section 9).
