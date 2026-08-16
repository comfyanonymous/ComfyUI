# LoD Attention, explained from scratch

This assumes you know what a matrix multiply is and nothing else. It builds up
to exactly what our implementation does to a MiniMax H3 video model, step by
step, with real tensor shapes at every stage.

**One sentence:** attention normally reads every key at full detail; LoD reads a
few regions at full detail and the rest as one-number summaries, in a single
softmax, so nothing is thrown away — it is a *resolution* choice, not a
*pruning* choice.

---

## Part 1 — Why attention gets expensive

### 1.1 What attention computes

Attention takes three tensors, all the same shape:

```
q  (queries)  [S, D]
k  (keys)     [S, D]
v  (values)   [S, D]
```

`S` is the number of tokens, `D` the size of each token's vector. For every
query row `i`, attention answers: *"which of the S keys should I listen to, and
what mixture of values does that give me?"*

```
1.  score[i, j] = c * (q[i] · k[j])          for all j        -> S numbers
2.  w[i, :]     = softmax(score[i, :])                        -> S weights, summing to 1
3.  out[i]      = Σ_j w[i, j] * v[j]                          -> one D-vector
```

`c` is a fixed scale (`1/√D`). Step 2 is what makes attention *attention*: the
weights are exponentials normalised by their own total.

```
                    exp(score[i, j])
    w[i, j]  =  ───────────────────────
                  Σ_j' exp(score[i, j'])
                       ↑
                  "the denominator" — every key contributes to it
```

Hold onto the denominator. It is the whole story later.

### 1.2 The cost

Step 1 does `S × S` dot products. That is **quadratic**: double the tokens,
quadruple the work.

For a language model, `S` is a few thousand. For a **video** model, a token is a
patch of pixels in a frame, and you have many frames. Our real case:

| | |
|---|---|
| clip | 640×640, 5 seconds |
| tokens `S` | **15,514** |
| heads `H` | 56 (attention runs 56 times in parallel, each with `D=128`) |
| layers | 50 |

`S² = 240,684,196` scores **per head, per layer, per denoising step**. Multiply
by 56 heads and 50 layers and one step costs ~345 TFLOP of attention alone. On
our GPU that is about 5 seconds, and a video needs 20–50 steps.

Worse, it grows with clip length:

| clip | `S` | share of the model's time spent in attention |
|---|---|---|
| 2 s | 7,215 | 27% |
| 5 s | 15,514 | 43% |
| 15 s | 44,606 | **71%** |

At 15 seconds the model spends most of its life on this one operation. That is
the thing worth attacking.

---

## Part 2 — The idea

### 2.1 The tempting wrong answer

Most sparse attention says: *most weights are near zero, so skip those keys.*
Compute scores for a subset, softmax over the subset, done.

This breaks the denominator. Softmax normalises by the sum over whatever you
gave it. Drop 90% of the keys and the surviving 10% are renormalised to sum to
1 — they get inflated ~10×. The model was calibrated on a denominator built
from *all* the keys, and you just changed it. Errors accumulate over 50 layers
and 30 steps.

There is a second problem. You have to compute a score to know it is small. If
deciding what to skip costs as much as not skipping, you have saved nothing.

### 2.2 The LoD answer

**Keep every key in the denominator. Change how precisely you read it.**

Group the keys into fixed **pages** of `ps` consecutive rows. For each page,
precompute one summary vector — the sum of its keys. Then per query:

- pick a small number of pages that look important → read those **key by key**
- for every other page → add **one** term standing for the whole page

Both kinds of term go into **the same softmax**. Every key is accounted for
exactly once. Nothing is renormalised away.

This is the "Level of Detail" idea from computer graphics: distant objects are
drawn with fewer polygons, not deleted.

### 2.3 Why one number can stand for a whole page

We need the summary term to approximate what those `ps` keys *would* have
contributed to the denominator:

```
    Σ_{j in page}  exp(c * q · k_j)
```

Move the sum inside the exponential (a first-order approximation):

```
    ≈  ps * exp( c * q · (Σ_j k_j) / ps )
    =  exp( c * q · (Ks / ps)  +  log(ps) )        where Ks = Σ_j k_j
```

So **one extra logit** covers the page:

```
    summary_logit(page) = c * q · (Ks / ps) + log(ps)
                          └──── mean key ────┘   └─ "there were ps of us" ─┘
```

The `log(ps)` is the count correction. Without it a page of 50 keys would
compete as if it were a single key.

Storing **sums** rather than means is deliberate: `ps` is then a constant, so
`log(ps)` is a constant, and no per-page bookkeeping is needed.

**The approximation is one-sided.** By Jensen's inequality this always
*underestimates* the true sum. Summaries are quieter than the region they stand
for, so weight shifts toward the pages read exactly. That is the safe direction
to be wrong in — you never over-trust a region you did not look at.

### 2.4 Worked toy example (real numbers)

12 keys, `D = 1` so a dot product is just a product. `page_size = 4`, so 3
pages. Budget: read **1** page exactly. `q = 1.0`, `c = 1.0`.

```
        page 0                page 1                 page 2
k  =  [2.0, 2.2, 1.8, 2.0]  [0.1, 0.0, -0.1, 0.0]  [0.2, 0.1, 0.0, -0.1]
v  =  [ 10,  11,   9,  10]  [  1,   0,   -1,   0]  [  2,   1,   0,   -1]
```

**Page sums** `Ks = [8.0, 0.0, 0.2]`.

**Summary logits** `= q·Ks/4 + log(4)`:

| page | summary logit | true `logsumexp` of its 4 keys | gap |
|---|---|---|---|
| 0 | 3.3863 | 3.3963 | −0.0100 |
| 1 | 1.3863 | 1.3888 | −0.0025 |
| 2 | 1.4363 | 1.4425 | −0.0062 |

Every summary sits just *below* the truth — the one-sided bias from §2.3, and
close enough to rank pages correctly.

**Select** the highest → page 0. Now build one softmax over 4 leaf logits (page
0's actual keys) plus 2 summary logits (pages 1 and 2):

```
                        denominator      output
    dense (all 12)         38.094         7.9893
    LoD  (read 6)          38.058         7.9775
```

**Half the reads, denominator off by 0.1%, output off by 0.15%.** And if you
raise the budget to 3 pages, the summary tier empties and you get 7.9893 —
*bit-for-bit the dense answer*.

That last property is the anchor the whole implementation is tested against.

---

## Part 3 — Which dimension becomes sparse

This is the part people usually get wrong when they hear "sparse attention", so
here it is precisely.

### 3.1 The attention matrix

Per head, attention conceptually builds an `S × S` matrix: rows are queries,
columns are keys.

```
                      keys  (15,514 columns)
              ┌──────────────────────────────────────┐
    queries   │                                      │
    (15,514   │        score[i, j]                   │
     rows)    │                                      │
              └──────────────────────────────────────┘
```

- **Rows (queries): fully dense. Every query is computed, always.** LoD never
  skips an output. Every one of the 15,514 tokens gets its own answer.
- **Columns (keys): this is the only axis LoD touches.**
- **Heads: not independent.** All 56 heads read the *same* selected pages. The
  choice is pooled across heads (§4.5) so the kernel loads one contiguous set
  of keys and reuses it for all heads.

### 3.2 The key axis is not "dropped", it is re-binned

Take the 15,514 columns for one query. Dense reads 15,514 of them. LoD replaces
that axis with a **shorter** axis:

```
DENSE  ── 15,514 columns, each one key ───────────────────────────────────────

  │████████████████████████████████████████████████████████████████████████│
   ↑                                                                       ↑
  key 0                                                             key 15,513


LoD  ── 4,146 columns, at two resolutions ────────────────────────────────────

  │▓▓▓▓▓│██████████████████│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
   ↑      ↑                 ↑
   │      │                 └─ 232 columns, one per unselected page
   │      │                    (each stands for 50 keys — 11,600 keys total)
   │      └─ 3,200 columns, one per key
   │         (64 selected pages × 50 keys)
   └─ 714 columns, one per key
      (the exact prefix: text / conditioning rows)
```

The 11,600 keys behind those 232 summary columns **still contribute to the
denominator**. They are compressed 50:1, not removed. That is the difference
between LoD and a mask.

**Read rate: 4,146 / 15,514 = 26.7%.**

### 3.3 The pattern is block-structured, not scattered

Queries do not choose pages individually. They are grouped into **query blocks**
of 64 consecutive rows, and all 64 share one page set. So the real decision
variable is a small binary matrix:

```
                        296 pages
                ┌───────────────────────────┐
   243 query    │ ▓ ░ ░ ▓ ░ ░ ░ ▓ ░ ▓ ░ ░ ░ │   ▓ = read this page's 50 keys
   blocks       │ ░ ▓ ▓ ░ ░ ▓ ░ ░ ░ ░ ▓ ░ ░ │   ░ = read this page's 1 summary
                │ ░ ░ ▓ ░ ▓ ░ ░ ░ ▓ ░ ░ ▓ ░ │
                └───────────────────────────┘
                  exactly 64 ▓ per row (the budget)
```

`243 × 296` = 71,928 decisions, versus 240 million scores. **Choosing is ~3,000×
cheaper than the thing being chosen** — that is what makes the saving real
rather than moved around.

Two consequences worth stating plainly:

- The work is **regular**. Every query block does an identical amount of work:
  64 pages × 50 keys, plus 232 summaries, plus 714 exact. No load imbalance, no
  ragged tiles. GPUs are fast at this and slow at scattered gathers.
- The cost of the leaf tier is `top_pages × page_size` — **a constant,
  independent of `S`**. Doubling the clip length doubles `S²` for dense but
  leaves the leaf tier untouched. This is why the speedup *grows* with length.

---

## Part 4 — What the implementation actually does, step by step

Real numbers throughout: **640×640, 5 s, `top_pages = 64`, `select_block = 64`.**

### Step 0 — What arrives

The DiT hands attention three tensors:

```
q, k, v :  [1, 56, 15514, 128]
           └─ batch
              └─ heads
                  └─ tokens (S)
                      └─ head dim (D)
```

The token axis is a *packed* sequence — several things concatenated:

```
[  text  |  cond  |  ref  |  audio  |════════ video ════════]
 └────────── 714 rows ───────────┘   └──── 14,800 rows ────┘
```

Video is always last. That matters: it means the non-video rows form one
contiguous block at the front.

### Step 1 — Split the key axis

```
prefix      = 714           (everything before video)
video rows  = 14,800
page_size   = 50            (chosen in Step 2)
n_pages     = 14,800 / 50 = 296
```

The 714 prefix rows are **never summarised**. They carry the text prompt; blur
them and prompt-following degrades. They are read exactly, always.

Only *complete* pages are summarised. Any leftover rows at the end join the
exact tier. There is no such thing as a half-full summary.

### Step 2 — Choose the page shape, and reorder

A page is `ps` **consecutive rows**. That is what makes the leaf tier a
contiguous memory read. So the only way to change *what a page covers* is to
change the order the rows arrive in.

The model emits video tokens in **raster order** — left to right, top to bottom,
frame by frame. At 640×640 the patch grid is 20 wide, so:

```
raster order, page_size 50:

    row 0:  ████████████████████    20 tokens
    row 1:  ████████████████████    20 tokens
    row 2:  ██████████              10 tokens   ← page ends mid-row
                                    ─────
                                    50

    a page = a strip 20 wide and 2.5 tall
```

A strip is a poor thing to summarise. The tokens directly above and below a
given token — its strongest neighbours in a video — land in *different* pages.

So we permute the video rows into **tile-major** order: sort by
`(frame, row//5, col//10, row%5, col%10)`. Now a page is exactly one 5×10
spatial block.

```
                     mean bounding box of a page (lower = better, floor = 50)

    raster order                  60.0
    tile order  (1,5,10)          50.0   ← perfectly compact
```

**This permutation is free and exact.** Permute `q`, `k` and `v` together and
un-permute the output, and attention gives an identical answer — softmax does
not care what order the keys are in. Only *which pages get selected* changes.

Picking the tile is subtler than it looks. The naive choice of a square 8×8
tile is **worse than doing nothing** on some grids: at 24×42 (a 1344×768 clip)
42 is not divisible by 8, so the edge tiles are short, page boundaries drift
across both tile rows and frames, and the mean bounding box goes **192 → 234**,
losing to raster order. So `best_tile` only considers tiles that *exactly
divide* the grid, and prefers the squarest before the largest. `page_size` is
then whatever that block contains — 50 here, 56 at 1344×768, 64 at 1024×1024.

> **Practical warning.** If `width/32` or `height/32` is prime, the grid barely
> factors and the best available tile is a thin strip. Avoid widths/heights of
> 416, 544, 608, 736, 928, 992, 1184.

**A page does not have to be flat in time, and probably should not be.** The
tile is `(bt, bh, bw)` and we currently pin `bt = 1`, on the reasoning that a
frame is a far longer stride than a row. Measured against real activations,
that reasoning is backwards. Holding `page_size` at 56 on a 1344×768 clip so the
read rate is identical and only the *shape* changes:

| page shape | rel. error, layer 0, budget 64 |
|---|---|
| raster (no reorder) | 0.1644 |
| `1×8×7` — one frame, 8×7 patches (current default) | 0.1452 |
| `2×4×7` — 2 frames, 4×7 patches | 0.1274 |
| **`4×2×7` — 4 frames, 2×7 patches** | **0.1262** |

Spending the page on *time* beats spending it on space, in every layer and
budget we measured. Latent frames are already temporally compressed, so two
tokens one latent frame apart are more alike than two tokens a few patches
apart.

But **more time is not simply better** -- sweeping the depth gives a U-curve,
with the optimum at 4 frames (mean over all layers, budget 32):

| frames per page | `1x8x7` | `2x4x7` | **`4x2x7`** | `8x1x7` |
|---|---|---|---|---|
| rel. error | 0.2578 | 0.2398 | **0.2352** | 0.2443 |

At 8 frames the spatial footprint collapses to 1 row x 7 columns and it gets
worse again. What wins is a **balanced 3D block**, not a temporal one.

This is measured but **not yet the default**, because changing it moves the
output and would invalidate quality evaluations done at the current setting.

### Step 3 — Build the page summaries

```
k[:, :, 714:15514]                  [1, 56, 14800, 128]
  .reshape(1, 56, 296, 50, 128)     group into pages
  .sum(dim=3, dtype=float32)        ─────────────────►  Ks : [1, 56, 296, 128]
```

Same for `v` → `Vs`. Cost: one pass over the keys, **linear** in `S`.

The `dtype=float32` inside `sum` matters. Writing `.float()` first would
materialise the entire paged region in fp32 — 1.2 GB at a 15 s clip. Accumulating
in fp32 while reading bf16 gives the same answer for free. Fixing this was worth
**3–5×** on its own.

### Step 4 — Group the queries

```
select_block = 64
n_blocks     = ceil(15514 / 64) = 243
```

Each block of 64 consecutive queries shares one page set. Coarser blocks are
faster (fewer selections, better memory reuse) but less precise, since the 64
queries must agree on what matters.

Because of Step 2, 64 consecutive queries are now spatially adjacent — one 5×10
tile plus a bit — so they genuinely do want the same regions. Before the
reorder they were a 3-row strip and agreement was weaker.

### Step 5 — Score every (query block, page) pair

For each block and page, score using the page sum:

```
score[block, page] = max over the 64 queries and 56 heads of   q · Ks[page]
```

giving `[1, 243, 296]` — 71,928 numbers.

Pooling across heads means all heads read the same pages, so the kernel loads
each selected page once and reuses it 56 times.

This is computed by a **fused kernel** that never materialises the intermediate.
The obvious implementation builds `[B, H, P, S]` and reduces it away — 2.5 GB at
this shape. Fusing was worth **2.13×**.

It also turned out to be *more accurate*. Checked against an fp64 ground truth:

| | mean error | top-32 selection matching ground truth |
|---|---|---|
| `torch.einsum` (bf16 accumulation) | 4.8e-01 | **80.3%** |
| our fused kernel (fp32 accumulation) | 2.8e-05 | **100.0%** |

Scores here have magnitude ~293, and bf16 has ~3 decimal digits, so accumulating
in bf16 misranks about a fifth of the selection.

### Step 6 — Force the local pages

Before ranking, each query block's **own** page is forced in, by setting its
score to `+∞`.

A video token's strongest neighbours are its immediate spatial surroundings. If
selection ever failed to pick a block's own page, a token would read its own
neighbourhood as a blur. Cheap to prevent, so we prevent it.

(`local_radius` widens this to adjacent pages; `-1` disables it. With the budget
opened to all pages, forcing changes nothing — so the dense-equivalence anchor
still holds.)

### Step 7 — Take the top 64

```
order = argsort(-score, stable=True)[..., :64]      ->  [1, 243, 64]
```

`argsort` rather than `topk`: a **stable** sort breaks ties toward the lower
page id, which is a rule the algorithm specifies and `topk` does not guarantee.
It is also 3.6× faster here (0.08 ms vs 0.29 ms) — at this size the sort is not
the bottleneck, the scoring is.

### Step 8 — The single-softmax read

One Triton kernel per (query block, head). For its 64 queries it walks three
tiers, maintaining a running max and running denominator (online softmax, so
nothing is ever fully materialised):

| tier | what | logit | count here |
|---|---|---|---|
| **X** exact | the 714 prefix rows | `c·q·k_j` | 714 |
| **L** leaf | keys of the 64 selected pages | `c·q·k_j` | 3,200 |
| **S** summary | one term per *unselected* page | `c·q·(Ks/ps) + log(ps)` | 232 |

The critical detail: **a selected page's summary is switched off.** The kernel
carries a "dead" flag per page; selected pages contribute their keys, everyone
else contributes their summary. Never both, never neither.

```
    every one of the 15,514 keys enters the denominator exactly once
```

Set `top_pages ≥ 296` and the summary tier empties, leaving a plain dense
softmax — the anchor from §2.4, pinned by the test suite on every code path.

### Step 9 — Un-permute

Invert the Step 2 permutation on the output. Attention is done, and the rest of
the model never knows anything happened.

---

## Part 5 — What it buys

Measured on a Radeon PRO W7900, MiniMax H3 (19.3B, 50 blocks), 640×640:

### Attention alone

| clip | `S` | pages | dense | LoD (64) | speedup |
|---|---|---|---|---|---|
| 2 s | 7,215 | 136 | 24.1 ms | 19.0 ms | 1.27× |
| 5 s | 15,514 | 296 | 109.5 ms | 50.6 ms | 2.16× |
| 15 s | 44,606 | 856 | 1008.4 ms | 222.5 ms | **4.53×** |

### The whole model

| clip | dense | LoD | speedup |
|---|---|---|---|
| 2 s | 4.84 s/step | 4.57 s/step | 1.06× |
| 5 s | 13.57 s/step | 10.32 s/step | 1.32× |
| 15 s | 72.69 s/step | 33.49 s/step | **2.17×** |

**The gap between those two tables is the point.** Attention gets 4.53× at 15 s
but the model only gets 2.17×, because attention is 71% of the time — Amdahl's
law. At 2 s attention is only 27% of the time, so even a large win barely moves
the total. Here is where the time actually goes at 2 s:

| | share |
|---|---|
| MLP (fc1+fc2, INT8) | 42.3% |
| attention qkv/out projections | 29.5% |
| **attention core (the S² part)** | **25.0%** |
| everything else | 3.2% |

**LoD is a tool for long, high-resolution clips.** On short ones the matrix
multiplies dominate and this does not help.

### Why read rate ≠ speedup

At 5 s we read 26.7% and get 2.16×, not 3.7×. Two reasons:

1. **Fixed costs.** Page sums, scoring, sorting, permutation — all linear in
   `S`, all still there. About 10–13% of LoD's runtime.
2. **The kernel is less efficient than the vendor's.** Ours runs at 38.8–50.6
   TFLOPS against 56.4–61.7 for the dense flash-attention kernel — **63–71%**.
   A hand-written kernel is fast; a hand-written kernel competing with one that
   an entire vendor team tuned is 70% as fast.

There is also a memory cost: LoD's extra allocations are **7× dense's**
(+4.51 GB vs +0.64 GB on a 20 s clip), from the permuted copies of q/k/v and an
fp32 kernel output.

---

## Part 6 — The two questions people ask

> **"So you average 5 tokens together?"**

No. Nothing is averaged in the output. The *summary* uses a mean key to build
one logit, but that logit competes in the same softmax as the exact ones, and
the regions you selected are read key by key at full precision. It is closer to
peripheral vision than to blurring: you see everything, in detail only where
you are looking.

> **"So you use 20% of the regions fully and drop the rest?"**

The first half, yes — with a budget of 64 out of 296 pages, ~22% of the video is
read at full detail. But the rest is not dropped. It is read at 1/50th
resolution, and it still contributes to the denominator, so the softmax stays
correctly normalised. Open the budget to all 296 pages and you get the dense
answer back exactly.

---

## Part 7 — Honest limitations

- **Quality is user-validated, not systematically measured.** Our testing proves
  the *dense-equivalence* property exactly and measures speed carefully. How
  output quality degrades as the budget falls has been checked by eye (32 pages
  produces usable video, 64 is more comfortable) but not with a metric sweep.
  The one systematic quality measurement we have run is the page-shape
  comparison in Step 2 -- which found the current default is not the best
  choice.
- **The accuracy fix in Step 5 changes which pages get selected** compared to
  the original LLM implementation this was ported from — about 20% of a top-32
  selection differs. Quality numbers do not transfer between the two.
- **Two of the additions are unvalidated.** Forced local pages (Step 6) and the
  coarser `select_block = 64` are both justified by argument and measured for
  speed, not verified for quality.
- **Tuning is GPU-specific.** The kernel's tile shapes were tuned on RDNA3,
  where the register file — not shared memory — is the binding constraint. The
  conclusions invert on CDNA3.
- **Short clips see almost nothing.** 1.06× at 2 seconds.

---

## Appendix — Notation

| | |
|---|---|
| `S` | sequence length (tokens) |
| `D` | head dimension (128 here) |
| `H` | number of heads (56 here) |
| `ps` | `page_size` — keys summarised by one vector (50 here) |
| `n_pages` | number of complete pages (296 here) |
| `top_pages` | budget — pages read at full detail (64 here) |
| `select_block` | queries sharing one page set (64 here) |
| `prefix` | leading rows never summarised (714 here) |
| `Ks` | page sum of keys, `Σ_{j in page} k_j` |
| tier X / L / S | exact / leaf / summary |
