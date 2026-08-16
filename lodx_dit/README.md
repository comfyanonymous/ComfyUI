# lodx_dit — LoD sparse attention for a bidirectional DiT

`Qwen3.5-2B-Lodx` の LoD 読み出しを、MiniMax H3 の DiT 形状に移植したもの。
本体（`comfy/`）への変更は `cli_args.py` の `--dit-gpus` 1 行だけ。

- **[docs/lod-explained.md](docs/lod-explained.md)** — 前提知識ゼロ向けの解説（英語）。
  どの次元がどう疎になるか、実寸の数値例つき。外部への説明はここから
- **[docs/lod-dit-design.md](docs/lod-dit-design.md)** — 仕組み、移植元との差分、落とし穴
- **[docs/lod-dit-results.md](docs/lod-dit-results.md)** — 実測値、最適化の記録、TP 検証

```bash
python lodx_dit/test_lod_dit.py       # 受け入れテスト 28 項目 (pytest 不要)
python lodx_dit/bench_h3.py           # H3 実形状の単層ベンチ
python lodx_dit/bench_h3.py --tune    # タイル形状の再チューニング
python lodx_dit/dit_profile.py        # 実モデルの forward 内訳
```

## 一行で

クエリブロックごとに、**選ばれたページの実トークン**・**選ばれなかったページの
1 項要約**・**先頭の条件行**を、単一の softmax で読む。全トークンが分母に
ちょうど 1 回入るので、予算を全ページに開けば **dense と厳密に一致する**。
枝刈りではなく解像度の切り替え。

## 効き方

**attention が forward に占める比が全て。** それは尺で動く（640x640）。

| 尺 | S | attn 比 | attention 単体 | DiT 全体 |
|---|---|---|---|---|
| 2秒 | 7,215 | 27% | 1.27x | 1.06x |
| 5秒 | 15,514 | 43% | 2.16x | 1.32x |
| 15秒 | 44,606 | 71% | **4.53x** | **2.17x** |

短尺では線形層（INT8 GEMM）が床になるので効かない。**長尺・高解像度の道具**。

## 使い方（ComfyUI）

`custom_nodes/lod_attention/` 経由で 2 ノードが登録される。

**MiniMax H3 Attention Mode (dense / LoD)** — `model/patch/minimax`。
出力ノードなので**キャンバスに置くだけで効く**。`model` 入力は任意で、
繋げばその MODEL クローンだけに、繋がなければグローバルに適用される。
`mode=dense` は素の読み出しと**ビット一致**（テストで固定）。

解像度・フレーム数・条件行の位置はモデルから自動導出するので入力は不要。

| つまみ | 既定 | |
|---|---|---|
| `mode` | lod | `dense` で A/B の基準を取る |
| `top_pages` | 128 | 予算。**64 前後が実用点**（ユーザ評価で 32 でも生成可） |
| `select_block` | 64 | 1 集合を共有するクエリ数。32 より 1.2x 速いが選択は粗い |
| `page_size` | 64 | フレームを割る空間ブロックに丸められる（1344x768 で 8x7=56） |
| `local_radius` | 0 | 自ページに加えて強制する隣接ページ数。-1 で強制を無効 |
| `tiled_pages` | true | ページを空間ブロックにする並べ替え |
| `contiguous_qkv` | true | 非連続 q/k/v の SDPA 劣化を回避。**両モードに**掛かる |
| `start_percent` / `end_percent` | 0.0 / 1.0 | **動かさないこと**（下記） |
| `kernel_variant` | default | 実験用カーネルの切り替え。全て default より遅い |

**Compare Image Batches (PSNR/SSIM)** — `image/compare`。フレーム別 PSNR/SSIM、
前半・後半の drift、最悪フレームと差分画像。1 ワークフローに dense 側と LoD 側の
サンプラーを 2 本置いて受けるのが最短。

数値だけでは決まらない。動画の知覚品質に PPL 相当の指標は無いので、PSNR/SSIM は
「どのフレームを見るか」を絞る道具として使う。**drift 行が要**で、疎な読みが
破綻するときは後半から崩れる。

### 落とし穴

- **`start_percent` / `end_percent` は percent では使えない。** H3 は `shift=12` で
  sigma 分布が偏るため、`0.3` は「序盤 30% を dense」ではなく**全ステップ LoD**、
  `end_percent=0.5` は **0 ステップ**になる。既定から動かさないこと。
- **`/32` が素数になる解像度を避ける** — 416, 544, 608, 736, 928, 992, 1184。
  グリッドが割れずページが細帯になる（`(1,1,23)` 等）。警告が出る。
- **LoD は dense の 7 倍の追加メモリを使う**（20秒で +4.51 GB 対 +0.64 GB）。

## ComfyUI 本体の既知問題（LoD とは独立）

| | 対処 |
|---|---|
| モデルロードが 27 GB で約 7 分 | **`--enable-dynamic-vram`**（→ 19.2 秒） |
| Qwen3-VL vision tower が segfault | **`COMFYUI_ENABLE_MIOPEN=1`** |

詳細は [results §9](docs/lod-dit-results.md)。

## マルチ GPU

**Tensor Parallel**（`tp.py`、本体無改変）は実測 1.47〜1.57x。

```bash
CUDA_VISIBLE_DEVICES=0,1 python lodx_dit/tp_run.py --gpus 2 --mode bf16
CUDA_VISIBLE_DEVICES=0,1 python lodx_dit/tp_run.py --gpus 1   # 自己検査
```

`--gpus 1` は分割せず同じコード経路を通す自己検査で、**ビット完全一致**する。
ただし行分割の量子化誤差が残り、2 step の軌道で 7.9e-2 の差が出る（results §8.7）。

**Pipeline Parallel** は `python main.py --dit-gpus 2 --disable-dynamic-vram`。
**速くはならない**（拡散ステップが直列 + batch=1 でパイプラインを
埋めるものが無い）。1 枚に載らない場合の容量対策。DynamicVRAM とは併用不可
（patcher が単一 `load_device` に引き戻す）。

## ファイル

```
lod_dit.py        参照実装 + 選択 + fast path (PagedLayout / lod_attention)
kernel.py         Triton: 層 X+S+L を 1 つの online softmax で / 融合選択スコア
kernel_exp.py     実験用カーネルの別系統 (VARIANTS に足せばテストが自動で回る)
ordering.py       ページを空間ブロックにする置換 (best_tile / tile_order)
comfy_node.py     ComfyUI ノード 2 種 + h3 へのフック
pipeline.py       DiT の pipeline 分割 (--dit-gpus)
tp.py             DiT の tensor 分割 (DiTBlock.forward を差し替え)
tp_probe.py       1 ブロックだけで TP の算術と速度を測る
tp_run.py         実モデルで 1 GPU と TP を比較 (--gpus 1 が自己検査)
test_lod_dit.py   受け入れテスト 28 項目 (全展開==dense を全経路で)
bench_h3.py       H3 実形状の単層ベンチ / --tune / --long
dit_profile.py    実モデルの forward 内訳
ab_h3.py          dense と LoD の A/B
probe_h3.py       実 H3 活性値での品質測定 (未実行)
```
