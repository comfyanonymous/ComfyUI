# ComfyUI Serving Benchmarks

Measures latency and throughput of a running ComfyUI server by submitting
concurrent prompt requests and collecting results from the history API.

## Dependencies

```bash
pip install aiohttp tqdm gdown
```

## Supported models / tasks

| Model | Task | Description |
|-------|------|-------------|
| `wan22` | `i2v` | Wan 2.2 Image-to-Video — LightX2V 4-step, 720×720, 81 frames |

To add a new model/task: drop a workflow JSON in `workflows/` (with
`__INPUT_IMAGE__` as the image placeholder) and add an entry to
`_MODEL_REGISTRY` in `benchmark_comfyui_serving.py`.

## How it works

On each run the script:

1. Downloads model weights into the ComfyUI `models/` directory (only if
   `--download-models` is passed).
2. Downloads the [VBench I2V](https://github.com/Vchitect/VBench) image
   dataset via `gdown` into ComfyUI's `input/` folder.
3. Generates one prompt JSON per input image under
   `benchmarks/prompts/<model>_<task>/`.
4. Submits `--num-requests` prompts to the server, cycling through the
   generated prompt files in round-robin order.
5. Polls `/history/{prompt_id}` for completion and prints a latency /
   throughput summary.

Per-node execution times are available when the server is started with
`--benchmark-server-only`.

## Usage

### Start the server

```bash
python main.py --listen 127.0.0.1 --port 8188 --benchmark-server-only
```

### Run the benchmark

```bash
# From the ComfyUI root directory:
python3 benchmarks/benchmark_comfyui_serving.py \
  --model wan22 --task i2v \
  --num-requests 50 --max-concurrency 4 \
  --host http://127.0.0.1:8188
```

Include model weight download on first run:

```bash
python3 benchmarks/benchmark_comfyui_serving.py \
  --model wan22 --task i2v \
  --download-models --comfyui-base-dir /path/to/ComfyUI \
  --num-requests 50 --max-concurrency 4 \
  --host http://127.0.0.1:8188
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Model name (e.g. `wan22`) |
| `--task` | *(required)* | Task type (e.g. `i2v`) |
| `--host` | `http://127.0.0.1:8188` | ComfyUI base URL |
| `--num-requests` | `50` | Total requests to submit |
| `--max-concurrency` | `8` | Max in-flight requests |
| `--request-rate` | `0` | Requests/sec; `0` = fire immediately |
| `--poisson` | off | Poisson inter-arrival when `--request-rate > 0` |
| `--num-images` | `20` | Synthetic images if VBench download unavailable |
| `--prompts-dir` | `benchmarks/prompts/<model>_<task>/` | Prompt JSON output directory |
| `--download-models` | off | Download model weights before benchmarking |
| `--comfyui-base-dir` | — | ComfyUI root (required with `--download-models`) |
| `--output-json` | — | Write full per-request results to a JSON file |

## Output

```
benchmark: 100%|█████████████| 5/5 [02:58<00:00, 35.73s/req, succeeded=5]

=== ComfyUI Serving Benchmark Summary ===
requests_total:   5
requests_success: 5
requests_failed:  0
wall_time_s:      178.652
throughput_req_s: 0.028
latency_p50_s:    109.594
latency_p90_s:    164.840
latency_p95_s:    171.744
latency_p99_s:    177.266
latency_mean_s:   109.781
latency_max_s:    178.647
execution_mean_ms:  35465.21
execution_p95_ms:   39685.06

--- Per-node execution time (mean ms across successful requests) ---
  KSamplerAdvanced (130:110): mean=12827.5  p95=14264.0  n=5
  KSamplerAdvanced (130:111): mean=12726.4  p95=13822.2  n=5
  VAEDecode (130:129): mean=3439.0  p95=3467.6  n=5
  SaveVideo (108): mean=2844.7  p95=3280.0  n=5
  WanImageToVideo (130:128): mean=2367.7  p95=2595.9  n=5
  CLIPTextEncode (130:125): mean=1785.0  p95=1785.0  n=1
  CLIPLoader (130:105): mean=700.7  p95=700.7  n=1
  LoadImage (97): mean=518.4  p95=970.0  n=5
  VAELoader (130:106): mean=507.7  p95=507.7  n=1
  CLIPTextEncode (130:107): mean=223.4  p95=223.4  n=1
  UNETLoader (130:122): mean=122.2  p95=122.2  n=1
  LoraLoaderModelOnly (130:126): mean=68.1  p95=68.1  n=1
  UNETLoader (130:123): mean=65.9  p95=65.9  n=1
  LoraLoaderModelOnly (130:127): mean=36.2  p95=36.2  n=1
  ModelSamplingSD3 (130:109): mean=1.0  p95=1.0  n=1
  ModelSamplingSD3 (130:124): mean=0.9  p95=0.9  n=1
  CreateVideo (130:117): mean=0.7  p95=1.1  n=5
```

> **Note:** Nodes with `n=1` (e.g. model loaders) are cached by ComfyUI after
> the first request and skipped in subsequent executions, so they only appear
> once across the benchmark run.
