# Brainstorm: BYOK Provider Migration

**Date:** 2026-03-10
**Status:** Reviewed

## What We're Building

Migrating all API-based generation nodes in this ComfyUI fork from the ComfyOrg proxy (`api.comfy.org`) to a bring-your-own-key (BYOK) model with two providers:

1. **Google API (direct)** — for Gemini, Veo 2/3, Nano Banana / ImageGen models
2. **fal.ai** — for everything else (Flux, SDXL, Kling, Runway, Stability, etc.) plus a generic node for any fal.ai model by ID

The ComfyOrg proxy and auth system is completely removed. No fallback to ComfyOrg keys.

## Why This Approach

- **Personal use** — no need for ComfyOrg billing proxy; direct API access is cheaper and more transparent
- **Two keys cover everything** — `GOOGLE_API_KEY` for Google models, `FAL_API_KEY` for the long tail via fal.ai
- **fal.ai as aggregator** — fal.ai hosts hundreds of models under one API key, avoiding the need for 20+ separate provider keys

## Key Decisions

### 1. Approach: Rewire Existing Nodes
Modify existing `nodes_*.py` files in-place rather than replacing them. This preserves node names and ComfyUI workflow compatibility.

### 2. Two Providers Only
- **Google direct API**: `nodes_gemini.py`, `nodes_veo2.py` → hit `generativelanguage.googleapis.com` / Vertex AI endpoints
- **fal.ai**: All other `nodes_*.py` files → hit `api.fal.ai` with equivalent model IDs
- **New generic node**: A `nodes_fal.py` with a flexible node that accepts any fal.ai model ID

### 3. API Key Storage: Environment Variables
- `GOOGLE_API_KEY` — for all Google model nodes
- `FAL_API_KEY` — for all fal.ai routed nodes
- Read via `os.environ` at request time (not startup), so keys can be set/changed without restart

### 4. No ComfyOrg Auth
- Remove `auth_token_comfy_org` and `api_key_comfy_org` from hidden inputs
- Remove `get_auth_header()` ComfyOrg logic
- Remove proxy URL routing in `_request_base`

### 5. Node-Level Key Status Indicators
Each node shows a green/red badge in the UI indicating whether its required API key environment variable is set. This is the "dev tools helper" — no separate settings page needed.

## Scope: Provider Mapping

| Current Node File | Current Provider | BYOK Target | Required Key |
|---|---|---|---|
| `nodes_gemini.py` | Google Gemini | Google API direct | `GOOGLE_API_KEY` |
| `nodes_veo2.py` | Google Veo 2/3 | Google API direct | `GOOGLE_API_KEY` |
| `nodes_bfl.py` | Black Forest Labs (Flux) | fal.ai | `FAL_API_KEY` |
| `nodes_openai.py` | OpenAI | fal.ai | `FAL_API_KEY` |
| `nodes_stability.py` | Stability AI | fal.ai | `FAL_API_KEY` |
| `nodes_runway.py` | Runway | fal.ai | `FAL_API_KEY` |
| `nodes_kling.py` | Kling | fal.ai | `FAL_API_KEY` |
| `nodes_luma.py` | Luma | fal.ai | `FAL_API_KEY` |
| `nodes_minimax.py` | MiniMax | fal.ai | `FAL_API_KEY` |
| `nodes_ideogram.py` | Ideogram | fal.ai | `FAL_API_KEY` |
| `nodes_recraft.py` | Recraft | fal.ai | `FAL_API_KEY` |
| `nodes_elevenlabs.py` | ElevenLabs | fal.ai | `FAL_API_KEY` |
| `nodes_sora.py` | OpenAI Sora | fal.ai | `FAL_API_KEY` |
| `nodes_meshy.py` | Meshy | fal.ai | `FAL_API_KEY` |
| `nodes_wavespeed.py` | WaveSpeed | fal.ai | `FAL_API_KEY` |
| `nodes_ltxv.py` | LTX Video | fal.ai | `FAL_API_KEY` |
| `nodes_bria.py` | Bria | fal.ai | `FAL_API_KEY` |
| `nodes_bytedance.py` | ByteDance | fal.ai | `FAL_API_KEY` |
| Others (`rodin`, `tripo`, `magnific`, `topaz`, `pixverse`, `moonvalley`, `wan`, `hunyuan3d`, `vidu`, `grok`, `hitpaw`) | Various | fal.ai (if available) | `FAL_API_KEY` |

**Note:** Not all providers in this table have confirmed fal.ai equivalents. The planning phase must audit fal.ai's model catalog to determine which nodes get working fal.ai routes vs. an "unavailable" badge.

## Technical Details (High Level)

### Auth System Changes
- `comfy_api_nodes/util/_helpers.py`: Replace `get_auth_header()` with `get_provider_auth_header(provider: str)` that reads from env vars
- `comfy_api_nodes/util/client.py`: `_request_base` always uses absolute URLs (no more relative proxy paths)
- `comfy_api/latest/_io.py`: Remove `auth_token_comfy_org` / `api_key_comfy_org` from `Hidden` enum and `HiddenHolder`

### Google API Integration
- Gemini: `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`
- Veo: `https://generativelanguage.googleapis.com/v1beta/models/{model}:predictLongRunning` (or Vertex AI equivalent)
- Auth: `x-goog-api-key: {GOOGLE_API_KEY}` header

### fal.ai Integration
- Base URL: `https://fal.run/{model_id}` (synchronous) or `https://queue.fal.run/{model_id}` (async/queued)
- Auth header: `Authorization: Key {FAL_API_KEY}`
- Submit + poll pattern maps well to existing `poll_op` infrastructure
- Generic node: user provides model ID string, node passes through inputs as JSON

### Node UI: Key Status Badge
- Each node checks `os.environ.get("GOOGLE_API_KEY")` or `os.environ.get("FAL_API_KEY")` at render time
- Display as a colored indicator (green = key present, red = missing) on the node widget
- Implementation likely in the node's `define_schema()` or as a custom widget

## Resolved Questions

1. **fal.ai model coverage gaps** → Keep nodes for unavailable providers but show a clear red "unavailable" badge. They exist in the UI but cannot run.

2. **File upload path** → Use provider-native upload mechanisms. fal.ai has its own upload endpoint; Google accepts base64 inline. Each provider path handles uploads correctly for that provider.

3. **Request/response model changes** → Rewrite Pydantic models to match real provider API schemas (Google API and fal.ai). More work upfront but correct and maintainable.
