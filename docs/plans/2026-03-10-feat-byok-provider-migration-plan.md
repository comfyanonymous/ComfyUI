---
title: "feat: BYOK Provider Migration -- Replace ComfyOrg Proxy with Direct API Keys"
type: feat
status: active
date: 2026-03-10
origin: docs/brainstorms/2026-03-10-byok-provider-migration-brainstorm.md
---

# BYOK Provider Migration

## Enhancement Summary

**Deepened on:** 2026-03-10
**Agents used:** security-sentinel, performance-oracle, architecture-strategist, pattern-recognition-specialist, code-simplicity-reviewer, kieran-python-reviewer, fal.ai-polling-researcher, comfyui-frontend-researcher, gemini-imagegen-skill

### Key Improvements
1. **Split Phase 1 into 1a/1b** for incremental migration (add new auth before removing old)
2. **Connection pooling** -- current codebase creates a new aiohttp session per request; must fix before migration
3. **Security hardening** -- domain allowlist for auth headers, model_id validation, header redaction, custom exception type
4. **Simplification** -- delete unavailable providers instead of maintaining dead code; use `dict` for fal.ai results instead of per-provider Pydantic; drop `check_api_key()` (let auth functions raise); consider `fal-client` SDK
5. **Combined `fal_run()` helper** -- wraps submit+poll+fetch into one call to reduce duplication across 20+ files
6. **Gemini JPEG gotcha** -- Gemini returns JPEG by default; must check `mime_type`, not assume PNG

### Critical Findings from Research
- **Security C-1:** Request logger at `request_logger.py:103` writes API keys to disk in plaintext. Must implement header redaction BEFORE any new auth code.
- **Security C-3:** Generic fal.ai node SSRF risk -- `model_id` input must be validated against strict regex, and auth headers must only be sent to allowlisted domains.
- **Performance P-1:** `_request_base` creates a new `aiohttp.ClientSession` per request (line 628). Post-migration, this means fresh TLS handshake per poll cycle to 3-5 different hosts. Implement per-host connection pooling.
- **Architecture A-1:** Phase 1 as written breaks all nodes at once. Split into additive (1a) and removal (1b) sub-phases.

---

## Overview

Replace the ComfyOrg proxy (`api.comfy.org`) with bring-your-own-key direct API access. Two environment variables cover all models:

- `GOOGLE_API_KEY` -- Gemini, Imagen, Veo (direct to `generativelanguage.googleapis.com`)
- `FAL_API_KEY` -- everything else via fal.ai (Flux, Kling, Luma, MiniMax, etc.)

ComfyOrg auth is completely removed. Each node shows a green/red/grey badge indicating key status. Nodes without fal.ai equivalents are deleted.

## Problem Statement

This fork is for personal use. The ComfyOrg proxy adds an unnecessary billing middleman. Direct API keys are cheaper and more transparent. Two keys (Google + fal.ai) cover the full model catalog.

## Proposed Solution

Rewire all existing `nodes_*.py` files in-place to hit real provider APIs instead of the ComfyOrg proxy. Replace the auth system, upload helpers, and Pydantic models. Add a generic fal.ai node for arbitrary model IDs. Delete node files for providers not available on fal.ai. (see brainstorm: `docs/brainstorms/2026-03-10-byok-provider-migration-brainstorm.md`)

## Technical Approach

### Architecture

```
Before:                              After:
Node → ApiEndpoint(/proxy/...)       Node → ApiEndpoint(https://queue.fal.run/...)
     → _request_base                      → _request_base
     → prepend api.comfy.org              → use absolute URL as-is
     → add ComfyOrg auth headers          → add provider auth from env var
     → ComfyOrg proxy                     → provider API directly
     → actual provider
```

### Provider Routing Map

| Current Node File | BYOK Target | fal.ai Model ID | Status |
|---|---|---|---|
| `nodes_gemini.py` | Google direct | N/A | Direct API |
| `nodes_veo2.py` | Google direct | N/A | Direct API |
| `nodes_bfl.py` | fal.ai | `fal-ai/flux-pro/v1.1-ultra`, `fal-ai/flux-kontext/pro`, etc. | Available |
| `nodes_openai.py` | fal.ai | `fal-ai/gpt-image-1/text-to-image`, `fal-ai/gpt-image-1.5/edit` | Available |
| `nodes_stability.py` | fal.ai | `fal-ai/stable-diffusion-v35-medium`, `fal-ai/fast-sdxl` | Available |
| `nodes_kling.py` | fal.ai | `fal-ai/kling-video/v2/master/image-to-video`, etc. | Available |
| `nodes_luma.py` | fal.ai | `fal-ai/luma-dream-machine/ray-2` | Available |
| `nodes_minimax.py` | fal.ai | `fal-ai/minimax/video-01-director` | Available |
| `nodes_ideogram.py` | fal.ai | `fal-ai/ideogram/v3` | Available |
| `nodes_recraft.py` | fal.ai | `fal-ai/recraft/v3/text-to-image` | Available |
| `nodes_elevenlabs.py` | fal.ai | `fal-ai/elevenlabs/tts/turbo-v2.5`, etc. | Available |
| `nodes_sora.py` | fal.ai | `fal-ai/sora-2/text-to-video` | Available |
| `nodes_meshy.py` | fal.ai | `fal-ai/meshy/v6/image-to-3d` | Available |
| `nodes_ltxv.py` | fal.ai | `fal-ai/ltx-video-v097` | Available |
| `nodes_bytedance.py` | fal.ai | `fal-ai/seedream-4.5` | Available |
| `nodes_pixverse.py` | fal.ai | `fal-ai/pixverse/v3.5/image-to-video` | Available |
| `nodes_wan.py` | fal.ai | `fal-ai/wan-pro/image-to-video` | Available |
| `nodes_hunyuan3d.py` | fal.ai | `fal-ai/hunyuan3d/v2` | Available |
| `nodes_vidu.py` | fal.ai | `fal-ai/vidu/q3-pro/image-to-video` | Available |
| `nodes_bria.py` | fal.ai | `fal-ai/bria/text-to-image/hd`, `fal-ai/bria/eraser`, `fal-ai/bria/background/remove` | Available |
| `nodes_rodin.py` | fal.ai | `fal-ai/hyper3d/rodin/v2` | Available |
| `nodes_runway.py` | N/A | Not on fal.ai | **Delete** |
| `nodes_tripo.py` | N/A | Not on fal.ai | **Delete** |
| `nodes_magnific.py` | N/A | Not on fal.ai | **Delete** |
| `nodes_topaz.py` | N/A | Not on fal.ai | **Delete** |
| `nodes_moonvalley.py` | N/A | Not on fal.ai | **Delete** |
| `nodes_grok.py` | N/A | Not on fal.ai | **Delete** |
| `nodes_hitpaw.py` | N/A | Not on fal.ai | **Delete** |
| `nodes_wavespeed.py` | N/A | Not on fal.ai | **Delete** |

### Google API Endpoints

All via `generativelanguage.googleapis.com` with `x-goog-api-key` header:

| Operation | Endpoint | Response Pattern |
|---|---|---|
| Gemini text/image gen | `POST /v1beta/models/{model}:generateContent` | Sync -- `candidates[].content.parts[]` |
| Imagen image gen | `POST /v1beta/models/{model}:predict` | Sync -- `predictions[].bytesBase64Encoded` |
| Veo video gen (submit) | `POST /v1beta/models/{model}:predictLongRunning` | Returns `{"name": "models/.../operations/..."}` |
| Veo video gen (poll) | `GET /v1beta/{operation_name}` | `{"done": true/false, "response": {...}}` |
| Veo video download | `GET /v1beta/files/{id}:download?alt=media` | Binary video data |
| File upload | `POST /upload/v1beta/files` | Resumable upload, returns file URI |

### fal.ai API Endpoints

All via `queue.fal.run` (async) or `fal.run` (sync) with `Authorization: Key {FAL_API_KEY}`:

| Operation | Endpoint | Response Pattern |
|---|---|---|
| Submit (async) | `POST queue.fal.run/{model_id}` | `{"request_id": "...", "status_url": "...", "response_url": "..."}` |
| Poll status | `GET queue.fal.run/{model_id}/requests/{id}/status` | `{"status": "IN_QUEUE"/"IN_PROGRESS"/"COMPLETED"}` |
| Fetch result | `GET queue.fal.run/{model_id}/requests/{id}` | Model-specific output (images, video, etc.) |
| Submit (sync) | `POST fal.run/{model_id}` | Direct result (for fast models only) |
| File upload | `POST rest.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3` | Returns `{upload_url, file_url}` |
| File upload (PUT) | `PUT {upload_url}` | Uploads to presigned URL |
| Cancel | `PUT queue.fal.run/{model_id}/requests/{id}/cancel` | Best-effort; only works while IN_QUEUE |

### Implementation Phases

#### Phase 1a: Add New Infrastructure (Additive)

Add new auth helpers and upload functions **alongside** existing ones. No existing code is removed yet, so all current nodes continue working throughout this phase.

**Files to modify:**

1. **`comfy_api_nodes/util/_helpers.py`** (92 lines)
   - Keep existing `get_auth_header()` and `default_base_url()` for now
   - Add `get_google_auth_header() -> dict[str, str]`: reads `GOOGLE_API_KEY` from env, returns `{"x-goog-api-key": key}`
   - Add `get_fal_auth_header() -> dict[str, str]`: reads `FAL_API_KEY` from env, returns `{"Authorization": f"Key {key}"}`
   - Both raise `MissingApiKeyError` if key is missing or empty (after `.strip()`)

2. **`comfy_api_nodes/util/common_exceptions.py`**
   - Add `MissingApiKeyError(Exception)` -- do NOT use `EnvironmentError` (it aliases `OSError` and would be caught by existing `except (ClientError, OSError)` blocks in `client.py:811` and `upload_helpers.py:343`)

3. **`comfy_api_nodes/util/client.py`** (951 lines)
   - **Connection pooling** (CRITICAL): Replace per-request `aiohttp.ClientSession()` creation at line 628 with a per-host session registry. Current code creates and destroys a session for every HTTP call. Post-migration, each Veo poll cycle (5+ requests) would incur separate TLS handshakes to the same host.
   - Add domain allowlist for auth headers: `x-goog-api-key` only sent to `*.googleapis.com`; `Authorization: Key` only sent to `*.fal.run`, `*.fal.ai`, `*.fal.media`
   - Add fal.ai concurrency semaphore: `asyncio.Semaphore(2)` to prevent wasteful 429 retry storms when >2 fal.ai nodes execute concurrently
   - `_friendly_http_message` (line 511-519): Add provider-appropriate messages for Google (403 = invalid API key) and fal.ai (401 = invalid key). Keep existing ComfyOrg messages until Phase 1b.
   - Sanitize error response bodies before including in exceptions -- parse known provider error formats (Google's `error.message`, fal.ai's `detail[].msg`), strip raw JSON dumps that could contain reflected auth info

4. **`comfy_api_nodes/util/request_logger.py`** (MUST DO FIRST)
   - Implement `_redact_headers(headers: dict) -> dict` that replaces values of `Authorization`, `x-goog-api-key`, `X-API-KEY`, and any header matching `*key*`/`*token*` with `[REDACTED]`
   - Apply to BOTH request headers (line 103) and response headers (line 114)
   - This MUST be implemented before any new auth headers are introduced

5. **`comfy_api_nodes/util/upload_helpers.py`** (388 lines)
   - Add `upload_file_to_fal(file_bytes: BytesIO, mime_type: str) -> str`: POST to `rest.fal.ai/storage/upload/initiate`, PUT file to returned URL, return CDN URL. Validate returned `upload_url` domain matches `*.fal.ai`/`*.fal.run`.
   - Add `upload_file_to_google(file_bytes: BytesIO, mime_type: str, display_name: str) -> str`: POST to Google Files API resumable upload, return `files/{name}` URI
   - Add thin convenience wrappers: `upload_image_to_fal(cls, image_tensor) -> str` that handles tensor-to-BytesIO conversion
   - Keep existing `upload_images_to_comfyapi` intact until Phase 1b
   - Use streaming upload for files >10MB (don't read entire file into memory via `.read()`)

6. **`comfy_api_nodes/util/download_helpers.py`** (298 lines)
   - Add optional `headers: dict | None = None` parameter to `download_url_to_bytesio` so callers can pass provider-specific auth headers for authenticated downloads (Google requires `x-goog-api-key` on download URLs)
   - Keep existing relative URL detection for now (removed in Phase 1b)

7. **Startup key status log** (add to node init path)
   ```python
   for var in ("GOOGLE_API_KEY", "FAL_API_KEY"):
       status = "configured" if os.environ.get(var, "").strip() else "NOT SET"
       logging.info("BYOK: %s: %s", var, status)
   ```

**Acceptance criteria for Phase 1a:**
- [x] `get_google_auth_header()` reads from `GOOGLE_API_KEY` env var, raises `MissingApiKeyError` if empty/unset
- [x] `get_fal_auth_header()` reads from `FAL_API_KEY` env var, raises `MissingApiKeyError` if empty/unset
- [x] API keys are redacted in request AND response logs
- [x] Auth headers only sent to allowlisted domains
- [x] Upload helpers for fal.ai and Google exist alongside old ones
- [x] Connection pooling implemented with per-host session registry
- [x] fal.ai concurrency semaphore initialized
- [x] All existing nodes still work (nothing removed yet)

#### Phase 1b: Remove Old Infrastructure

After Phases 2-4 migrate all nodes, remove the old ComfyOrg auth system.

- [ ] Remove `get_auth_header()` and `default_base_url()` from `_helpers.py` (still referenced by client.py fallback path)
- [x] Remove `auth_token_comfy_org` and `api_key_comfy_org` from `Hidden` enum in `_io.py`
- [x] Remove auto-injection in `Schema.finalize()` (line 1524-1529)
- [ ] Make `_request_base` reject relative URLs (raise `ValueError`)
- [ ] Remove relative URL detection in `download_helpers.py`
- [x] Remove old ComfyOrg messages from `_friendly_http_message` (replaced with Google/fal.ai messages)
- [ ] Remove `_diagnose_connectivity` health check against `api.comfy.org/health`
- [ ] Delete old `upload_images_to_comfyapi` and related functions

#### Phase 2: Google Direct Nodes

Migrate Gemini and Veo nodes to hit Google APIs directly.

**Files to modify:**

1. **`comfy_api_nodes/nodes_gemini.py`** (1012 lines)
   - Replace `GEMINI_BASE_ENDPOINT = "/proxy/vertexai/gemini"` (line 48) with `GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"`
   - Update all `ApiEndpoint(path=...)` calls to use absolute Google API URLs
   - For text/image generation: `{GEMINI_BASE_URL}/{model}:generateContent`
   - For Imagen: `{GEMINI_BASE_URL}/{model}:predict`
   - Add `headers=get_google_auth_header()` to every `ApiEndpoint`
   - Replace `upload_images_to_comfyapi()` calls (line 99-114) with inline base64 `inlineData` for all images (Google supports up to 100MB per request)
   - Remove `uploadImagesToStorage` field usage
   - Remove `hidden=[IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org]` from all schemas

2. **`comfy_api_nodes/apis/gemini.py`** (242 lines)
   - Remove `uploadImagesToStorage` field from `GeminiImageGenerateContentRequest` (line 148)
   - Verify all request/response models match Google's direct API format (they're already close)
   - Add Imagen-specific request/response models (`ImagenPredictRequest`, `ImagenPredictResponse`) if not already present

3. **`comfy_api_nodes/nodes_veo2.py`** (561 lines)
   - Replace proxy paths `/proxy/veo/{model}/generate` with `{GEMINI_BASE_URL}/{model}:predictLongRunning`
   - Replace polling path `/proxy/veo/{model}/poll` with `GET https://generativelanguage.googleapis.com/v1beta/{operation_name}` (where `operation_name` comes from the submit response)
   - Add `headers=get_google_auth_header()` to all endpoints
   - Update download to use `GET /v1beta/files/{id}:download?alt=media` with API key header (pass via new `headers` parameter on download helpers)
   - For image inputs (first/last frame), use inline base64 instead of ComfyOrg upload
   - Use adaptive polling: 2s for first 10s, 5s for 10-60s, 10s after 60s

4. **`comfy_api_nodes/apis/veo.py`** (100 lines)
   - Update `VeoGenVidResponse` to match direct API response: `{"name": "models/.../operations/..."}` instead of proxy wrapper
   - Update `VeoGenVidPollResponse` to match direct API: `{"done": bool, "response": {"generateVideoResponse": {"generatedSamples": [...]}}}` or `{"error": {...}}`
   - Update `VeoGenVidRequest` if the proxy was transforming the request format

### Research Insights for Phase 2

**Gemini Image Generation Gotcha:** Gemini returns JPEG by default regardless of what you name the output file. Always check `part.inline_data.mime_type` -- do not assume PNG. Saving a JPEG as `.png` creates a JPEG with a PNG extension, causing "Image does not match media type" errors downstream.

**Two image gen systems:** Gemini native image gen (`generateContent` with `responseModalities: ["IMAGE"]`) returns inline base64 in `candidates[].content.parts[].inline_data`. Imagen (`predict`) returns in `predictions[].bytesBase64Encoded`. These are different endpoints and response shapes.

**File size strategy:** Use inline base64 for images <1MB (thumbnails, reference images). Use Google Files API resumable upload for anything larger. Current code splits at image 10 (first 10 as fileUri, rest as inline) -- similar threshold but base on size, not count.

**Acceptance criteria for Phase 2:**
- [x] Gemini text generation works via direct Google API
- [x] Gemini image generation (Nano Banana) works via direct Google API
- [x] Imagen image generation works via `:predict` endpoint
- [x] Veo 2/3 video generation works via `:predictLongRunning` + poll
- [x] Image inputs use base64 inline data (no ComfyOrg upload)
- [ ] JPEG vs PNG mime type handled correctly in Gemini responses

#### Phase 3: fal.ai Infrastructure + Generic Node

Build the fal.ai integration layer and the generic fal.ai node. **Also resolve TBD entries** for `nodes_bria.py` and `nodes_rodin.py` before Phase 4 begins.

**Files to create:**

1. **`comfy_api_nodes/apis/fal.py`** (new file)
   - `FalQueueSubmitResponse`: `request_id`, `response_url`, `status_url`, `cancel_url`
   - `FalQueueStatusResponse`: `status` (IN_QUEUE/IN_PROGRESS/COMPLETED), `queue_position`, `logs`, `response_url`
   - `FalError`: `detail` array with `loc`, `msg`, `type`, `ctx`
   - Use Pydantic default `extra = "ignore"` (NOT `extra = "allow"` -- silently swallows typos)

2. **`comfy_api_nodes/nodes_fal.py`** (new file)
   - `FalGenericNode(IO.ComfyNode)`: accepts `model_id` (string), `input_json` (string, valid JSON), optional `image` (IMAGE tensor)
   - **Validate `model_id`** against strict regex: `^[a-zA-Z0-9_-]+(/[a-zA-Z0-9_.-]+)*$` -- reject `..`, `://`, `?`, `#`, whitespace (prevents SSRF)
   - Schema: `is_api_node=True`
   - Execute: parse input JSON, upload image if provided, submit to fal.ai queue, poll until complete, download result images/video immediately (fal.ai CDN URLs expire after 7 days)
   - `FalExtension(ComfyExtension)` + `comfy_entrypoint()`

**fal.ai helpers -- add to `comfy_api_nodes/util/client.py`** (thin wrappers around existing `sync_op`/`poll_op`, NOT a separate file):

3. **`fal_run(cls, model_id, data, *, estimated_duration=None) -> dict`**
   - Combined submit + poll + fetch in one call. Wraps the three steps below.
   - 20+ node files will call this; having it as one function prevents duplication.

4. **`fal_submit(cls, model_id, data) -> FalQueueSubmitResponse`**
   - `sync_op(cls, ApiEndpoint(f"https://queue.fal.run/{model_id}", "POST", headers=get_fal_auth_header()), data=data, response_model=FalQueueSubmitResponse)`

5. **`fal_poll(cls, status_url) -> FalQueueStatusResponse`**
   - Use existing `poll_op` with `status_extractor=lambda r: r["status"]`, `completed_statuses=["COMPLETED"]`, `failed_statuses=[]`
   - Map fal statuses: `IN_QUEUE`/`IN_PROGRESS` → keep polling, `COMPLETED` → done

6. **`fal_fetch_result(cls, response_url) -> dict`**
   - `sync_op_raw` GET to response URL with fal auth header
   - Returns `dict[str, Any]` (model-specific output)

7. **Resolve TBD entries:**
   - [ ] Verify `nodes_bria.py` against fal.ai model catalog -- assign model ID or move to delete list
   - [ ] Verify `nodes_rodin.py` against fal.ai model catalog -- assign model ID or move to delete list
   - TBD resolution is a **prerequisite** for Phase 4

### Research Insights for Phase 3

**Consider `fal-client` SDK:** The official `fal-client` Python SDK (`pip install fal-client`) handles retry, cancellation, timeout, and auth automatically. `fal_client.subscribe()` / `submit()` + `iter_events()` would replace the raw HTTP integration. Tradeoff: adds a dependency but eliminates manual queue management. For personal use, the SDK simplicity may be worth it.

**fal.ai CDN URLs expire after 7 days.** Always download result bytes immediately after completion. Never persist fal.ai URLs as references.

**Cancellation is best-effort.** `PUT .../cancel` only works when status is `IN_QUEUE`. Once processing starts, the request completes (and you're billed). Always attempt cancellation on user interrupt, but also stop your local polling loop.

**Concurrency: 2 tasks on standard tier.** Additional requests are queued server-side (not rejected). The client-side semaphore from Phase 1a prevents wasteful 429 retries.

**Acceptance criteria for Phase 3:**
- [x] Generic fal.ai node can submit to any model by ID
- [x] `fal_run()` submit/poll/fetch cycle works end-to-end
- [x] File upload to fal.ai CDN works (images, video, audio)
- [x] Generic node handles both image and video outputs
- [x] fal.ai errors are translated to user-friendly messages
- [x] `model_id` validated against strict regex (no SSRF)
- [x] TBD entries for `nodes_bria.py` and `nodes_rodin.py` resolved (both available on fal.ai)

#### Phase 4: Migrate Non-Google Nodes to fal.ai

Rewire each existing node file to use fal.ai instead of the ComfyOrg proxy. This is the largest phase by file count but follows a repeatable pattern.

**Migration pattern per node file:**

For each `nodes_*.py`:
1. Replace proxy endpoint paths with fal.ai model IDs (use module-level constants: `FAL_MODEL_FLUX_PRO_ULTRA = "fal-ai/flux-pro/v1.1-ultra"`)
2. Replace `sync_op`/`poll_op` calls with `fal_run()` (or `fal_submit`+`fal_poll`+`fal_fetch_result` for unusual flows)
3. Replace `upload_images_to_comfyapi` calls with `upload_file_to_fal`
4. Remove ComfyOrg hidden fields from schema
5. Update response parsing to extract from fal.ai's `dict` output (use plain dict access, not new Pydantic models)
6. Remove `price_badge` entries (ComfyOrg pricing no longer applies)
7. Download result URLs immediately (images, video, audio) -- don't store fal.ai CDN URLs

**For each corresponding `apis/*.py`:**
- For fal.ai-routed providers, **delete the per-provider Pydantic models** (they modeled the ComfyOrg proxy's API). Use `dict[str, Any]` for fal.ai results and build request dicts inline. The fal.ai queue envelope models in `apis/fal.py` are shared by all fal.ai nodes.

**Migration order (simple → complex):**

Batch 1 -- Synchronous image generation (simplest, establishes patterns):
- [x] `nodes_recraft.py` → `fal-ai/recraft/v3/text-to-image` (6 nodes) -- placeholder paths, needs fal.ai schema verification
- [x] `nodes_ideogram.py` → `fal-ai/ideogram/v3` (4 nodes) -- fully migrated to fal_run
- [x] `nodes_bfl.py` → `fal-ai/flux-pro/v1.1-ultra`, `fal-ai/flux-kontext/pro`, etc. (10 nodes) -- fully migrated to fal_run
- [x] `nodes_stability.py` → `fal-ai/stable-diffusion-v35-medium` (5 nodes) -- fully migrated to fal_run
- [x] `nodes_bria.py` → `fal-ai/bria/text-to-image/hd` (4 nodes) -- fully migrated to fal_run

Batch 2 -- OpenAI / Sora:
- [x] `nodes_openai.py` → `fal-ai/gpt-image-1/text-to-image` (4 nodes) -- fully migrated to fal_run
- [x] `nodes_sora.py` → `fal-ai/sora-2/text-to-video` (2 nodes) -- fully migrated to fal_run

Batch 3 -- Video generation (async/poll):
- [x] `nodes_kling.py` → `fal-ai/kling-video/v2/master/*` (24 nodes) -- placeholder paths, needs fal.ai schema verification
- [x] `nodes_luma.py` → `fal-ai/luma-dream-machine/ray-2` (5 nodes) -- fully migrated to fal_run
- [x] `nodes_minimax.py` → `fal-ai/minimax/video-01-director` (4 nodes) -- fully migrated to fal_run
- [x] `nodes_ltxv.py` → `fal-ai/ltx-video-v097` (3 nodes) -- fully migrated to fal_run
- [x] `nodes_pixverse.py` → `fal-ai/pixverse/v3.5/*` (4 nodes) -- fully migrated to fal_run
- [x] `nodes_wan.py` → `fal-ai/wan-pro/*` (5 nodes) -- fully migrated to fal_run
- [x] `nodes_vidu.py` → `fal-ai/vidu/q3-pro/*` (13 nodes) -- fully migrated to fal_run
- [x] `nodes_bytedance.py` → `fal-ai/seedream-4.5` (4 nodes) -- fully migrated to fal_run

Batch 4 -- Audio:
- [x] `nodes_elevenlabs.py` → `fal-ai/elevenlabs/tts/*` (7 nodes) -- placeholder paths, needs fal.ai schema verification

Batch 5 -- 3D:
- [x] `nodes_meshy.py` → `fal-ai/meshy/v6/image-to-3d` (3 nodes) -- placeholder paths, needs fal.ai schema verification
- [x] `nodes_hunyuan3d.py` → `fal-ai/hunyuan3d/v2` (6 nodes) -- placeholder paths, needs fal.ai schema verification
- [x] `nodes_rodin.py` → `fal-ai/hyper3d/rodin/v2` (3 nodes) -- placeholder paths, needs fal.ai schema verification

Batch 6 -- Delete unavailable providers:
- [x] Delete `nodes_runway.py` + `apis/runway.py`
- [x] Delete `nodes_tripo.py` + `apis/tripo.py`
- [x] Delete `nodes_magnific.py` + `apis/magnific.py`
- [x] Delete `nodes_topaz.py` + `apis/topaz.py`
- [x] Delete `nodes_moonvalley.py` + `apis/moonvalley.py`
- [x] Delete `nodes_grok.py` + `apis/grok.py`
- [x] Delete `nodes_hitpaw.py` + `apis/hitpaw.py`
- [x] Delete `nodes_wavespeed.py` + `apis/wavespeed.py`

### Research Insights for Phase 4

**Drift risk:** `nodes_kling.py` (3277 lines, 24 nodes) is larger than all Batch 1 files combined. Migrating it will likely force infrastructure refinements that affect earlier-migrated nodes. Run a **consistency sweep** after the last file to ensure early and late batches follow the same patterns.

**`nodes_recraft.py` exception:** Uses `multipart/form-data` file uploads via a custom `recraft_multipart_parser` (lines 73-119), not `upload_images_to_comfyapi`. The standard 7-step migration pattern doesn't cover this. Document as an exception.

**fal.ai parameter names differ from original providers.** Each fal.ai model wrapper may use different field names than the native API. For example, Kling's native `model_name` field may be different in fal.ai's wrapper. Verify each model's fal.ai input schema during migration by checking `https://fal.ai/models/{model_id}/api`.

#### Phase 5: Node-Level Key Status Badges

Add green/red badges to each API node in the ComfyUI frontend.

**Backend (`comfy_api/latest/_io.py`):**
- Add `required_api_key: str | None` to `Schema` and `NodeInfoV1` (static metadata)
- Add `api_key_status: str | None` to `NodeInfoV1`, computed server-side in `get_v1_info()`:
  - If `os.environ.get(required_api_key, "").strip()` → `"configured"` (green badge)
  - Else → `"missing"` (red badge)

**Frontend (JS extension via `WEB_DIRECTORY`):**
- Register a `Comfy.KeyStatusBadge` extension using `nodeCreated` hook
- Read `api_key_status` from `node.constructor.nodeData`
- Push an `LGraphBadge` to `node.badges` array with appropriate color:
  - Green (`#4CAF50`): key configured
  - Red (`#f44336`): key missing
- This uses ComfyUI's existing badge system (same as price badges) -- no core frontend modification needed

### Research Insights for Phase 5

**Do NOT create a separate `/api/key-status` endpoint.** The security review found it leaks key presence without authentication (ComfyUI has no auth middleware). Instead, embed `api_key_status` in the existing `/object_info` response via `NodeInfoV1`. The frontend already consumes this data for node rendering.

**Use opaque provider names** in any exposed data: `"google"` / `"fal"`, not env var names like `"GOOGLE_API_KEY"`.

**Badge refreshes on page reload only** (no polling needed). This is consistent with how other ComfyUI badges work.

**Simplification alternative:** For personal use, the startup log from Phase 1a may be sufficient. Phase 5 can be deferred or skipped entirely -- the execute-time error from `MissingApiKeyError` already tells you when a key is missing.

**Acceptance criteria for Phase 5:**
- [ ] Each API node displays correct badge color based on env var status
- [ ] Badge updates on page refresh
- [ ] No separate key-status endpoint (embedded in /object_info)

#### Phase 6: Cleanup

Final sweep for straggling references after all nodes are migrated and Phase 1b is complete.

- [x] Remove `--comfy-api-base` CLI argument from `comfy/cli_args.py`
- [ ] Remove price badges (ComfyOrg proxy pricing no longer applies)
- [ ] Grep codebase for remaining references to `api.comfy.org`, `comfy_org`, `proxy/` -- remove all
- [ ] Clean up `apis/__init__.py` (auto-generated from ComfyOrg OpenAPI spec -- remove or regenerate)
- [ ] Delete unused old upload helper functions
- [ ] Delete `apis/*.py` files for fal.ai-routed providers (replaced by `dict` access)

## System-Wide Impact

### Interaction Graph

1. Node `execute()` calls `get_google_auth_header()` or `get_fal_auth_header()` → reads `os.environ` → validates non-empty → constructs auth header
2. Domain allowlist check verifies auth header matches target URL domain
3. Node calls `sync_op` / `fal_run()` with absolute URL + auth headers → `_request_base` sends via pooled connection → provider API directly
4. For uploads: node calls `upload_file_to_fal()` or uses base64 inline → file reaches provider
5. For downloads: node calls `download_url_to_image_output()` with absolute URL + optional auth headers → downloads from provider CDN

### Error Propagation

- Missing env var → `get_*_auth_header()` raises `MissingApiKeyError` → caught by node → displayed in UI
- Empty env var → same `MissingApiKeyError` with message "...is set but empty. Please provide a valid API key."
- Invalid key → provider returns 401/403 → `_friendly_http_message` translates (Google: "Invalid API key", fal.ai: "Invalid API key") → displayed in UI
- Rate limit → provider returns 429 → existing retry logic in `_request_base` handles with exponential backoff
- fal.ai concurrency limit → client-side semaphore queues locally before sending
- Content policy → Google returns `candidates[].finishReason == "SAFETY"`, fal.ai returns `422` with `content_policy_violation` type → node-level handling
- Error response bodies sanitized before display (no raw JSON dumps with reflected auth info)

### State Lifecycle Risks

- No persistent state changes -- env vars are read-only, no database, no session state
- File uploads to fal.ai CDN are ephemeral (7-day default, configurable via `X-Fal-Object-Lifecycle-Preference` header)
- Google Files API uploads persist for 48 hours then auto-delete
- fal.ai result URLs must be downloaded immediately; never persisted as references

### API Surface Parity

- All `nodes_*.py` files share the same `ApiEndpoint` → `sync_op`/`fal_run()` → `_request_base` pipeline
- The auth change in `_helpers.py` affects every API node
- The upload change in `upload_helpers.py` affects 21 node files

## Acceptance Criteria

### Functional Requirements

- [ ] All Google nodes (Gemini, Imagen, Veo) work with `GOOGLE_API_KEY` env var
- [ ] All fal.ai-routed nodes work with `FAL_API_KEY` env var
- [ ] Generic fal.ai node accepts arbitrary model ID and JSON input
- [ ] File uploads work via provider-native mechanisms
- [ ] Async/poll flows work for Veo and fal.ai queue
- [ ] Error messages are provider-appropriate (no ComfyOrg references)

### Non-Functional Requirements

- [ ] API keys are never logged in debug output (request AND response headers redacted)
- [ ] Missing/empty key produces a clear, actionable `MissingApiKeyError`
- [ ] Auth headers only sent to allowlisted provider domains (no SSRF)
- [ ] Generic fal.ai node validates `model_id` input (no path traversal)
- [ ] No references to `api.comfy.org` remain in the codebase
- [ ] All provider connections use default TLS verification (never `ssl=False`)

### Quality Gates

- [ ] Each migrated node tested with a real API call (at least one per provider)
- [ ] All existing node names preserved for workflow compatibility
- [ ] Generic fal.ai node tested with at least 3 different model IDs
- [ ] Consistency sweep after Phase 4 verifies uniform patterns across all migrated files

## Dependencies & Prerequisites

- **Google API key** with paid tier (required for Veo and Imagen)
- **fal.ai API key** (standard tier, 2 concurrent tasks)
- **Python venv** -- this fork runs in its own virtual environment (see setup below)
- No new external library dependencies needed -- existing `aiohttp` client handles all HTTP. Optionally add `fal-client` SDK for simplified fal.ai integration.

**Time-sensitive:** Google Veo preview models (`veo-3.1-generate-preview`, `veo-3-generate-preview`, `veo-2-generate-preview`) are scheduled for deprecation on **April 2, 2026** (~3 weeks from plan date). Phase 2 should use preview model IDs initially but be prepared to update to GA model IDs as soon as they're published.

### Virtual Environment Setup

This fork uses its own isolated venv to avoid conflicts with system Python or other projects:

```bash
cd /Users/mkorovkin/workplace/mimos/comfyui-custom-fork
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional: pip install fal-client  (if using fal-client SDK)
```

All development and execution should happen within this venv. Add `.venv/` to `.gitignore`.

## Risk Analysis & Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| fal.ai model IDs change or become unavailable | Nodes break silently | Pin model IDs as module-level constants, easy to update |
| Google API response format changes | Pydantic validation fails | Use default `extra = "ignore"` on Pydantic models; define all consumed fields explicitly |
| fal.ai rate limit (2 concurrent tasks on standard tier) | Queue contention | Client-side `asyncio.Semaphore(2)` prevents wasteful retries; upgrade to premium if needed |
| Some nodes' fal.ai parameter names differ from original provider | Wrong inputs sent | Verify each model's fal.ai input schema at `fal.ai/models/{id}/api` during migration |
| Google Veo preview models deprecated April 2026 | Veo stops working | Monitor for GA model availability, update model IDs |
| SSRF via generic fal.ai node `model_id` | API key exfiltration | Strict regex validation + domain allowlist for auth headers |
| Gemini returns JPEG when PNG expected | Image format errors downstream | Always check `part.inline_data.mime_type`; convert explicitly if PNG needed |
| Connection pool exhaustion under load | Request failures | Configure `limit_per_host=10` for Google, `limit_per_host=4` for fal.ai |
| Early-migrated nodes drift from late-migrated patterns | Inconsistency | Consistency sweep after Phase 4; `nodes_kling.py` gets own sub-batch |

## Sources & References

### Origin

- **Brainstorm document:** [docs/brainstorms/2026-03-10-byok-provider-migration-brainstorm.md](docs/brainstorms/2026-03-10-byok-provider-migration-brainstorm.md) -- Key decisions: BYOK for everything, env vars for keys, rewire existing nodes, node-level badges, delete unavailable providers

### Internal References

- Auth system: `comfy_api_nodes/util/_helpers.py:30-35` (`get_auth_header`)
- HTTP client: `comfy_api_nodes/util/client.py:572-891` (`_request_base`)
- Session creation (perf issue): `comfy_api_nodes/util/client.py:628`
- Hidden fields: `comfy_api/latest/_io.py:1322-1337` (`Hidden` enum)
- Schema finalize: `comfy_api/latest/_io.py:1524-1529` (auto-inject)
- Upload helpers: `comfy_api_nodes/util/upload_helpers.py:188-217`
- Download helpers: `comfy_api_nodes/util/download_helpers.py:62-67`
- Request logger (key leak): `comfy_api_nodes/util/request_logger.py:103-104`
- Node discovery: `nodes.py:2463-2472` (`init_builtin_api_nodes`)
- Gemini nodes: `comfy_api_nodes/nodes_gemini.py:48` (`GEMINI_BASE_ENDPOINT`)
- Veo nodes: `comfy_api_nodes/nodes_veo2.py`
- Error messages: `comfy_api_nodes/util/client.py:511-519` (`_friendly_http_message`)
- Custom exceptions: `comfy_api_nodes/util/common_exceptions.py`

### External References

- Google Generative AI API: https://ai.google.dev/gemini-api/docs
- Google Veo documentation: https://ai.google.dev/gemini-api/docs/video
- Google Files API: https://ai.google.dev/gemini-api/docs/files
- Google Imagen: https://ai.google.dev/gemini-api/docs/imagen
- fal.ai Queue API: https://docs.fal.ai/model-apis/model-endpoints/queue
- fal.ai Authentication: https://docs.fal.ai/reference/platform-apis/authentication
- fal.ai Error Reference: https://docs.fal.ai/model-apis/errors
- fal.ai Model Explorer: https://fal.ai/explore/models
- fal-client Python SDK: https://pypi.org/project/fal-client/
- ComfyUI Extension API: https://docs.comfy.org/custom-nodes/js/javascript_objects_and_hijacking
- ComfyUI-fal-API (community integration): https://github.com/gokayfem/ComfyUI-fal-API
