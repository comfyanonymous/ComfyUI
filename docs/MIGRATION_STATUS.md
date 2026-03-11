# BYOK Migration Status

**Date:** 2026-03-11

## Status: CORE MIGRATION COMPLETE

All ComfyOrg proxy infrastructure has been replaced with direct BYOK API access. Zero references to `api.comfy.org` remain.

## What's Done

### Infrastructure (Phase 1a + 1b)
- `MissingApiKeyError` exception in `common_exceptions.py`
- `get_google_auth_header()` / `get_fal_auth_header()` in `_helpers.py`
- `validate_auth_header_domain()` -- domain allowlist preventing SSRF
- `_redact_headers()` in `request_logger.py` -- redacts API keys from logs
- Per-host connection pooling in `client.py`
- `asyncio.Semaphore(2)` for fal.ai concurrency
- `.env` loading via `python-dotenv` in `main.py`
- Startup BYOK key status log in `nodes.py`
- **Old `get_auth_header()` / `default_base_url()` DELETED** -- no ComfyOrg auth remains
- **`_request_base` rejects relative URLs** -- all endpoints must be absolute
- **`_diagnose_connectivity()` no longer checks `api.comfy.org`**

### Upload Helpers (Complete)
- `upload_file_to_fal()`, `upload_image_to_fal()`, `upload_images_to_fal()`
- `upload_video_to_fal()`, `upload_audio_to_fal()`, `upload_3d_model_to_fal()`
- `upload_file_to_google()` (resumable upload)
- **Old `upload_*_to_comfyapi` functions DELETED** -- ~350 lines removed

### Google Direct (Phase 2)
- `nodes_gemini.py` -- all 4 nodes hit `generativelanguage.googleapis.com` directly
- `nodes_veo2.py` -- all 3 nodes use `:predictLongRunning` + poll

### fal.ai Infrastructure (Phase 3)
- `apis/fal.py` -- Pydantic models for queue envelope
- `fal_run()`, `fal_submit()`, `fal_poll()`, `fal_fetch_result()` in `client.py`
- `nodes_fal.py` -- generic fal.ai node (any model by ID)

### Node Migration (Phase 4) -- ALL 19 FILES COMPLETE
- 8 unavailable providers **deleted** (runway, tripo, magnific, topaz, moonvalley, grok, hitpaw, wavespeed)
- All 19 remaining node files fully migrated to `fal_run()` with proper fal.ai model IDs
- All upload calls migrated from `upload_*_to_comfyapi` to `upload_*_to_fal`
- All `__FAL_*__` placeholder strings eliminated
- All ComfyOrg hidden fields and price badges removed

### Cleanup (Phase 6) -- COMPLETE
- `--comfy-api-base` CLI arg removed
- `.env` added to `.gitignore`
- `apis/rodin.py` deleted (unused)
- Dead imports cleaned from all migrated node files
- Old upload functions deleted from `upload_helpers.py`
- Old auth functions deleted from `_helpers.py`
- Zero references to `api.comfy.org` in codebase

## What's Left

| Task | Effort |
|------|--------|
| Phase 5: Node-level key status badges in frontend (optional) | Medium |
| Verify fal.ai field names at runtime (TODO comments in files) | Ongoing |
| Commit all changes | -- |
