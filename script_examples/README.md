# API Script Examples

This folder contains small Python examples for driving a local ComfyUI server from code.

## Files

- `basic_api_example.py`: submits a workflow exported from `File -> Export (API)`.
- `websockets_api_example.py`: submits a workflow, waits for completion over websocket, and then fetches generated images from history.
- `websockets_api_example_ws_images.py`: streams image bytes directly through the websocket with `SaveImageWebsocket` instead of saving files to disk first.

## What These Examples Use

These scripts target the local scripting routes exposed by `server.py`:

- `POST /prompt`
- `GET /history/{prompt_id}`
- `GET /queue`
- `GET /view`
- `GET /ws`

For the newer documented REST API, see [`openapi.yaml`](../openapi.yaml). In particular:

- `POST /api/prompt`
- `GET /api/queue`
- `GET /api/jobs/{job_id}`

## Usage Notes

1. Start ComfyUI locally, which defaults to `127.0.0.1:8188`.
2. Load or create a workflow in the UI.
3. Export the workflow with `File -> Export (API)`.
4. Replace the sample `prompt_text` in one of these scripts with the exported JSON if needed.
5. Make sure the referenced models and nodes are available in your local ComfyUI install.

## Dependencies

- `basic_api_example.py` uses only the Python standard library.
- The websocket examples also require `websocket-client`.

Install it with:

```bash
pip install websocket-client
```
