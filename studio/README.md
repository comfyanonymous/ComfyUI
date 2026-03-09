# Studio

Film/TV production tool built on top of ComfyUI.

## Quick start

1. Make sure ComfyUI is running (default: `http://localhost:8188`)
2. From this directory:

```bash
./start.sh
```

Then open **http://localhost:8189** in your browser.

## Custom port / ComfyUI URL

```bash
./start.sh 8190 http://localhost:8188
```

Or with environment variables:

```bash
STUDIO_PORT=8190 COMFYUI_URL=http://localhost:8188 python3 server.py
```

## Project data

All project data is stored in `data/projects/`. Each project gets its own subdirectory:

```
data/projects/
  my-film/
    config.json          # project name, style settings
    characters/
      elena/
        config.json      # name, type (photomaker|lora), lora settings
        refs/            # uploaded reference photos
    scenes/
      a1b2c3d4.json      # scene description, characters, outputs
```

## Workflow templates

ComfyUI workflow JSON templates live in `workflows/`. The server selects the right template based on what's being generated:

| Template | Used when |
|---|---|
| `still_base.json` | Still image, no characters |
| `still_lora.json` | Still image with LoRA character |
| `still_photomaker.json` | Still image with PhotoMaker character refs |
| `video_base.json` | Video clip |

Edit these templates to change models, resolution, sampler settings, etc.
The `_role` field on `CLIPTextEncode` nodes (`"positive"` or `"negative"`) tells the server where to inject the scene prompt and negative prompt.

## Keeping ComfyUI up to date

Studio talks to ComfyUI only via its HTTP API. You can update your ComfyUI fork freely without touching anything in `studio/`.
