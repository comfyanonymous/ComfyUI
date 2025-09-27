# Missing Models Downloader for ComfyUI

This extension adds automatic download functionality to ComfyUI's "Missing Models" dialog, allowing users to download missing models directly from the interface with a single click.

## Features

- **Automatic Download Buttons**: Adds a "Download" button next to each missing model in the dialog
- **Known Model Repository**: Pre-configured URLs for popular models (SDXL, SD1.5, VAEs, LoRAs, ControlNet, etc.)
- **Custom URL Support**: Prompts for URL if model is not in the repository
- **Real-time Progress**: Shows download percentage directly in the button
- **Bulk Downloads**: "Download All" button for multiple missing models
- **Smart Detection**: Automatically detects model type and places files in correct folders

## Installation

### For ComfyUI Frontend Repository

If you're using the separate ComfyUI frontend repository:

1. Clone the frontend repository:
```bash
git clone https://github.com/Comfy-Org/ComfyUI_frontend.git
cd ComfyUI_frontend
```

2. Copy this extension to the extensions folder:
```bash
cp -r path/to/frontend_extensions/missingModelsDownloader web/extensions/
```

3. Build and run the frontend as usual

### For ComfyUI with Built-in Frontend

If your ComfyUI still has the built-in frontend:

1. Copy the extension files to ComfyUI's web extensions:
```bash
cp -r frontend_extensions/missingModelsDownloader ComfyUI/web/extensions/core/
```

2. Restart ComfyUI

## Backend Requirements

The backend (ComfyUI server) must have the model downloader API endpoints installed. These are included in the `easy-download` branch or can be added manually:

1. Ensure `app/model_downloader.py` exists
2. Ensure `comfy_config/download_config.py` exists
3. Ensure `server.py` includes the download API endpoints

## How It Works

1. When ComfyUI shows the "Missing Models" dialog, the extension automatically detects it
2. Each missing model gets a "Download" button
3. For known models, clicking downloads immediately from the pre-configured source
4. For unknown models, you'll be prompted to enter the download URL
5. Download progress is shown as a percentage in the button
6. Once complete, the model is ready to use (refresh the node to see it)

## Supported Model Sources

The extension includes pre-configured URLs for models from:

- **HuggingFace**: Stable Diffusion models, VAEs, LoRAs
- **GitHub**: Upscale models (ESRGAN, RealESRGAN)
- **ComfyAnonymous**: Flux text encoders

## Configuration

Edit `missingModelsDownloader.js` to add more models to the repository:

```javascript
this.modelRepositories = {
    "checkpoints": {
        "your_model.safetensors": "https://url/to/model"
    }
}
```

## API Endpoints

The extension uses these backend API endpoints:

- `POST /api/models/download` - Start a download
- `GET /api/models/download/{task_id}` - Check download status
- `POST /api/models/download/{task_id}/cancel` - Cancel a download

## Troubleshooting

If the download buttons don't appear:

1. Check the browser console for errors
2. Ensure the backend API endpoints are working: `curl http://localhost:8188/api/models/downloads`
3. Verify the extension is loaded (should see `[MissingModelsDownloader]` in console)

## License

MIT