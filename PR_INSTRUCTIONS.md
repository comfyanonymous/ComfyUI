# Pull Request Instructions for Missing Models Downloader

## Frontend Changes

The frontend changes have been prepared in `/tmp/ComfyUI_frontend` on the branch `add-missing-models-downloader`.

### To submit the PR to ComfyUI_frontend:

1. **Fork the ComfyUI_frontend repository** on GitHub:
   - Go to https://github.com/Comfy-Org/ComfyUI_frontend
   - Click "Fork" in the top right

2. **Push the changes to your fork**:
   ```bash
   cd /tmp/ComfyUI_frontend
   git remote add fork https://github.com/YOUR_USERNAME/ComfyUI_frontend.git
   git push fork add-missing-models-downloader
   ```

3. **Create the Pull Request**:
   - Go to your fork on GitHub
   - Click "Pull requests" → "New pull request"
   - Set base repository: `Comfy-Org/ComfyUI_frontend` base: `main`
   - Set head repository: `YOUR_USERNAME/ComfyUI_frontend` compare: `add-missing-models-downloader`
   - Click "Create pull request"

4. **PR Title**: "Add Missing Models Downloader extension"

5. **PR Description**:
   ```markdown
   ## Summary

   This PR adds automatic download functionality to the "Missing Models" dialog, allowing users to download missing models directly from the interface without manually searching and moving files.

   ## Features

   - ✅ Automatically adds "Download" buttons to each missing model in the dialog
   - ✅ Pre-configured URLs for popular models:
     - SDXL & SD 1.5 checkpoints
     - VAE models (sdxl_vae, vae-ft-mse)
     - LoRA models (LCM LoRAs)
     - ControlNet models (canny, openpose, depth)
     - Upscale models (ESRGAN, RealESRGAN)
     - CLIP encoders
     - Flux models
   - ✅ Real-time download progress shown as percentage in button
   - ✅ Custom URL prompt for unknown models
   - ✅ "Download All" button for bulk downloads
   - ✅ TypeScript implementation with proper typing

   ## How it works

   1. The extension monitors for the "Missing Models" dialog
   2. When detected, it adds a download button next to each missing model
   3. Known models download immediately from pre-configured sources
   4. Unknown models prompt for a custom URL
   5. Progress is shown in real-time (0% → 100%)
   6. Models are automatically placed in the correct folders

   ## Backend Requirements

   This extension requires the ComfyUI backend to have the model download API endpoints:
   - `POST /api/models/download` - Start download
   - `GET /api/models/download/{task_id}` - Check status
   - `POST /api/models/download/{task_id}/cancel` - Cancel download

   These endpoints are available in ComfyUI with the model_downloader module.

   ## Testing

   1. Load a workflow with missing models
   2. The "Missing Models" dialog should appear with download buttons
   3. Click a button to download the model
   4. Progress should update in real-time
   5. Once complete, the model is ready to use

   ## Screenshots

   [Would add screenshots here of the dialog with download buttons]
   ```

## Backend Changes

The backend changes are already committed to your ComfyUI repository:

### Files Added/Modified:
- `app/model_downloader.py` - Core download functionality
- `comfy_config/download_config.py` - Configuration system
- `server.py` - API endpoints
- `nodes.py` - ModelDownloader node

### To use the complete system:

1. **Backend (ComfyUI)**: Your changes are ready in the current branch
2. **Frontend (ComfyUI_frontend)**: Submit the PR as described above

## Alternative: Direct Installation

If you want to use this immediately without waiting for PR approval:

### Frontend:
```bash
# Clone ComfyUI_frontend
git clone https://github.com/Comfy-Org/ComfyUI_frontend.git
cd ComfyUI_frontend

# Apply the changes
git remote add temp /tmp/ComfyUI_frontend
git fetch temp
git cherry-pick 8c2f919ba128

# Build and use
npm install
npm run build
```

### Backend:
Your ComfyUI already has all necessary backend components installed and ready.

## Notes

- The extension is fully TypeScript compliant
- It integrates seamlessly with the existing Missing Models dialog
- No modifications to existing code, only additions
- Backwards compatible - if backend endpoints don't exist, buttons simply won't work