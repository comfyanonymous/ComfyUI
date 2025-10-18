import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// Missing Models Dialog Enhancer - Adds download buttons to the missing models dialog
app.registerExtension({
    name: "Comfy.MissingModelsDownloader",

    async setup() {
        console.log("[MissingModelsDownloader] Extension loading...");

        // Model repository with known URLs
        this.modelRepositories = {
            "checkpoints": {
                "sd_xl_base_1.0.safetensors": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
                "sd_xl_refiner_1.0.safetensors": "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors",
                "v1-5-pruned-emaonly.safetensors": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
                "v1-5-pruned.safetensors": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors",
                "v2-1_768-ema-pruned.safetensors": "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors"
            },
            "vae": {
                "vae-ft-mse-840000-ema-pruned.safetensors": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
                "sdxl_vae.safetensors": "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
                "sdxl.vae.safetensors": "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors"
            },
            "clip": {
                "clip_l.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
                "t5xxl_fp16.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors",
                "t5xxl_fp8_e4m3fn.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors"
            },
            "loras": {
                "lcm-lora-sdv1-5.safetensors": "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors",
                "lcm-lora-sdxl.safetensors": "https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors"
            },
            "controlnet": {
                "control_sd15_canny.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth",
                "control_sd15_openpose.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth",
                "control_sd15_depth.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_depth.pth",
                "control_v11p_sd15_canny.pth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth",
                "control_v11f1p_sd15_depth.pth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth"
            },
            "upscale_models": {
                "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                "realesr-general-x4v3.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                "4x-UltraSharp.pth": "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"
            },
            "unet": {
                "flux1-dev.safetensors": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
                "flux1-schnell.safetensors": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors"
            }
        };

        // Active downloads tracking
        this.activeDownloads = new Map();

        // Hook into the app to monitor for missing models dialog
        this.setupDialogMonitoring();
    },

    setupDialogMonitoring() {
        const self = this;

        // Monitor DOM mutations for dialog creation
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === 1) { // Element node
                        // Check for dialog containers
                        self.checkForMissingModelsDialog(node);
                    }
                });
            });
        });

        // Start observing
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        console.log("[MissingModelsDownloader] Dialog monitoring active");
    },

    checkForMissingModelsDialog(element) {
        // Look for the missing models dialog by its content
        const isDialog = element.classList && (
            element.classList.contains('p-dialog') ||
            element.classList.contains('comfy-modal') ||
            element.tagName === 'DIALOG'
        );

        if (!isDialog && element.querySelector) {
            const dialogs = element.querySelectorAll('dialog, .p-dialog, .comfy-modal');
            dialogs.forEach(dialog => this.checkForMissingModelsDialog(dialog));
            return;
        }

        const textContent = element.textContent || "";

        // Check for missing models dialog indicators
        if (textContent.includes("Missing Models") ||
            textContent.includes("When loading the graph") ||
            textContent.includes("models were not found")) {

            console.log("[MissingModelsDownloader] Found missing models dialog");

            // Add a small delay to ensure dialog is fully rendered
            setTimeout(() => {
                this.enhanceMissingModelsDialog(element);
            }, 100);
        }
    },

    enhanceMissingModelsDialog(dialogElement) {
        // Don't enhance twice
        if (dialogElement.dataset.enhancedWithDownloads) {
            return;
        }
        dialogElement.dataset.enhancedWithDownloads = "true";

        // Find model entries in the dialog
        const modelEntries = this.findModelEntries(dialogElement);

        if (modelEntries.length === 0) {
            console.log("[MissingModelsDownloader] No model entries found in dialog");
            return;
        }

        console.log(`[MissingModelsDownloader] Found ${modelEntries.length} missing models`);

        // Add download button to each model
        modelEntries.forEach(entry => {
            this.addDownloadButton(entry);
        });

        // Add "Download All" button if multiple models
        if (modelEntries.length > 1) {
            this.addDownloadAllButton(dialogElement, modelEntries);
        }
    },

    findModelEntries(dialogElement) {
        const entries = [];

        // Look for list items containing model paths
        const listItems = dialogElement.querySelectorAll('li, .model-item, [class*="missing"]');

        listItems.forEach(item => {
            const text = item.textContent || "";
            // Pattern: folder.filename or folder/filename
            if (text.match(/\w+[\.\/]\w+/)) {
                entries.push({
                    element: item,
                    text: text.trim()
                });
            }
        });

        // Also check for any divs or spans that might contain model names
        if (entries.length === 0) {
            const textElements = dialogElement.querySelectorAll('div, span, p');
            textElements.forEach(elem => {
                const text = elem.textContent || "";
                if (text.match(/\w+\.\w+/) && !elem.querySelector('button')) {
                    // Check if this looks like a model filename
                    const parts = text.split(/[\.\/]/);
                    if (parts.length >= 2 && this.looksLikeModelName(parts[parts.length - 1])) {
                        entries.push({
                            element: elem,
                            text: text.trim()
                        });
                    }
                }
            });
        }

        return entries;
    },

    looksLikeModelName(filename) {
        const modelExtensions = ['safetensors', 'ckpt', 'pt', 'pth', 'bin'];
        const lower = filename.toLowerCase();
        return modelExtensions.some(ext => lower.includes(ext));
    },

    addDownloadButton(entry) {
        const { element, text } = entry;

        // Parse model info from text
        const modelInfo = this.parseModelInfo(text);
        if (!modelInfo) return;

        // Create download button
        const btn = document.createElement('button');
        btn.textContent = 'Download';
        btn.style.cssText = `
            margin-left: 10px;
            padding: 4px 12px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
        `;

        // Check if we have a known URL
        const knownUrl = this.getKnownUrl(modelInfo.folder, modelInfo.filename);
        if (knownUrl) {
            btn.style.background = '#2196F3';
            btn.title = 'Download from known source';
        }

        btn.onclick = () => this.startDownload(modelInfo, btn);

        element.appendChild(btn);
        entry.button = btn;
    },

    parseModelInfo(text) {
        // Try different patterns
        const patterns = [
            /(\w+)\.(\w+(?:\.\w+)*)/,  // folder.filename
            /(\w+)\/(\w+(?:\.\w+)*)/,   // folder/filename
            /^(\w+(?:\.\w+)*)$/          // just filename
        ];

        for (const pattern of patterns) {
            const match = text.match(pattern);
            if (match) {
                if (match.length === 2) {
                    // Just filename, try to guess folder
                    return {
                        folder: this.guessFolder(match[1]),
                        filename: match[1]
                    };
                } else {
                    return {
                        folder: match[1],
                        filename: match[2]
                    };
                }
            }
        }

        return null;
    },

    guessFolder(filename) {
        const lower = filename.toLowerCase();
        if (lower.includes('vae')) return 'vae';
        if (lower.includes('lora')) return 'loras';
        if (lower.includes('control')) return 'controlnet';
        if (lower.includes('upscale') || lower.includes('esrgan')) return 'upscale_models';
        if (lower.includes('clip')) return 'clip';
        if (lower.includes('unet') || lower.includes('flux')) return 'unet';
        return 'checkpoints';
    },

    getKnownUrl(folder, filename) {
        const repo = this.modelRepositories[folder];
        if (repo && repo[filename]) {
            return repo[filename];
        }

        // Try alternate folders
        const alternateFolders = {
            'text_encoders': 'clip',
            'diffusion_models': 'unet'
        };

        const altFolder = alternateFolders[folder];
        if (altFolder) {
            const altRepo = this.modelRepositories[altFolder];
            if (altRepo && altRepo[filename]) {
                return altRepo[filename];
            }
        }

        return null;
    },

    async startDownload(modelInfo, button) {
        const knownUrl = this.getKnownUrl(modelInfo.folder, modelInfo.filename);

        let url = knownUrl;
        if (!url) {
            // Prompt for custom URL
            url = prompt(
                `Enter download URL for:\n${modelInfo.filename}\n\n` +
                `Model type: ${modelInfo.folder}\n\n` +
                `You can find models at:\n` +
                `• HuggingFace: https://huggingface.co/models\n` +
                `• CivitAI: https://civitai.com/models`
            );

            if (!url || !url.trim()) return;
            url = url.trim();
        }

        // Update button state
        button.textContent = 'Starting...';
        button.disabled = true;
        button.style.background = '#FF9800';

        try {
            const response = await api.fetchApi("/api/models/download", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    url: url,
                    model_type: modelInfo.folder,
                    filename: modelInfo.filename
                })
            });

            const data = await response.json();

            if (response.ok) {
                console.log(`[MissingModelsDownloader] Started download: ${data.task_id}`);
                this.activeDownloads.set(data.task_id, { button, modelInfo });
                this.monitorDownload(data.task_id, button);
            } else {
                button.textContent = 'Failed';
                button.style.background = '#F44336';
                button.disabled = false;
                alert(`Download failed: ${data.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('[MissingModelsDownloader] Download error:', error);
            button.textContent = 'Error';
            button.style.background = '#F44336';
            button.disabled = false;
        }
    },

    async monitorDownload(taskId, button) {
        const checkStatus = async () => {
            try {
                const response = await api.fetchApi(`/api/models/download/${taskId}`);
                const status = await response.json();

                if (!response.ok) {
                    button.textContent = 'Failed';
                    button.style.background = '#F44336';
                    button.disabled = false;
                    this.activeDownloads.delete(taskId);
                    return;
                }

                switch (status.status) {
                    case 'completed':
                        button.textContent = '✓ Downloaded';
                        button.style.background = '#4CAF50';
                        button.disabled = true;
                        this.activeDownloads.delete(taskId);

                        // Refresh model lists
                        if (app.refreshComboInNodes) {
                            app.refreshComboInNodes();
                        }
                        break;

                    case 'downloading':
                        const progress = Math.round(status.progress || 0);
                        button.textContent = `${progress}%`;
                        button.style.background = '#2196F3';
                        setTimeout(checkStatus, 2000);
                        break;

                    case 'failed':
                        button.textContent = 'Failed';
                        button.style.background = '#F44336';
                        button.disabled = false;
                        this.activeDownloads.delete(taskId);
                        break;

                    default:
                        button.textContent = status.status;
                        setTimeout(checkStatus, 2000);
                }
            } catch (error) {
                console.error('[MissingModelsDownloader] Status check error:', error);
                button.textContent = 'Error';
                button.style.background = '#F44336';
                button.disabled = false;
                this.activeDownloads.delete(taskId);
            }
        };

        checkStatus();
    },

    addDownloadAllButton(dialogElement, modelEntries) {
        // Find dialog footer or create button container
        let buttonContainer = dialogElement.querySelector('.p-dialog-footer, .dialog-footer');

        if (!buttonContainer) {
            buttonContainer = document.createElement('div');
            buttonContainer.style.cssText = `
                padding: 15px;
                border-top: 1px solid #444;
                text-align: center;
                margin-top: 15px;
            `;
            dialogElement.appendChild(buttonContainer);
        }

        const downloadAllBtn = document.createElement('button');
        downloadAllBtn.textContent = `Download All (${modelEntries.length} models)`;
        downloadAllBtn.style.cssText = `
            padding: 8px 16px;
            background: #FF9800;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            margin: 0 5px;
        `;

        downloadAllBtn.onclick = () => {
            modelEntries.forEach(entry => {
                if (entry.button && !entry.button.disabled) {
                    entry.button.click();
                }
            });
            downloadAllBtn.disabled = true;
            downloadAllBtn.textContent = 'Downloads Started';
        };

        buttonContainer.appendChild(downloadAllBtn);
    }
});