import { showToast, openCivitai, copyToClipboard, copyLoraSyntax, sendLoraToWorkflow, openExampleImagesFolder } from '../../utils/uiHelpers.js';
import { state, getCurrentPageState } from '../../state/index.js';
import { showModelModal } from './ModelModal.js';
import { toggleShowcase } from './showcase/ShowcaseView.js';
import { bulkManager } from '../../managers/BulkManager.js';
import { modalManager } from '../../managers/ModalManager.js';
import { NSFW_LEVELS } from '../../utils/constants.js';
import { MODEL_TYPES } from '../../api/apiConfig.js';
import { getModelApiClient } from '../../api/modelApiFactory.js';
import { showDeleteModal } from '../../utils/modalUtils.js';

// Add global event delegation handlers
export function setupModelCardEventDelegation(modelType) {
    const gridElement = document.getElementById('modelGrid');
    if (!gridElement) return;
    
    // Remove any existing event listener to prevent duplication
    gridElement.removeEventListener('click', gridElement._handleModelCardEvent);
    
    // Create event handler with modelType context
    const handleModelCardEvent = (event) => handleModelCardEvent_internal(event, modelType);
    
    // Add the event delegation handler
    gridElement.addEventListener('click', handleModelCardEvent);
    
    // Store reference to the handler for cleanup
    gridElement._handleModelCardEvent = handleModelCardEvent;
}

// Event delegation handler for all model card events
function handleModelCardEvent_internal(event, modelType) {
    // Find the closest card element
    const card = event.target.closest('.model-card');
    if (!card) return;
    
    // Handle specific elements within the card
    if (event.target.closest('.toggle-blur-btn')) {
        event.stopPropagation();
        toggleBlurContent(card);
        return;
    }
    
    if (event.target.closest('.show-content-btn')) {
        event.stopPropagation();
        showBlurredContent(card);
        return;
    }
    
    if (event.target.closest('.fa-star')) {
        event.stopPropagation();
        toggleFavorite(card);
        return;
    }
    
    if (event.target.closest('.fa-globe')) {
        event.stopPropagation();
        if (card.dataset.from_civitai === 'true') {
            openCivitai(card.dataset.filepath);
        }
        return;
    }
    
    if (event.target.closest('.fa-paper-plane')) {
        event.stopPropagation();
        handleSendToWorkflow(card, event.shiftKey, modelType);
        return;
    }
    
    if (event.target.closest('.fa-copy')) {
        event.stopPropagation();
        handleCopyAction(card, modelType);
        return;
    }
    
    if (event.target.closest('.fa-trash')) {
        event.stopPropagation();
        showDeleteModal(card.dataset.filepath);
        return;
    }
    
    if (event.target.closest('.fa-image')) {
        event.stopPropagation();
        getModelApiClient().replaceModelPreview(card.dataset.filepath);
        return;
    }
    
    if (event.target.closest('.fa-folder-open')) {
        event.stopPropagation();
        handleExampleImagesAccess(card, modelType);
        return;
    }
    
    // If no specific element was clicked, handle the card click (show modal or toggle selection)
    handleCardClick(card, modelType);
}

// Helper functions for event handling
function toggleBlurContent(card) {
    const preview = card.querySelector('.card-preview');
    const isBlurred = preview.classList.toggle('blurred');
    const icon = card.querySelector('.toggle-blur-btn i');
    
    // Update the icon based on blur state
    if (isBlurred) {
        icon.className = 'fas fa-eye';
    } else {
        icon.className = 'fas fa-eye-slash';
    }
    
    // Toggle the overlay visibility
    const overlay = card.querySelector('.nsfw-overlay');
    if (overlay) {
        overlay.style.display = isBlurred ? 'flex' : 'none';
    }
}

function showBlurredContent(card) {
    const preview = card.querySelector('.card-preview');
    preview.classList.remove('blurred');
    
    // Update the toggle button icon
    const toggleBtn = card.querySelector('.toggle-blur-btn');
    if (toggleBtn) {
        toggleBtn.querySelector('i').className = 'fas fa-eye-slash';
    }
    
    // Hide the overlay
    const overlay = card.querySelector('.nsfw-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

async function toggleFavorite(card) {
    const starIcon = card.querySelector('.fa-star');
    const isFavorite = starIcon.classList.contains('fas');
    const newFavoriteState = !isFavorite;
    
    try {
        await getModelApiClient().saveModelMetadata(card.dataset.filepath, { 
            favorite: newFavoriteState 
        });

        if (newFavoriteState) {
            showToast('Added to favorites', 'success');
        } else {
            showToast('Removed from favorites', 'success');
        }
    } catch (error) {
        console.error('Failed to update favorite status:', error);
        showToast('Failed to update favorite status', 'error');
    }
}

function handleSendToWorkflow(card, replaceMode, modelType) {
    if (modelType === MODEL_TYPES.LORA) {
        const usageTips = JSON.parse(card.dataset.usage_tips || '{}');
        const strength = usageTips.strength || 1;
        const loraSyntax = `<lora:${card.dataset.file_name}:${strength}>`;
        sendLoraToWorkflow(loraSyntax, replaceMode, 'lora');
    } else {
        // Checkpoint send functionality - to be implemented
        showToast('Send checkpoint to workflow - feature to be implemented', 'info');
    }
}

function handleCopyAction(card, modelType) {
    if (modelType === MODEL_TYPES.LORA) {
        copyLoraSyntax(card);
    } else if (modelType === MODEL_TYPES.CHECKPOINT) {
        // Checkpoint copy functionality - copy checkpoint name
        const checkpointName = card.dataset.file_name;
        copyToClipboard(checkpointName, 'Checkpoint name copied');
    } else if (modelType === MODEL_TYPES.EMBEDDING) {
        const embeddingName = card.dataset.file_name;
        copyToClipboard(embeddingName, 'Embedding name copied');
    }
}

function handleReplacePreview(filePath, modelType) {
    apiClient.replaceModelPreview(filePath);
}

async function handleExampleImagesAccess(card, modelType) {
    const modelHash = card.dataset.sha256;
    
    try {
        const response = await fetch(`/api/has-example-images?model_hash=${modelHash}`);
        const data = await response.json();
        
        if (data.has_images) {
            openExampleImagesFolder(modelHash);
        } else {
            showExampleAccessModal(card, modelType);
        }
    } catch (error) {
        console.error('Error checking for example images:', error);
        showToast('Error checking for example images', 'error');
    }
}

function handleCardClick(card, modelType) {
    const pageState = getCurrentPageState();
    
    if (state.bulkMode) {
        // Toggle selection using the bulk manager
        bulkManager.toggleCardSelection(card);
    } else if (pageState && pageState.duplicatesMode) {
        // In duplicates mode, don't open modal when clicking cards
        return;
    } else {
        // Normal behavior - show modal
        showModelModalFromCard(card, modelType);
    }
}

function showModelModalFromCard(card, modelType) {
    // Get the appropriate preview versions map
    const previewVersionsKey = modelType;
    const previewVersions = state.pages[previewVersionsKey]?.previewVersions || new Map();
    const version = previewVersions.get(card.dataset.filepath);
    const previewUrl = card.dataset.preview_url || '/loras_static/images/no-preview.png';
    const versionedPreviewUrl = version ? `${previewUrl}?t=${version}` : previewUrl;
    
    // Create model metadata object
    const modelMeta = {
        sha256: card.dataset.sha256,
        file_path: card.dataset.filepath,
        model_name: card.dataset.name,
        file_name: card.dataset.file_name,
        folder: card.dataset.folder,
        modified: card.dataset.modified,
        file_size: parseInt(card.dataset.file_size || '0'),
        from_civitai: card.dataset.from_civitai === 'true',
        base_model: card.dataset.base_model,
        notes: card.dataset.notes || '',
        preview_url: versionedPreviewUrl,
        favorite: card.dataset.favorite === 'true',
        // Parse civitai metadata from the card's dataset
        civitai: JSON.parse(card.dataset.meta || '{}'),
        tags: JSON.parse(card.dataset.tags || '[]'),
        modelDescription: card.dataset.modelDescription || '',
        // LoRA specific fields
        ...(modelType === MODEL_TYPES.LORA && {
            usage_tips: card.dataset.usage_tips,
        })
    };
    
    showModelModal(modelMeta, modelType);
}

// Function to show the example access modal (generalized for lora and checkpoint)
function showExampleAccessModal(card, modelType) {
    const modal = document.getElementById('exampleAccessModal');
    if (!modal) return;

    // Get download button and determine if download should be enabled
    const downloadBtn = modal.querySelector('#downloadExamplesBtn');
    let hasRemoteExamples = false;

    try {
        const metaData = JSON.parse(card.dataset.meta || '{}');
        hasRemoteExamples = metaData.images &&
                            Array.isArray(metaData.images) &&
                            metaData.images.length > 0 &&
                            metaData.images[0].url;
    } catch (e) {
        console.error('Error parsing meta data:', e);
    }

    // Enable or disable download button
    if (downloadBtn) {
        if (hasRemoteExamples) {
            downloadBtn.classList.remove('disabled');
            downloadBtn.removeAttribute('title');
            downloadBtn.onclick = () => {
                modalManager.closeModal('exampleAccessModal');
                // Open settings modal and scroll to example images section
                const settingsModal = document.getElementById('settingsModal');
                if (settingsModal) {
                    modalManager.showModal('settingsModal');
                    setTimeout(() => {
                        const exampleSection = settingsModal.querySelector('.settings-section:nth-child(7)');
                        if (exampleSection) {
                            exampleSection.scrollIntoView({ behavior: 'smooth' });
                        }
                    }, 300);
                }
            };
        } else {
            downloadBtn.classList.add('disabled');
            downloadBtn.setAttribute('title', 'No remote example images available for this model on Civitai');
            downloadBtn.onclick = null;
        }
    }

    // Set up import button
    const importBtn = modal.querySelector('#importExamplesBtn');
    if (importBtn) {
        importBtn.onclick = () => {
            modalManager.closeModal('exampleAccessModal');

            // Get the model data from card dataset (works for both lora and checkpoint)
            const modelMeta = {
                sha256: card.dataset.sha256,
                file_path: card.dataset.filepath,
                model_name: card.dataset.name,
                file_name: card.dataset.file_name,
                folder: card.dataset.folder,
                modified: card.dataset.modified,
                file_size: card.dataset.file_size,
                from_civitai: card.dataset.from_civitai === 'true',
                base_model: card.dataset.base_model,
                notes: card.dataset.notes,
                favorite: card.dataset.favorite === 'true',
                civitai: JSON.parse(card.dataset.meta || '{}'),
                tags: JSON.parse(card.dataset.tags || '[]'),
                modelDescription: card.dataset.modelDescription || ''
            };

            // Add usage_tips if present (for lora)
            if (card.dataset.usage_tips) {
                modelMeta.usage_tips = card.dataset.usage_tips;
            }

            // Show the model modal
            showModelModal(modelMeta, modelType);

            // Scroll to import area after modal is visible
            setTimeout(() => {
                const importArea = document.querySelector('.example-import-area');
                if (importArea) {
                    const showcaseTab = document.getElementById('showcase-tab');
                    if (showcaseTab) {
                        // First make sure showcase tab is visible
                        const tabBtn = document.querySelector('.tab-btn[data-tab="showcase"]');
                        if (tabBtn && !tabBtn.classList.contains('active')) {
                            tabBtn.click();
                        }

                        // Then toggle showcase if collapsed
                        const carousel = showcaseTab.querySelector('.carousel');
                        if (carousel && carousel.classList.contains('collapsed')) {
                            const scrollIndicator = showcaseTab.querySelector('.scroll-indicator');
                            if (scrollIndicator) {
                                toggleShowcase(scrollIndicator);
                            }
                        }

                        // Finally scroll to the import area
                        importArea.scrollIntoView({ behavior: 'smooth' });
                    }
                }
            }, 500);
        };
    }

    // Show the modal
    modalManager.showModal('exampleAccessModal');
}

export function createModelCard(model, modelType) {
    const card = document.createElement('div');
    card.className = 'model-card';  // Reuse the same class for styling
    card.dataset.sha256 = model.sha256;
    card.dataset.filepath = model.file_path;
    card.dataset.name = model.model_name;
    card.dataset.file_name = model.file_name;
    card.dataset.folder = model.folder;
    card.dataset.modified = model.modified;
    card.dataset.file_size = model.file_size;
    card.dataset.from_civitai = model.from_civitai;
    card.dataset.notes = model.notes || '';
    card.dataset.base_model = model.base_model || 'Unknown';
    card.dataset.favorite = model.favorite ? 'true' : 'false';

    // LoRA specific data
    if (modelType === MODEL_TYPES.LORA) {
        card.dataset.usage_tips = model.usage_tips;
    }

    // checkpoint specific data
    if (modelType === MODEL_TYPES.CHECKPOINT) {
        card.dataset.model_type = model.model_type; // checkpoint or diffusion_model
    }

    // Store metadata if available
    if (model.civitai) {
        card.dataset.meta = JSON.stringify(model.civitai || {});
    }
    
    // Store tags if available
    if (model.tags && Array.isArray(model.tags)) {
        card.dataset.tags = JSON.stringify(model.tags);
    }

    if (model.modelDescription) {
        card.dataset.modelDescription = model.modelDescription;
    }

    // Store NSFW level if available
    const nsfwLevel = model.preview_nsfw_level !== undefined ? model.preview_nsfw_level : 0;
    card.dataset.nsfwLevel = nsfwLevel;
    
    // Determine if the preview should be blurred based on NSFW level and user settings
    const shouldBlur = state.settings.blurMatureContent && nsfwLevel > NSFW_LEVELS.PG13;
    if (shouldBlur) {
        card.classList.add('nsfw-content');
    }

    // Apply selection state if in bulk mode and this card is in the selected set (LoRA only)
    if (modelType === MODEL_TYPES.LORA && state.bulkMode && state.selectedLoras.has(model.file_path)) {
        card.classList.add('selected');
    }

    // Get the appropriate preview versions map
    const previewVersionsKey = modelType;
    const previewVersions = state.pages[previewVersionsKey]?.previewVersions || new Map();
    const version = previewVersions.get(model.file_path);
    const previewUrl = model.preview_url || '/loras_static/images/no-preview.png';
    const versionedPreviewUrl = version ? `${previewUrl}?t=${version}` : previewUrl;

    // Determine NSFW warning text based on level
    let nsfwText = "Mature Content";
    if (nsfwLevel >= NSFW_LEVELS.XXX) {
        nsfwText = "XXX-rated Content";
    } else if (nsfwLevel >= NSFW_LEVELS.X) {
        nsfwText = "X-rated Content";
    } else if (nsfwLevel >= NSFW_LEVELS.R) {
        nsfwText = "R-rated Content";
    }

    // Check if autoplayOnHover is enabled for video previews
    const autoplayOnHover = state.global?.settings?.autoplayOnHover || false;
    const isVideo = previewUrl.endsWith('.mp4');
    const videoAttrs = autoplayOnHover ? 'controls muted loop' : 'controls autoplay muted loop';

    // Get favorite status from model data
    const isFavorite = model.favorite === true;

    // Generate action icons based on model type
    const actionIcons = `
        <i class="${isFavorite ? 'fas fa-star favorite-active' : 'far fa-star'}" 
           title="${isFavorite ? 'Remove from favorites' : 'Add to favorites'}">
        </i>
        <i class="fas fa-globe" 
           title="${model.from_civitai ? 'View on Civitai' : 'Not available from Civitai'}"
           ${!model.from_civitai ? 'style="opacity: 0.5; cursor: not-allowed"' : ''}>
        </i>
        <i class="fas fa-paper-plane" 
           title="Send to ComfyUI (Click: Append, Shift+Click: Replace)">
        </i>
        <i class="fas fa-copy" 
           title="Copy LoRA Syntax">
        </i>`;

    card.innerHTML = `
        <div class="card-preview ${shouldBlur ? 'blurred' : ''}">
            ${isVideo ? 
                `<video ${videoAttrs} style="pointer-events: none;">
                    <source src="${versionedPreviewUrl}" type="video/mp4">
                </video>` :
                `<img src="${versionedPreviewUrl}" alt="${model.model_name}">`
            }
            <div class="card-header">
                ${shouldBlur ? 
                  `<button class="toggle-blur-btn" title="Toggle blur">
                      <i class="fas fa-eye"></i>
                  </button>` : ''}
                <span class="base-model-label ${shouldBlur ? 'with-toggle' : ''}" title="${model.base_model}">
                    ${model.base_model}
                </span>
                <div class="card-actions">
                    ${actionIcons}
                </div>
            </div>
            ${shouldBlur ? `
                <div class="nsfw-overlay">
                    <div class="nsfw-warning">
                        <p>${nsfwText}</p>
                        <button class="show-content-btn">Show</button>
                    </div>
                </div>
            ` : ''}
            <div class="card-footer">
                <div class="model-info">
                    <span class="model-name">${model.model_name}</span>
                    ${model.civitai?.name ? `<span class="version-name">${model.civitai.name}</span>` : ''}
                </div>
                <div class="card-actions">
                    <i class="fas fa-folder-open" 
                       title="Open Example Images Folder">
                    </i>
                </div>
            </div>
        </div>
    `;
    
    // Add video auto-play on hover functionality if needed
    const videoElement = card.querySelector('video');
    if (videoElement && autoplayOnHover) {
        const cardPreview = card.querySelector('.card-preview');
        
        // Remove autoplay attribute and pause initially
        videoElement.removeAttribute('autoplay');
        videoElement.pause();
        
        // Add mouse events to trigger play/pause using event attributes
        cardPreview.setAttribute('onmouseenter', 'this.querySelector("video")?.play()');
        cardPreview.setAttribute('onmouseleave', 'const v=this.querySelector("video"); if(v){v.pause();v.currentTime=0;}');
    }

    return card;
}

// Add a method to update card appearance based on bulk mode (LoRA only)
export function updateCardsForBulkMode(isBulkMode) {
    // Update the state
    state.bulkMode = isBulkMode;
    
    document.body.classList.toggle('bulk-mode', isBulkMode);
    
    // Get all lora cards - this can now be from the DOM or through the virtual scroller
    const loraCards = document.querySelectorAll('.model-card');
    
    loraCards.forEach(card => {
        // Get all action containers for this card
        const actions = card.querySelectorAll('.card-actions');
        
        // Handle display property based on mode
        if (isBulkMode) {
            // Hide actions when entering bulk mode
            actions.forEach(actionGroup => {
                actionGroup.style.display = 'none';
            });
        } else {
            // Ensure actions are visible when exiting bulk mode
            actions.forEach(actionGroup => {
                // We need to reset to default display style which is flex
                actionGroup.style.display = 'flex';
            });
        }
    });
    
    // If using virtual scroller, we need to rerender after toggling bulk mode
    if (state.virtualScroller && typeof state.virtualScroller.scheduleRender === 'function') {
        state.virtualScroller.scheduleRender();
    }
    
    // Apply selection state to cards if entering bulk mode
    if (isBulkMode) {
        bulkManager.applySelectionState();
    }
}