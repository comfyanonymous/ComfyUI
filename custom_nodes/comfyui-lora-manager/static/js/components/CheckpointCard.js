import { showToast, copyToClipboard, openExampleImagesFolder, openCivitai } from '../utils/uiHelpers.js';
import { state } from '../state/index.js';
import { showCheckpointModal } from './checkpointModal/index.js';
import { NSFW_LEVELS } from '../utils/constants.js';
import { replaceCheckpointPreview as apiReplaceCheckpointPreview, saveModelMetadata } from '../api/checkpointApi.js';
import { showDeleteModal } from '../utils/modalUtils.js';

// Add a global event delegation handler
export function setupCheckpointCardEventDelegation() {
    const gridElement = document.getElementById('checkpointGrid');
    if (!gridElement) return;
    
    // Remove any existing event listener to prevent duplication
    gridElement.removeEventListener('click', handleCheckpointCardEvent);
    
    // Add the event delegation handler
    gridElement.addEventListener('click', handleCheckpointCardEvent);
}

// Event delegation handler for all checkpoint card events
function handleCheckpointCardEvent(event) {
    // Find the closest card element
    const card = event.target.closest('.lora-card');
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
    
    if (event.target.closest('.fa-copy')) {
        event.stopPropagation();
        copyCheckpointName(card);
        return;
    }
    
    if (event.target.closest('.fa-trash')) {
        event.stopPropagation();
        showDeleteModal(card.dataset.filepath);
        return;
    }
    
    if (event.target.closest('.fa-image')) {
        event.stopPropagation();
        replaceCheckpointPreview(card.dataset.filepath);
        return;
    }
    
    if (event.target.closest('.fa-folder-open')) {
        event.stopPropagation();
        openExampleImagesFolder(card.dataset.sha256);
        return;
    }
    
    // If no specific element was clicked, handle the card click (show modal)
    showCheckpointModalFromCard(card);
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
        // Save the new favorite state to the server
        await saveModelMetadata(card.dataset.filepath, { 
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

async function copyCheckpointName(card) {
    const checkpointName = card.dataset.file_name;
    
    try {
        await copyToClipboard(checkpointName, 'Checkpoint name copied');
    } catch (err) {
        console.error('Copy failed:', err);
        showToast('Copy failed', 'error');
    }
}

function showCheckpointModalFromCard(card) {
    // Get the page-specific previewVersions map
    const previewVersions = state.pages.checkpoints.previewVersions || new Map();
    const version = previewVersions.get(card.dataset.filepath);
    const previewUrl = card.dataset.preview_url || '/loras_static/images/no-preview.png';
    const versionedPreviewUrl = version ? `${previewUrl}?t=${version}` : previewUrl;
    
    // Show checkpoint details modal
    const checkpointMeta = {
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
        // Parse civitai metadata from the card's dataset
        civitai: (() => {
            try {
                return JSON.parse(card.dataset.meta || '{}');
            } catch (e) {
                console.error('Failed to parse civitai metadata:', e);
                return {}; // Return empty object on error
            }
        })(),
        tags: (() => {
            try {
                return JSON.parse(card.dataset.tags || '[]');
            } catch (e) {
                console.error('Failed to parse tags:', e);
                return []; // Return empty array on error
            }
        })(),
        modelDescription: card.dataset.modelDescription || ''
    };
    showCheckpointModal(checkpointMeta);
}

function replaceCheckpointPreview(filePath) {
    if (window.replaceCheckpointPreview) {
        window.replaceCheckpointPreview(filePath);
    } else {
        apiReplaceCheckpointPreview(filePath);
    }
}

export function createCheckpointCard(checkpoint) {
    const card = document.createElement('div');
    card.className = 'lora-card';  // Reuse the same class for styling
    card.dataset.sha256 = checkpoint.sha256;
    card.dataset.filepath = checkpoint.file_path;
    card.dataset.name = checkpoint.model_name;
    card.dataset.file_name = checkpoint.file_name;
    card.dataset.folder = checkpoint.folder;
    card.dataset.modified = checkpoint.modified;
    card.dataset.file_size = checkpoint.file_size;
    card.dataset.from_civitai = checkpoint.from_civitai;
    card.dataset.notes = checkpoint.notes || '';
    card.dataset.base_model = checkpoint.base_model || 'Unknown';
    card.dataset.favorite = checkpoint.favorite ? 'true' : 'false';

    // Store metadata if available
    if (checkpoint.civitai) {
        card.dataset.meta = JSON.stringify(checkpoint.civitai || {});
    }
    
    // Store tags if available
    if (checkpoint.tags && Array.isArray(checkpoint.tags)) {
        card.dataset.tags = JSON.stringify(checkpoint.tags);
    }

    if (checkpoint.modelDescription) {
        card.dataset.modelDescription = checkpoint.modelDescription;
    }

    // Store NSFW level if available
    const nsfwLevel = checkpoint.preview_nsfw_level !== undefined ? checkpoint.preview_nsfw_level : 0;
    card.dataset.nsfwLevel = nsfwLevel;
    
    // Determine if the preview should be blurred based on NSFW level and user settings
    const shouldBlur = state.settings.blurMatureContent && nsfwLevel > NSFW_LEVELS.PG13;
    if (shouldBlur) {
        card.classList.add('nsfw-content');
    }

    // Determine preview URL
    const previewUrl = checkpoint.preview_url || '/loras_static/images/no-preview.png';
    
    // Get the page-specific previewVersions map
    const previewVersions = state.pages.checkpoints.previewVersions || new Map();
    const version = previewVersions.get(checkpoint.file_path);
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

    // Get favorite status from checkpoint data
    const isFavorite = checkpoint.favorite === true;

    card.innerHTML = `
        <div class="card-preview ${shouldBlur ? 'blurred' : ''}">
            ${isVideo ? 
                `<video ${videoAttrs}>
                    <source src="${versionedPreviewUrl}" type="video/mp4">
                </video>` :
                `<img src="${versionedPreviewUrl}" alt="${checkpoint.model_name}">`
            }
            <div class="card-header">
                ${shouldBlur ? 
                  `<button class="toggle-blur-btn" title="Toggle blur">
                      <i class="fas fa-eye"></i>
                  </button>` : ''}
                <span class="base-model-label ${shouldBlur ? 'with-toggle' : ''}" title="${checkpoint.base_model}">
                    ${checkpoint.base_model}
                </span>
                <div class="card-actions">
                    <i class="${isFavorite ? 'fas fa-star favorite-active' : 'far fa-star'}" 
                       title="${isFavorite ? 'Remove from favorites' : 'Add to favorites'}">
                    </i>
                    <i class="fas fa-globe" 
                       title="${checkpoint.from_civitai ? 'View on Civitai' : 'Not available from Civitai'}"
                       ${!checkpoint.from_civitai ? 'style="opacity: 0.5; cursor: not-allowed"' : ''}>
                    </i>
                    <i class="fas fa-copy" 
                       title="Copy Checkpoint Name">
                    </i>
                    <i class="fas fa-trash" 
                       title="Delete Model">
                    </i>
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
                    <span class="model-name">${checkpoint.model_name}</span>
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