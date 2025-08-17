import { showToast, openCivitai, copyToClipboard, sendLoraToWorkflow, openExampleImagesFolder } from '../utils/uiHelpers.js';
import { state, getCurrentPageState } from '../state/index.js';
import { showLoraModal } from './loraModal/index.js';
import { bulkManager } from '../managers/BulkManager.js';
import { NSFW_LEVELS } from '../utils/constants.js';
import { replacePreview, saveModelMetadata } from '../api/loraApi.js'

// Add a global event delegation handler
export function setupLoraCardEventDelegation() {
    const gridElement = document.getElementById('loraGrid');
    if (!gridElement) return;
    
    // Remove any existing event listener to prevent duplication
    gridElement.removeEventListener('click', handleLoraCardEvent);
    
    // Add the event delegation handler
    gridElement.addEventListener('click', handleLoraCardEvent);
}

// Event delegation handler for all lora card events
function handleLoraCardEvent(event) {
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
    
    if (event.target.closest('.fa-paper-plane')) {
        event.stopPropagation();
        sendLoraToComfyUI(card, event.shiftKey);
        return;
    }
    
    if (event.target.closest('.fa-copy')) {
        event.stopPropagation();
        copyLoraSyntax(card);
        return;
    }
    
    if (event.target.closest('.fa-image')) {
        event.stopPropagation();
        replacePreview(card.dataset.filepath);
        return;
    }
    
    if (event.target.closest('.fa-folder-open')) {
        event.stopPropagation();
        handleExampleImagesAccess(card);
        return;
    }
    
    // If no specific element was clicked, handle the card click (show modal or toggle selection)
    const pageState = getCurrentPageState();
    if (state.bulkMode) {
        // Toggle selection using the bulk manager
        bulkManager.toggleCardSelection(card);
    } else if (pageState && pageState.duplicatesMode) {
        // In duplicates mode, don't open modal when clicking cards
        return;
    } else {
        // Normal behavior - show modal
        const loraMeta = {
            sha256: card.dataset.sha256,
            file_path: card.dataset.filepath,
            model_name: card.dataset.name,
            file_name: card.dataset.file_name,
            folder: card.dataset.folder,
            modified: card.dataset.modified,
            file_size: card.dataset.file_size,
            from_civitai: card.dataset.from_civitai === 'true',
            base_model: card.dataset.base_model,
            usage_tips: card.dataset.usage_tips,
            notes: card.dataset.notes,
            favorite: card.dataset.favorite === 'true',
            // Parse civitai metadata from the card's dataset
            civitai: (() => {
                try {
                    // Attempt to parse the JSON string
                    return JSON.parse(card.dataset.meta || '{}');
                } catch (e) {
                    console.error('Failed to parse civitai metadata:', e);
                    return {}; // Return empty object on error
                }
            })(),
            tags: JSON.parse(card.dataset.tags || '[]'),
            modelDescription: card.dataset.modelDescription || ''
        };
        showLoraModal(loraMeta);
    }
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

// Function to send LoRA to ComfyUI workflow
async function sendLoraToComfyUI(card, replaceMode) {
    const usageTips = JSON.parse(card.dataset.usage_tips || '{}');
    const strength = usageTips.strength || 1;
    const loraSyntax = `<lora:${card.dataset.file_name}:${strength}>`;
    
    sendLoraToWorkflow(loraSyntax, replaceMode, 'lora');
}

// Add function to copy lora syntax
function copyLoraSyntax(card) {
    const usageTips = JSON.parse(card.dataset.usage_tips || '{}');
    const strength = usageTips.strength || 1;
    const loraSyntax = `<lora:${card.dataset.file_name}:${strength}>`;
    
    copyToClipboard(loraSyntax, 'LoRA syntax copied to clipboard');
}

// New function to handle example images access
async function handleExampleImagesAccess(card) {
    const modelHash = card.dataset.sha256;
    
    try {
        // Check if example images exist
        const response = await fetch(`/api/has-example-images?model_hash=${modelHash}`);
        const data = await response.json();
        
        if (data.has_images) {
            // If images exist, open the folder directly (existing behavior)
            openExampleImagesFolder(modelHash);
        } else {
            // If no images exist, show the new modal
            showExampleAccessModal(card);
        }
    } catch (error) {
        console.error('Error checking for example images:', error);
        showToast('Error checking for example images', 'error');
    }
}

// Function to show the example access modal
function showExampleAccessModal(card) {
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
            downloadBtn.removeAttribute('title'); // Remove any previous tooltip
            downloadBtn.onclick = () => {
                modalManager.closeModal('exampleAccessModal');
                // Open settings modal and scroll to example images section
                const settingsModal = document.getElementById('settingsModal');
                if (settingsModal) {
                    modalManager.showModal('settingsModal');
                    // Scroll to example images section after modal is visible
                    setTimeout(() => {
                        const exampleSection = settingsModal.querySelector('.settings-section:nth-child(5)'); // Example Images section
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
            
            // Get the lora data from card dataset
            const loraMeta = {
                sha256: card.dataset.sha256,
                file_path: card.dataset.filepath,
                model_name: card.dataset.name,
                file_name: card.dataset.file_name,
                // Other properties needed for showLoraModal
                folder: card.dataset.folder,
                modified: card.dataset.modified,
                file_size: card.dataset.file_size,
                from_civitai: card.dataset.from_civitai === 'true',
                base_model: card.dataset.base_model,
                usage_tips: card.dataset.usage_tips,
                notes: card.dataset.notes,
                favorite: card.dataset.favorite === 'true',
                civitai: (() => {
                    try {
                        return JSON.parse(card.dataset.meta || '{}');
                    } catch (e) {
                        return {};
                    }
                })(),
                tags: JSON.parse(card.dataset.tags || '[]'),
                modelDescription: card.dataset.modelDescription || ''
            };
            
            // Show the lora modal
            showLoraModal(loraMeta);
            
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
                                scrollIndicator.click();
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

export function createLoraCard(lora) {
    const card = document.createElement('div');
    card.className = 'lora-card';
    card.dataset.sha256 = lora.sha256;
    card.dataset.filepath = lora.file_path;
    card.dataset.name = lora.model_name;
    card.dataset.file_name = lora.file_name;
    card.dataset.folder = lora.folder;
    card.dataset.modified = lora.modified;
    card.dataset.file_size = lora.file_size;
    card.dataset.from_civitai = lora.from_civitai;
    card.dataset.base_model = lora.base_model;
    card.dataset.usage_tips = lora.usage_tips;
    card.dataset.notes = lora.notes;
    card.dataset.meta = JSON.stringify(lora.civitai || {});
    card.dataset.favorite = lora.favorite ? 'true' : 'false';
    
    // Store tags and model description
    if (lora.tags && Array.isArray(lora.tags)) {
        card.dataset.tags = JSON.stringify(lora.tags);
    }
    if (lora.modelDescription) {
        card.dataset.modelDescription = lora.modelDescription;
    }

    // Store NSFW level if available
    const nsfwLevel = lora.preview_nsfw_level !== undefined ? lora.preview_nsfw_level : 0;
    card.dataset.nsfwLevel = nsfwLevel;
    
    // Determine if the preview should be blurred based on NSFW level and user settings
    const shouldBlur = state.settings.blurMatureContent && nsfwLevel > NSFW_LEVELS.PG13;
    if (shouldBlur) {
        card.classList.add('nsfw-content');
    }

    // Apply selection state if in bulk mode and this card is in the selected set
    if (state.bulkMode && state.selectedLoras.has(lora.file_path)) {
        card.classList.add('selected');
    }

    // Get the page-specific previewVersions map
    const previewVersions = state.pages.loras.previewVersions || new Map();
    const version = previewVersions.get(lora.file_path);
    const previewUrl = lora.preview_url || '/loras_static/images/no-preview.png';
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
    const autoplayOnHover = state.global.settings.autoplayOnHover || false;
    const isVideo = previewUrl.endsWith('.mp4');
    const videoAttrs = autoplayOnHover ? 'controls muted loop' : 'controls autoplay muted loop';

    // Get favorite status from the lora data
    const isFavorite = lora.favorite === true;

    card.innerHTML = `
        <div class="card-preview ${shouldBlur ? 'blurred' : ''}">
            ${isVideo ? 
                `<video ${videoAttrs}>
                    <source src="${versionedPreviewUrl}" type="video/mp4">
                </video>` :
                `<img src="${versionedPreviewUrl}" alt="${lora.model_name}">`
            }
            <div class="card-header">
                ${shouldBlur ? 
                  `<button class="toggle-blur-btn" title="Toggle blur">
                      <i class="fas fa-eye"></i>
                  </button>` : ''}
                <span class="base-model-label ${shouldBlur ? 'with-toggle' : ''}" title="${lora.base_model}">
                    ${lora.base_model}
                </span>
                <div class="card-actions">
                    <i class="${isFavorite ? 'fas fa-star favorite-active' : 'far fa-star'}" 
                       title="${isFavorite ? 'Remove from favorites' : 'Add to favorites'}">
                    </i>
                    <i class="fas fa-globe" 
                       title="${lora.from_civitai ? 'View on Civitai' : 'Not available from Civitai'}"
                       ${!lora.from_civitai ? 'style="opacity: 0.5; cursor: not-allowed"' : ''}>
                    </i>
                    <i class="fas fa-paper-plane" 
                       title="Send to ComfyUI (Click: Append, Shift+Click: Replace)">
                    </i>
                    <i class="fas fa-copy" 
                       title="Copy LoRA Syntax">
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
                    <span class="model-name">${lora.model_name}</span>
                </div>
                <div class="card-actions">
                    <i class="fas fa-folder-open" 
                       title="Open Example Images Folder">
                    </i>
                </div>
            </div>
        </div>
    `;
    
    // Add a special class for virtual scroll positioning if needed
    if (state.virtualScroller) {
        card.classList.add('virtual-scroll-item');
    }
    
    // Add video auto-play on hover functionality if needed
    const videoElement = card.querySelector('video');
    if (videoElement && autoplayOnHover) {
        const cardPreview = card.querySelector('.card-preview');
        
        // Remove autoplay attribute and pause initially
        videoElement.removeAttribute('autoplay');
        videoElement.pause();
        
        // Add mouse events to trigger play/pause using event attributes
        // This approach reduces the number of event listeners created
        cardPreview.setAttribute('onmouseenter', 'this.querySelector("video")?.play()');
        cardPreview.setAttribute('onmouseleave', 'const v=this.querySelector("video"); if(v){v.pause();v.currentTime=0;}');
    }

    return card;
}

// Add a method to update card appearance based on bulk mode
export function updateCardsForBulkMode(isBulkMode) {
    // Update the state
    state.bulkMode = isBulkMode;
    
    document.body.classList.toggle('bulk-mode', isBulkMode);
    
    // Get all lora cards - this can now be from the DOM or through the virtual scroller
    const loraCards = document.querySelectorAll('.lora-card');
    
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