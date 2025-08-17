/**
 * CheckpointModal - Main entry point
 * 
 * Modularized checkpoint modal component that handles checkpoint model details display
 */
import { showToast } from '../../utils/uiHelpers.js';
import { modalManager } from '../../managers/ModalManager.js';
import { 
    toggleShowcase,
    setupShowcaseScroll, 
    scrollToTop,
    loadExampleImages
} from '../shared/showcase/ShowcaseView.js';
import { setupTabSwitching, loadModelDescription } from './ModelDescription.js';
import { 
    setupModelNameEditing, 
    setupBaseModelEditing, 
    setupFileNameEditing
} from './ModelMetadata.js';
import { setupTagEditMode } from './ModelTags.js'; // Add import for tag editing
import { saveModelMetadata } from '../../api/checkpointApi.js';
import { renderCompactTags, setupTagTooltip, formatFileSize } from './utils.js';

/**
 * Display the checkpoint modal with the given checkpoint data
 * @param {Object} checkpoint - Checkpoint data object
 */
export function showCheckpointModal(checkpoint) {
    const content = `
        <div class="modal-content">
            <button class="close" onclick="modalManager.closeModal('checkpointModal')">&times;</button>
            <header class="modal-header">
                <div class="model-name-header">
                    <h2 class="model-name-content">${checkpoint.model_name || 'Checkpoint Details'}</h2>
                    <button class="edit-model-name-btn" title="Edit model name">
                        <i class="fas fa-pencil-alt"></i>
                    </button>
                </div>

                ${checkpoint.civitai?.creator ? `
                <div class="creator-info">
                    ${checkpoint.civitai.creator.image ? 
                        `<div class="creator-avatar">
                            <img src="${checkpoint.civitai.creator.image}" alt="${checkpoint.civitai.creator.username}" onerror="this.onerror=null; this.src='static/icons/user-placeholder.png';">
                        </div>` : 
                        `<div class="creator-avatar creator-placeholder">
                            <i class="fas fa-user"></i>
                        </div>`
                    }
                    <span class="creator-username">${checkpoint.civitai.creator.username}</span>
                </div>` : ''}

                ${renderCompactTags(checkpoint.tags || [], checkpoint.file_path)}
            </header>

            <div class="modal-body">
                <div class="info-section">
                    <div class="info-grid">
                        <div class="info-item">
                            <label>Version</label>
                            <span>${checkpoint.civitai?.name || 'N/A'}</span>
                        </div>
                        <div class="info-item">
                            <label>File Name</label>
                            <div class="file-name-wrapper">
                                <span id="file-name" class="file-name-content">${checkpoint.file_name || 'N/A'}</span>
                                <button class="edit-file-name-btn" title="Edit file name">
                                    <i class="fas fa-pencil-alt"></i>
                                </button>
                            </div>
                        </div>
                        <div class="info-item location-size">
                            <div class="location-wrapper">
                                <label>Location</label>
                                <span class="file-path">${checkpoint.file_path.replace(/[^/]+$/, '')}</span>
                            </div>
                        </div>
                        <div class="info-item base-size">
                            <div class="base-wrapper">
                                <label>Base Model</label>
                                <div class="base-model-display">
                                    <span class="base-model-content">${checkpoint.base_model || 'Unknown'}</span>
                                    <button class="edit-base-model-btn" title="Edit base model">
                                        <i class="fas fa-pencil-alt"></i>
                                    </button>
                                </div>
                            </div>
                            <div class="size-wrapper">
                                <label>Size</label>
                                <span>${formatFileSize(checkpoint.file_size)}</span>
                            </div>
                        </div>
                        <div class="info-item notes">
                            <label>Additional Notes</label>
                            <div class="editable-field">
                                <div class="notes-content" contenteditable="true" spellcheck="false">${checkpoint.notes || 'Add your notes here...'}</div>
                                <button class="save-btn" onclick="saveCheckpointNotes('${checkpoint.file_path}')">
                                    <i class="fas fa-save"></i>
                                </button>
                            </div>
                        </div>
                        <div class="info-item full-width">
                            <label>About this version</label>
                            <div class="description-text">${checkpoint.civitai?.description || 'N/A'}</div>
                        </div>
                    </div>
                </div>

                <div class="showcase-section" data-model-hash="${checkpoint.sha256 || ''}" data-filepath="${checkpoint.file_path}">
                    <div class="showcase-tabs">
                        <button class="tab-btn active" data-tab="showcase">Examples</button>
                        <button class="tab-btn" data-tab="description">Model Description</button>
                    </div>
                    
                    <div class="tab-content">
                        <div id="showcase-tab" class="tab-pane active">
                            <div class="recipes-loading">
                                <i class="fas fa-spinner fa-spin"></i> Loading recipes...
                            </div>
                        </div>
                        
                        <div id="description-tab" class="tab-pane">
                            <div class="model-description-container">
                                <div class="model-description-loading">
                                    <i class="fas fa-spinner fa-spin"></i> Loading model description...
                                </div>
                                <div class="model-description-content">
                                    ${checkpoint.modelDescription || ''}
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <button class="back-to-top" onclick="scrollToTopCheckpoint(this)">
                        <i class="fas fa-arrow-up"></i>
                    </button>
                </div>
            </div>
        </div>
    `;
    
    modalManager.showModal('checkpointModal', content);
    setupEditableFields(checkpoint.file_path);
    setupShowcaseScroll('checkpointModal');
    setupTabSwitching();
    setupTagTooltip();
    setupTagEditMode(); // Initialize tag editing functionality
    setupModelNameEditing(checkpoint.file_path);
    setupBaseModelEditing(checkpoint.file_path);
    setupFileNameEditing(checkpoint.file_path);
    
    // If we have a model ID but no description, fetch it
    if (checkpoint.civitai?.modelId && !checkpoint.modelDescription) {
        loadModelDescription(checkpoint.civitai.modelId, checkpoint.file_path);
    }
    
    // Load example images asynchronously - merge regular and custom images
    const regularImages = checkpoint.civitai?.images || [];
    const customImages = checkpoint.civitai?.customImages || [];
    // Combine images - regular images first, then custom images
    const allImages = [...regularImages, ...customImages];
    loadExampleImages(allImages, checkpoint.sha256);
}

/**
 * Set up editable fields in the checkpoint modal
* @param {string} filePath - The full file path of the model.
 */
function setupEditableFields(filePath) {
    const editableFields = document.querySelectorAll('.editable-field [contenteditable]');
    
    editableFields.forEach(field => {
        field.addEventListener('focus', function() {
            if (this.textContent === 'Add your notes here...') {
                this.textContent = '';
            }
        });

        field.addEventListener('blur', function() {
            if (this.textContent.trim() === '') {
                if (this.classList.contains('notes-content')) {
                    this.textContent = 'Add your notes here...';
                }
            }
        });
    });

    // Add keydown event listeners for notes
    const notesContent = document.querySelector('.notes-content');
    if (notesContent) {
        notesContent.addEventListener('keydown', async function(e) {
            if (e.key === 'Enter') {
                if (e.shiftKey) {
                    // Allow shift+enter for new line
                    return;
                }
                e.preventDefault();
                await saveNotes(filePath);
            }
        });
    }
}

/**
 * Save checkpoint notes
 * @param {string} filePath - Path to the checkpoint file
 */
async function saveNotes(filePath) {
    const content = document.querySelector('.notes-content').textContent;
    try {
        await saveModelMetadata(filePath, { notes: content });

        showToast('Notes saved successfully', 'success');
    } catch (error) {
        showToast('Failed to save notes', 'error');
    }
}

// Export the checkpoint modal API
const checkpointModal = {
    show: showCheckpointModal,
    toggleShowcase,
    scrollToTop
};

export { checkpointModal };

// Define global functions for use in HTML
window.toggleShowcase = function(element) {
    toggleShowcase(element);
};

window.scrollToTopCheckpoint = function(button) {
    scrollToTop(button);
};

window.saveCheckpointNotes = function(filePath) {
    saveNotes(filePath);
};