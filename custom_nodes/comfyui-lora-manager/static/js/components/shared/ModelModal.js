import { showToast, openCivitai } from '../../utils/uiHelpers.js';
import { modalManager } from '../../managers/ModalManager.js';
import { 
    toggleShowcase,
    setupShowcaseScroll, 
    scrollToTop,
    loadExampleImages
} from './showcase/ShowcaseView.js';
import { setupTabSwitching, setupModelDescriptionEditing } from './ModelDescription.js';
import { 
    setupModelNameEditing, 
    setupBaseModelEditing, 
    setupFileNameEditing
} from './ModelMetadata.js';
import { setupTagEditMode } from './ModelTags.js';
import { getModelApiClient } from '../../api/modelApiFactory.js';
import { renderCompactTags, setupTagTooltip, formatFileSize } from './utils.js';
import { renderTriggerWords, setupTriggerWordsEditMode } from './TriggerWords.js';
import { parsePresets, renderPresetTags } from './PresetTags.js';
import { loadRecipesForLora } from './RecipeTab.js';

/**
 * Display the model modal with the given model data
 * @param {Object} model - Model data object
 * @param {string} modelType - Type of model ('lora' or 'checkpoint')
 */
export function showModelModal(model, modelType) {
    const modalId = 'modelModal';
    const modalTitle = model.model_name;
    
    // Prepare LoRA specific data
    const escapedWords = (modelType === 'loras' || modelType === 'embeddings') && model.civitai?.trainedWords?.length ? 
        model.civitai.trainedWords.map(word => word.replace(/'/g, '\\\'')) : [];
    
    // Generate model type specific content
    let typeSpecificContent;
    if (modelType === 'loras') {
        typeSpecificContent = renderLoraSpecificContent(model, escapedWords);
    } else if (modelType === 'embeddings') {
        typeSpecificContent = renderEmbeddingSpecificContent(model, escapedWords);
    } else {
        typeSpecificContent = '';
    }
    
    // Generate tabs based on model type
    const tabsContent = modelType === 'loras' ? 
        `<button class="tab-btn active" data-tab="showcase">Examples</button>
         <button class="tab-btn" data-tab="description">Model Description</button>
         <button class="tab-btn" data-tab="recipes">Recipes</button>` :
        `<button class="tab-btn active" data-tab="showcase">Examples</button>
         <button class="tab-btn" data-tab="description">Model Description</button>`;
    
    const tabPanesContent = modelType === 'loras' ? 
        `<div id="showcase-tab" class="tab-pane active">
            <div class="example-images-loading">
                <i class="fas fa-spinner fa-spin"></i> Loading example images...
            </div>
        </div>
        
        <div id="description-tab" class="tab-pane">
            <div class="model-description-container">
                <div class="model-description-loading">
                    <i class="fas fa-spinner fa-spin"></i> Loading model description...
                </div>
                <div class="model-description-content">
                    ${model.modelDescription || ''}
                </div>
            </div>
        </div>
        
        <div id="recipes-tab" class="tab-pane">
            <div class="recipes-loading">
                <i class="fas fa-spinner fa-spin"></i> Loading recipes...
            </div>
        </div>` :
        `<div id="showcase-tab" class="tab-pane active">
            <div class="recipes-loading">
                <i class="fas fa-spinner fa-spin"></i> Loading examples...
            </div>
        </div>
        
        <div id="description-tab" class="tab-pane">
            <div class="model-description-container">
                <div class="model-description-loading">
                    <i class="fas fa-spinner fa-spin"></i> Loading model description...
                </div>
                <div class="model-description-content">
                    ${model.modelDescription || ''}
                </div>
            </div>
        </div>`;

    const content = `
        <div class="modal-content">
            <button class="close" onclick="modalManager.closeModal('${modalId}')">&times;</button>
            <header class="modal-header">
                <div class="model-name-header">
                    <h2 class="model-name-content">${modalTitle}</h2>
                    <button class="edit-model-name-btn" title="Edit model name">
                        <i class="fas fa-pencil-alt"></i>
                    </button>
                </div>

                <div class="creator-actions">
                    ${model.from_civitai ? `
                    <div class="civitai-view" title="View on Civitai" data-action="view-civitai" data-filepath="${model.file_path}">
                        <i class="fas fa-globe"></i> View on Civitai
                    </div>` : ''}

                    ${model.civitai?.creator ? `
                    <div class="creator-info" data-username="${model.civitai.creator.username}" data-action="view-creator" title="View Creator Profile">
                        ${model.civitai.creator.image ? 
                            `<div class="creator-avatar">
                                <img src="${model.civitai.creator.image}" alt="${model.civitai.creator.username}" onerror="this.onerror=null; this.src='static/icons/user-placeholder.png';">
                            </div>` : 
                            `<div class="creator-avatar creator-placeholder">
                                <i class="fas fa-user"></i>
                            </div>`
                        }
                        <span class="creator-username">${model.civitai.creator.username}</span>
                    </div>` : ''}
                </div>

                ${renderCompactTags(model.tags || [], model.file_path)}
            </header>

            <div class="modal-body">
                <div class="info-section">
                    <div class="info-grid">
                        <div class="info-item">
                            <label>Version</label>
                            <span>${model.civitai?.name || 'N/A'}</span>
                        </div>
                        <div class="info-item">
                            <label>File Name</label>
                            <div class="file-name-wrapper">
                                <span id="file-name" class="file-name-content">${model.file_name || 'N/A'}</span>
                                <button class="edit-file-name-btn" title="Edit file name">
                                    <i class="fas fa-pencil-alt"></i>
                                </button>
                            </div>
                        </div>
                        <div class="info-item location-size">
                            <div class="location-wrapper">
                                <label>Location</label>
                                <span class="file-path">${model.file_path.replace(/[^/]+$/, '') || 'N/A'}</span>
                            </div>
                        </div>
                        <div class="info-item base-size">
                            <div class="base-wrapper">
                                <label>Base Model</label>
                                <div class="base-model-display">
                                    <span class="base-model-content">${model.base_model || 'Unknown'}</span>
                                    <button class="edit-base-model-btn" title="Edit base model">
                                        <i class="fas fa-pencil-alt"></i>
                                    </button>
                                </div>
                            </div>
                            <div class="size-wrapper">
                                <label>Size</label>
                                <span>${formatFileSize(model.file_size)}</span>
                            </div>
                        </div>
                        ${typeSpecificContent}
                        <div class="info-item notes">
                            <label>Additional Notes <i class="fas fa-info-circle notes-hint" title="Press Enter to save, Shift+Enter for new line"></i></label>
                            <div class="editable-field">
                                <div class="notes-content" contenteditable="true" spellcheck="false">${model.notes || 'Add your notes here...'}</div>
                            </div>
                        </div>
                        <div class="info-item full-width">
                            <label>About this version</label>
                            <div class="description-text">${model.civitai?.description || 'N/A'}</div>
                        </div>
                    </div>
                </div>

                <div class="showcase-section" data-model-hash="${model.sha256 || ''}" data-filepath="${model.file_path}">
                    <div class="showcase-tabs">
                        ${tabsContent}
                    </div>
                    
                    <div class="tab-content">
                        ${tabPanesContent}
                    </div>
                    
                    <button class="back-to-top" data-action="scroll-to-top">
                        <i class="fas fa-arrow-up"></i>
                    </button>
                </div>
            </div>
        </div>
    `;
    
    const onCloseCallback = function() {
        // Clean up all handlers when modal closes for LoRA
        const modalElement = document.getElementById(modalId);
        if (modalElement && modalElement._clickHandler) {
            modalElement.removeEventListener('click', modalElement._clickHandler);
            delete modalElement._clickHandler;
        }
    };
    
    modalManager.showModal(modalId, content, null, onCloseCallback);
    setupEditableFields(model.file_path, modelType);
    setupShowcaseScroll(modalId);
    setupTabSwitching();
    setupTagTooltip();
    setupTagEditMode();
    setupModelNameEditing(model.file_path);
    setupBaseModelEditing(model.file_path);
    setupFileNameEditing(model.file_path);
    setupModelDescriptionEditing(model.file_path, model.modelDescription || '');
    setupEventHandlers(model.file_path);
    
    // LoRA specific setup
    if (modelType === 'loras' || modelType === 'embeddings') {
        setupTriggerWordsEditMode();
        
        if (modelType == 'loras') {
            // Load recipes for this LoRA
            loadRecipesForLora(model.model_name, model.sha256);
        }
    }
    
    // Load example images asynchronously - merge regular and custom images
    const regularImages = model.civitai?.images || [];
    const customImages = model.civitai?.customImages || [];
    // Combine images - regular images first, then custom images
    const allImages = [...regularImages, ...customImages];
    loadExampleImages(allImages, model.sha256);
}

function renderLoraSpecificContent(lora, escapedWords) {
    return `
        <div class="info-item usage-tips">
            <label>Usage Tips</label>
            <div class="editable-field">
                <div class="preset-controls">
                    <select id="preset-selector">
                        <option value="">Add preset parameter...</option>
                        <option value="strength_min">Strength Min</option>
                        <option value="strength_max">Strength Max</option>
                        <option value="strength">Strength</option>
                        <option value="clip_skip">Clip Skip</option>
                    </select>
                    <input type="number" id="preset-value" step="0.01" placeholder="Value" style="display:none;">
                    <button class="add-preset-btn">Add</button>
                </div>
                <div class="preset-tags">
                    ${renderPresetTags(parsePresets(lora.usage_tips))}
                </div>
            </div>
        </div>
        ${renderTriggerWords(escapedWords, lora.file_path)}
    `;
}

function renderEmbeddingSpecificContent(embedding, escapedWords) {
    return `${renderTriggerWords(escapedWords, embedding.file_path)}`;
}

/**
 * Sets up event handlers using event delegation for LoRA modal
 * @param {string} filePath - Path to the model file
 */
function setupEventHandlers(filePath) {
    const modalElement = document.getElementById('modelModal');
    
    // Remove existing event listeners first
    modalElement.removeEventListener('click', handleModalClick);
    
    // Create and store the handler function
    function handleModalClick(event) {
        const target = event.target.closest('[data-action]');
        if (!target) return;
        
        const action = target.dataset.action;
        
        switch (action) {
            case 'close-modal':
                modalManager.closeModal('modelModal');
                break;
            case 'scroll-to-top':
                scrollToTop(target);
                break;
            case 'view-civitai':
                openCivitai(target.dataset.filepath);
                break;
            case 'view-creator':
                const username = target.dataset.username;
                if (username) {
                    window.open(`https://civitai.com/user/${username}`, '_blank');
                }
                break;
        }
    }
    
    // Add the event listener with the named function
    modalElement.addEventListener('click', handleModalClick);
    
    // Store reference to the handler on the element for potential cleanup
    modalElement._clickHandler = handleModalClick;
}

/**
 * Set up editable fields (notes and usage tips) in the model modal
 * @param {string} filePath - The full file path of the model
 * @param {string} modelType - Type of model ('loras' or 'checkpoints' or 'embeddings')
 */
function setupEditableFields(filePath, modelType) {
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

    // LoRA specific field setup
    if (modelType === 'loras') {
        setupLoraSpecificFields(filePath);
    }
}

function setupLoraSpecificFields(filePath) {
    const presetSelector = document.getElementById('preset-selector');
    const presetValue = document.getElementById('preset-value');
    const addPresetBtn = document.querySelector('.add-preset-btn');
    const presetTags = document.querySelector('.preset-tags');

    if (!presetSelector || !presetValue || !addPresetBtn || !presetTags) return;

    presetSelector.addEventListener('change', function() {
        const selected = this.value;
        if (selected) {
            presetValue.style.display = 'inline-block';
            presetValue.min = selected.includes('strength') ? -10 : 0;
            presetValue.max = selected.includes('strength') ? 10 : 10;
            presetValue.step = 0.5;
            if (selected === 'clip_skip') {
                presetValue.type = 'number';
                presetValue.step = 1;
            }
            // Add auto-focus
            setTimeout(() => presetValue.focus(), 0);
        } else {
            presetValue.style.display = 'none';
        }
    });

    addPresetBtn.addEventListener('click', async function() {
        const key = presetSelector.value;
        const value = presetValue.value;
        
        if (!key || !value) return;

        const loraCard = document.querySelector(`.model-card[data-filepath="${filePath}"]`);
        const currentPresets = parsePresets(loraCard?.dataset.usage_tips);
        
        currentPresets[key] = parseFloat(value);
        const newPresetsJson = JSON.stringify(currentPresets);

        await getModelApiClient().saveModelMetadata(filePath, { usage_tips: newPresetsJson });

        presetTags.innerHTML = renderPresetTags(currentPresets);
        
        presetSelector.value = '';
        presetValue.value = '';
        presetValue.style.display = 'none';
    });

    // Add keydown event for preset value
    presetValue.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            addPresetBtn.click();
        }
    });
}

/**
 * Save model notes
 * @param {string} filePath - Path to the model file
 */
async function saveNotes(filePath) {
    const content = document.querySelector('.notes-content').textContent;
    try {
        await getModelApiClient().saveModelMetadata(filePath, { notes: content });

        showToast('Notes saved successfully', 'success');
    } catch (error) {
        showToast('Failed to save notes', 'error');
    }
}

// Export the model modal API
const modelModal = {
    show: showModelModal,
    toggleShowcase,
    scrollToTop
};

export { modelModal };