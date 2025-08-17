/**
 * ModelMetadata.js
 * Handles model metadata editing functionality - General version
 */
import { showToast } from '../../utils/uiHelpers.js';
import { BASE_MODELS } from '../../utils/constants.js';
import { getModelApiClient } from '../../api/modelApiFactory.js';

/**
 * Set up model name editing functionality
 * @param {string} filePath - File path
 */
export function setupModelNameEditing(filePath) {
    const modelNameContent = document.querySelector('.model-name-content');
    const editBtn = document.querySelector('.edit-model-name-btn');
    
    if (!modelNameContent || !editBtn) return;
    
    // Store the file path in a data attribute for later use
    modelNameContent.dataset.filePath = filePath;
    
    // Show edit button on hover
    const modelNameHeader = document.querySelector('.model-name-header');
    modelNameHeader.addEventListener('mouseenter', () => {
        editBtn.classList.add('visible');
    });
    
    modelNameHeader.addEventListener('mouseleave', () => {
        if (!modelNameHeader.classList.contains('editing')) {
            editBtn.classList.remove('visible');
        }
    });
    
    // Handle edit button click
    editBtn.addEventListener('click', () => {
        modelNameHeader.classList.add('editing');
        modelNameContent.setAttribute('contenteditable', 'true');
        // Store original value for comparison later
        modelNameContent.dataset.originalValue = modelNameContent.textContent.trim();
        modelNameContent.focus();
        
        // Place cursor at the end
        const range = document.createRange();
        const sel = window.getSelection();
        if (modelNameContent.childNodes.length > 0) {
            range.setStart(modelNameContent.childNodes[0], modelNameContent.textContent.length);
            range.collapse(true);
            sel.removeAllRanges();
            sel.addRange(range);
        }
        
        editBtn.classList.add('visible');
    });
    
    // Handle keyboard events in edit mode
    modelNameContent.addEventListener('keydown', function(e) {
        if (!this.getAttribute('contenteditable')) return;
        
        if (e.key === 'Enter') {
            e.preventDefault();
            this.blur(); // Trigger save on Enter
        } else if (e.key === 'Escape') {
            e.preventDefault();
            // Restore original value
            this.textContent = this.dataset.originalValue;
            exitEditMode();
        }
    });
    
    // Limit model name length
    modelNameContent.addEventListener('input', function() {
        if (!this.getAttribute('contenteditable')) return;
        
        // Limit model name length
        if (this.textContent.length > 100) {
            this.textContent = this.textContent.substring(0, 100);
            // Place cursor at the end
            const range = document.createRange();
            const sel = window.getSelection();
            range.setStart(this.childNodes[0], 100);
            range.collapse(true);
            sel.removeAllRanges();
            sel.addRange(range);
            
            showToast('Model name is limited to 100 characters', 'warning');
        }
    });
    
    // Handle focus out - save changes
    modelNameContent.addEventListener('blur', async function() {
        if (!this.getAttribute('contenteditable')) return;
        
        const newModelName = this.textContent.trim();
        const originalValue = this.dataset.originalValue;
        
        // Basic validation
        if (!newModelName) {
            // Restore original value if empty
            this.textContent = originalValue;
            showToast('Model name cannot be empty', 'error');
            exitEditMode();
            return;
        }
        
        if (newModelName === originalValue) {
            // No changes, just exit edit mode
            exitEditMode();
            return;
        }
        
        try {
            // Get the file path from the dataset
            const filePath = this.dataset.filePath;
            
            await getModelApiClient().saveModelMetadata(filePath, { model_name: newModelName });
            
            showToast('Model name updated successfully', 'success');
        } catch (error) {
            console.error('Error updating model name:', error);
            this.textContent = originalValue; // Restore original model name
            showToast('Failed to update model name', 'error');
        } finally {
            exitEditMode();
        }
    });
    
    function exitEditMode() {
        modelNameContent.removeAttribute('contenteditable');
        modelNameHeader.classList.remove('editing');
        editBtn.classList.remove('visible');
    }
}

/**
 * Set up base model editing functionality
 * @param {string} filePath - File path
 */
export function setupBaseModelEditing(filePath) {
    const baseModelContent = document.querySelector('.base-model-content');
    const editBtn = document.querySelector('.edit-base-model-btn');
    
    if (!baseModelContent || !editBtn) return;
    
    // Store the file path in a data attribute for later use
    baseModelContent.dataset.filePath = filePath;
    
    // Show edit button on hover
    const baseModelDisplay = document.querySelector('.base-model-display');
    baseModelDisplay.addEventListener('mouseenter', () => {
        editBtn.classList.add('visible');
    });
    
    baseModelDisplay.addEventListener('mouseleave', () => {
        if (!baseModelDisplay.classList.contains('editing')) {
            editBtn.classList.remove('visible');
        }
    });
    
    // Handle edit button click
    editBtn.addEventListener('click', () => {
        baseModelDisplay.classList.add('editing');
        
        // Store the original value to check for changes later
        const originalValue = baseModelContent.textContent.trim();
        
        // Create dropdown selector to replace the base model content
        const currentValue = originalValue;
        const dropdown = document.createElement('select');
        dropdown.className = 'base-model-selector';
        
        // Flag to track if a change was made
        let valueChanged = false;
        
        // Add options from BASE_MODELS constants
        const baseModelCategories = {
            'Stable Diffusion 1.x': [BASE_MODELS.SD_1_4, BASE_MODELS.SD_1_5, BASE_MODELS.SD_1_5_LCM, BASE_MODELS.SD_1_5_HYPER],
            'Stable Diffusion 2.x': [BASE_MODELS.SD_2_0, BASE_MODELS.SD_2_1],
            'Stable Diffusion 3.x': [BASE_MODELS.SD_3, BASE_MODELS.SD_3_5, BASE_MODELS.SD_3_5_MEDIUM, BASE_MODELS.SD_3_5_LARGE, BASE_MODELS.SD_3_5_LARGE_TURBO],
            'SDXL': [BASE_MODELS.SDXL, BASE_MODELS.SDXL_LIGHTNING, BASE_MODELS.SDXL_HYPER],
            'Video Models': [BASE_MODELS.SVD, BASE_MODELS.LTXV, BASE_MODELS.WAN_VIDEO, BASE_MODELS.HUNYUAN_VIDEO],
            'Other Models': [
                BASE_MODELS.FLUX_1_D, BASE_MODELS.FLUX_1_S, BASE_MODELS.FLUX_1_KONTEXT, BASE_MODELS.AURAFLOW,
                BASE_MODELS.PIXART_A, BASE_MODELS.PIXART_E, BASE_MODELS.HUNYUAN_1,
                BASE_MODELS.LUMINA, BASE_MODELS.KOLORS, BASE_MODELS.NOOBAI,
                BASE_MODELS.ILLUSTRIOUS, BASE_MODELS.PONY, BASE_MODELS.HIDREAM,
                BASE_MODELS.QWEN, BASE_MODELS.UNKNOWN
            ]
        };
        
        // Create option groups for better organization
        Object.entries(baseModelCategories).forEach(([category, models]) => {
            const group = document.createElement('optgroup');
            group.label = category;
            
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                option.selected = model === currentValue;
                group.appendChild(option);
            });
            
            dropdown.appendChild(group);
        });
        
        // Replace content with dropdown
        baseModelContent.style.display = 'none';
        baseModelDisplay.insertBefore(dropdown, editBtn);
        
        // Hide edit button during editing
        editBtn.style.display = 'none';
        
        // Focus the dropdown
        dropdown.focus();
        
        // Handle dropdown change
        dropdown.addEventListener('change', function() {
            const selectedModel = this.value;
            baseModelContent.textContent = selectedModel;
            
            // Mark that a change was made if the value differs from original
            if (selectedModel !== originalValue) {
                valueChanged = true;
            } else {
                valueChanged = false;
            }
        });
        
        // Function to save changes and exit edit mode
        const saveAndExit = function() {
            // Check if dropdown still exists and remove it
            if (dropdown && dropdown.parentNode === baseModelDisplay) {
                baseModelDisplay.removeChild(dropdown);
            }
            
            // Show the content and edit button
            baseModelContent.style.display = '';
            editBtn.style.display = '';
            
            // Remove editing class
            baseModelDisplay.classList.remove('editing');
            
            // Only save if the value has actually changed
            if (valueChanged || baseModelContent.textContent.trim() !== originalValue) {
                // Get file path from the dataset
                const filePath = baseModelContent.dataset.filePath;
                
                // Save the changes, passing the original value for comparison
                saveBaseModel(filePath, originalValue);
            }
            
            // Remove this event listener
            document.removeEventListener('click', outsideClickHandler);
        };
        
        // Handle outside clicks to save and exit
        const outsideClickHandler = function(e) {
            // If click is outside the dropdown and base model display
            if (!baseModelDisplay.contains(e.target)) {
                saveAndExit();
            }
        };
        
        // Add delayed event listener for outside clicks
        setTimeout(() => {
            document.addEventListener('click', outsideClickHandler);
        }, 0);
        
        // Also handle dropdown blur event
        dropdown.addEventListener('blur', function(e) {
            // Only save if the related target is not the edit button or inside the baseModelDisplay
            if (!baseModelDisplay.contains(e.relatedTarget)) {
                saveAndExit();
            }
        });
    });
}

/**
 * Save base model
 * @param {string} filePath - File path
 * @param {string} originalValue - Original value (for comparison)
 */
async function saveBaseModel(filePath, originalValue) {
    const baseModelElement = document.querySelector('.base-model-content');
    const newBaseModel = baseModelElement.textContent.trim();
    
    // Only save if the value has actually changed
    if (newBaseModel === originalValue) {
        return; // No change, no need to save
    }
    
    try {
        await getModelApiClient().saveModelMetadata(filePath, { base_model: newBaseModel });
        
        showToast('Base model updated successfully', 'success');
    } catch (error) {
        showToast('Failed to update base model', 'error');
    }
}

/**
 * Set up file name editing functionality
 * @param {string} filePath - File path
 */
export function setupFileNameEditing(filePath) {
    const fileNameContent = document.querySelector('.file-name-content');
    const editBtn = document.querySelector('.edit-file-name-btn');
    
    if (!fileNameContent || !editBtn) return;
    
    // Store the original file path
    fileNameContent.dataset.filePath = filePath;
    
    // Show edit button on hover
    const fileNameWrapper = document.querySelector('.file-name-wrapper');
    fileNameWrapper.addEventListener('mouseenter', () => {
        editBtn.classList.add('visible');
    });
    
    fileNameWrapper.addEventListener('mouseleave', () => {
        if (!fileNameWrapper.classList.contains('editing')) {
            editBtn.classList.remove('visible');
        }
    });
    
    // Handle edit button click
    editBtn.addEventListener('click', () => {
        fileNameWrapper.classList.add('editing');
        fileNameContent.setAttribute('contenteditable', 'true');
        fileNameContent.focus();
        
        // Store original value for comparison later
        fileNameContent.dataset.originalValue = fileNameContent.textContent.trim();
        
        // Place cursor at the end
        const range = document.createRange();
        const sel = window.getSelection();
        range.selectNodeContents(fileNameContent);
        range.collapse(false);
        sel.removeAllRanges();
        sel.addRange(range);
        
        editBtn.classList.add('visible');
    });
    
    // Handle keyboard events in edit mode
    fileNameContent.addEventListener('keydown', function(e) {
        if (!this.getAttribute('contenteditable')) return;
        
        if (e.key === 'Enter') {
            e.preventDefault();
            this.blur(); // Trigger save on Enter
        } else if (e.key === 'Escape') {
            e.preventDefault();
            // Restore original value
            this.textContent = this.dataset.originalValue;
            exitEditMode();
        }
    });
    
    // Handle input validation
    fileNameContent.addEventListener('input', function() {
        if (!this.getAttribute('contenteditable')) return;
        
        // Replace invalid characters for filenames
        const invalidChars = /[\\/:*?"<>|]/g;
        if (invalidChars.test(this.textContent)) {
            const cursorPos = window.getSelection().getRangeAt(0).startOffset;
            this.textContent = this.textContent.replace(invalidChars, '');
            
            // Restore cursor position
            const range = document.createRange();
            const sel = window.getSelection();
            const newPos = Math.min(cursorPos, this.textContent.length);
            
            if (this.firstChild) {
                range.setStart(this.firstChild, newPos);
                range.collapse(true);
                sel.removeAllRanges();
                sel.addRange(range);
            }
            
            showToast('Invalid characters removed from filename', 'warning');
        }
    });
    
    // Handle focus out - save changes
    fileNameContent.addEventListener('blur', async function() {
        if (!this.getAttribute('contenteditable')) return;
        
        const newFileName = this.textContent.trim();
        const originalValue = this.dataset.originalValue;
        
        // Basic validation
        if (!newFileName) {
            // Restore original value if empty
            this.textContent = originalValue;
            showToast('File name cannot be empty', 'error');
            exitEditMode();
            return;
        }
        
        if (newFileName === originalValue) {
            // No changes, just exit edit mode
            exitEditMode();
            return;
        }
        
        try {
            // Get the file path from the dataset
            const filePath = this.dataset.filePath;
            
            await getModelApiClient().renameModelFile(filePath, newFileName);
        } catch (error) {
            console.error('Error renaming file:', error);
            this.textContent = originalValue; // Restore original file name
            showToast(`Failed to rename file: ${error.message}`, 'error');
        } finally {
            exitEditMode();
        }
    });
    
    function exitEditMode() {
        fileNameContent.removeAttribute('contenteditable');
        fileNameWrapper.classList.remove('editing');
        editBtn.classList.remove('visible');
    }
}
