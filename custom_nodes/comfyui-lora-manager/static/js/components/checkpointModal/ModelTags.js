/**
 * ModelTags.js
 * Module for handling checkpoint model tag editing functionality
 */
import { showToast } from '../../utils/uiHelpers.js';
import { saveModelMetadata } from '../../api/checkpointApi.js';

// Preset tag suggestions
const PRESET_TAGS = [
  'character', 'style', 'concept', 'clothing', 'base model',
  'poses', 'background', 'vehicle', 'buildings', 
  'objects', 'animal'
];

// Create a named function so we can remove it later
let saveTagsHandler = null;

/**
 * Set up tag editing mode
 */
export function setupTagEditMode() {
    const editBtn = document.querySelector('.edit-tags-btn');
    if (!editBtn) return;
    
    // Store original tags for restoring on cancel
    let originalTags = [];
    
    // Remove any previously attached click handler
    if (editBtn._hasClickHandler) {
        editBtn.removeEventListener('click', editBtn._clickHandler);
    }
    
    // Create new handler and store reference
    const editBtnClickHandler = function() {
        const tagsSection = document.querySelector('.model-tags-container');
        const isEditMode = tagsSection.classList.toggle('edit-mode');
        const filePath = this.dataset.filePath;
        
        // Toggle edit mode UI elements
        const compactTagsDisplay = tagsSection.querySelector('.model-tags-compact');
        const tagsEditContainer = tagsSection.querySelector('.metadata-edit-container');
        
        if (isEditMode) {
            // Enter edit mode
            this.innerHTML = '<i class="fas fa-times"></i>'; // Change to cancel icon
            this.title = "Cancel editing";
            
            // Get all tags from tooltip, not just the visible ones in compact display
            originalTags = Array.from(
                tagsSection.querySelectorAll('.tooltip-tag')
            ).map(tag => tag.textContent);
            
            // Hide compact display, show edit container
            compactTagsDisplay.style.display = 'none';
            
            // If edit container doesn't exist yet, create it
            if (!tagsEditContainer) {
                const editContainer = document.createElement('div');
                editContainer.className = 'metadata-edit-container';
                
                // Move the edit button inside the container header for better visibility
                const editBtnClone = editBtn.cloneNode(true);
                editBtnClone.classList.add('metadata-header-btn');
                
                // Create edit UI with edit button in the header
                editContainer.innerHTML = createTagEditUI(originalTags, editBtnClone.outerHTML);
                tagsSection.appendChild(editContainer);
                
                // Setup the tag input field behavior
                setupTagInput();
                
                // Create and add preset suggestions dropdown
                const tagForm = editContainer.querySelector('.metadata-add-form');
                const suggestionsDropdown = createSuggestionsDropdown(originalTags);
                tagForm.appendChild(suggestionsDropdown);
                
                // Setup delete buttons for existing tags
                setupDeleteButtons();
                
                // Transfer click event from original button to the cloned one
                const newEditBtn = editContainer.querySelector('.metadata-header-btn');
                if (newEditBtn) {
                    newEditBtn.addEventListener('click', function() {
                        editBtn.click();
                    });
                }
                
                // Hide the original button when in edit mode
                editBtn.style.display = 'none';
            } else {
                // Just show the existing edit container
                tagsEditContainer.style.display = 'block';
                editBtn.style.display = 'none';
            }
        } else {
            // Exit edit mode
            this.innerHTML = '<i class="fas fa-pencil-alt"></i>'; // Change back to edit icon
            this.title = "Edit tags";
            editBtn.style.display = 'block';
            
            // Show compact display, hide edit container
            compactTagsDisplay.style.display = 'flex';
            if (tagsEditContainer) tagsEditContainer.style.display = 'none';
            
            // Check if we're exiting edit mode due to "Save" or "Cancel"
            if (!this.dataset.skipRestore) {
                // If canceling, restore original tags
                restoreOriginalTags(tagsSection, originalTags);
            } else {
                // Reset the skip restore flag
                delete this.dataset.skipRestore;
            }
        }
    };
    
    // Store the handler reference on the button itself
    editBtn._clickHandler = editBtnClickHandler;
    editBtn._hasClickHandler = true;
    editBtn.addEventListener('click', editBtnClickHandler);
    
    // Clean up any previous document click handler
    if (saveTagsHandler) {
        document.removeEventListener('click', saveTagsHandler);
    }
    
    // Create new save handler and store reference
    saveTagsHandler = function(e) {
        if (e.target.classList.contains('save-tags-btn') || 
            e.target.closest('.save-tags-btn')) {
            saveTags();
        }
    };
    
    // Add the new handler
    document.addEventListener('click', saveTagsHandler);
}

/**
 * Create the tag editing UI
 * @param {Array} currentTags - Current tags
 * @param {string} editBtnHTML - HTML for the edit button to include in header
 * @returns {string} HTML markup for tag editing UI
 */
function createTagEditUI(currentTags, editBtnHTML = '') {
    return `
        <div class="metadata-edit-content">
            <div class="metadata-edit-header">
                <label>Edit Tags</label>
                ${editBtnHTML}
            </div>
            <div class="metadata-items">
                ${currentTags.map(tag => `
                    <div class="metadata-item" data-tag="${tag}">
                        <span class="metadata-item-content">${tag}</span>
                        <button class="metadata-delete-btn">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                `).join('')}
            </div>
            <div class="metadata-edit-controls">
                <button class="save-tags-btn" title="Save changes">
                    <i class="fas fa-save"></i> Save
                </button>
            </div>
            <div class="metadata-add-form">
                <input type="text" class="metadata-input" placeholder="Type to add or click suggestions below">
            </div>
        </div>
    `;
}

/**
 * Create suggestions dropdown with preset tags
 * @param {Array} existingTags - Already added tags
 * @returns {HTMLElement} - Dropdown element
 */
function createSuggestionsDropdown(existingTags = []) {
    const dropdown = document.createElement('div');
    dropdown.className = 'metadata-suggestions-dropdown';
    
    // Create header
    const header = document.createElement('div');
    header.className = 'metadata-suggestions-header';
    header.innerHTML = `
        <span>Suggested Tags</span>
        <small>Click to add</small>
    `;
    dropdown.appendChild(header);
    
    // Create tag container
    const container = document.createElement('div');
    container.className = 'metadata-suggestions-container';
    
    // Add each preset tag as a suggestion
    PRESET_TAGS.forEach(tag => {
        const isAdded = existingTags.includes(tag);
        
        const item = document.createElement('div');
        item.className = `metadata-suggestion-item ${isAdded ? 'already-added' : ''}`;
        item.title = tag;
        item.innerHTML = `
            <span class="metadata-suggestion-text">${tag}</span>
            ${isAdded ? '<span class="added-indicator"><i class="fas fa-check"></i></span>' : ''}
        `;
        
        if (!isAdded) {
            item.addEventListener('click', () => {
                addNewTag(tag);
                
                // Also populate the input field for potential editing
                const input = document.querySelector('.metadata-input');
                if (input) input.value = tag;
                
                // Focus on the input
                if (input) input.focus();
                
                // Update dropdown without removing it
                updateSuggestionsDropdown();
            });
        }
        
        container.appendChild(item);
    });
    
    dropdown.appendChild(container);
    return dropdown;
}

/**
 * Set up tag input behavior
 */
function setupTagInput() {
    const tagInput = document.querySelector('.metadata-input');
    
    if (tagInput) {
        tagInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                addNewTag(this.value);
                this.value = ''; // Clear input after adding
            }
        });
    }
}

/**
 * Set up delete buttons for tags
 */
function setupDeleteButtons() {
    document.querySelectorAll('.metadata-delete-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.stopPropagation();
            const tag = this.closest('.metadata-item');
            tag.remove();
            
            // Update status of items in the suggestion dropdown
            updateSuggestionsDropdown();
        });
    });
}

/**
 * Add a new tag
 * @param {string} tag - Tag to add
 */
function addNewTag(tag) {
    tag = tag.trim().toLowerCase();
    if (!tag) return;
    
    const tagsContainer = document.querySelector('.metadata-items');
    if (!tagsContainer) return;
    
    // Validation: Check length
    if (tag.length > 30) {
        showToast('Tag should not exceed 30 characters', 'error');
        return;
    }
    
    // Validation: Check total number
    const currentTags = tagsContainer.querySelectorAll('.metadata-item');
    if (currentTags.length >= 30) {
        showToast('Maximum 30 tags allowed', 'error');
        return;
    }
    
    // Validation: Check for duplicates
    const existingTags = Array.from(currentTags).map(tag => tag.dataset.tag);
    if (existingTags.includes(tag)) {
        showToast('This tag already exists', 'error');
        return;
    }
    
    // Create new tag
    const newTag = document.createElement('div');
    newTag.className = 'metadata-item';
    newTag.dataset.tag = tag;
    newTag.innerHTML = `
        <span class="metadata-item-content">${tag}</span>
        <button class="metadata-delete-btn">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add event listener to delete button
    const deleteBtn = newTag.querySelector('.metadata-delete-btn');
    deleteBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        newTag.remove();
        
        // Update status of items in the suggestion dropdown
        updateSuggestionsDropdown();
    });
    
    tagsContainer.appendChild(newTag);
    
    // Update status of items in the suggestions dropdown
    updateSuggestionsDropdown();
}

/**
 * Update status of items in the suggestions dropdown
 */
function updateSuggestionsDropdown() {
    const dropdown = document.querySelector('.metadata-suggestions-dropdown');
    if (!dropdown) return;
    
    // Get all current tags
    const currentTags = document.querySelectorAll('.metadata-item');
    const existingTags = Array.from(currentTags).map(tag => tag.dataset.tag);
    
    // Update status of each item in dropdown
    dropdown.querySelectorAll('.metadata-suggestion-item').forEach(item => {
        const tagText = item.querySelector('.metadata-suggestion-text').textContent;
        const isAdded = existingTags.includes(tagText);
        
        if (isAdded) {
            item.classList.add('already-added');
            
            // Add indicator if it doesn't exist
            let indicator = item.querySelector('.added-indicator');
            if (!indicator) {
                indicator = document.createElement('span');
                indicator.className = 'added-indicator';
                indicator.innerHTML = '<i class="fas fa-check"></i>';
                item.appendChild(indicator);
            }
            
            // Remove click event
            item.onclick = null;
        } else {
            // Re-enable items that are no longer in the list
            item.classList.remove('already-added');
            
            // Remove indicator if it exists
            const indicator = item.querySelector('.added-indicator');
            if (indicator) indicator.remove();
            
            // Restore click event if not already set
            if (!item.onclick) {
                item.onclick = () => {
                    const tag = item.querySelector('.metadata-suggestion-text').textContent;
                    addNewTag(tag);
                    
                    // Also populate the input field
                    const input = document.querySelector('.metadata-input');
                    if (input) input.value = tag;
                    
                    // Focus the input
                    if (input) input.focus();
                };
            }
        }
    });
}

/**
 * Restore original tags when canceling edit
 * @param {HTMLElement} section - The tags section
 * @param {Array} originalTags - Original tags array
 */
function restoreOriginalTags(section, originalTags) {
    // Nothing to do here as we're just hiding the edit UI
    // and showing the original compact tags which weren't modified
}

/**
 * Save tags
 */
async function saveTags() {
    const editBtn = document.querySelector('.edit-tags-btn');
    if (!editBtn) return;
    
    const filePath = editBtn.dataset.filePath;
    const tagElements = document.querySelectorAll('.metadata-item');
    const tags = Array.from(tagElements).map(tag => tag.dataset.tag);

    // Get original tags to compare
    const originalTagElements = document.querySelectorAll('.tooltip-tag');
    const originalTags = Array.from(originalTagElements).map(tag => tag.textContent);
    
    // Check if tags have actually changed
    const tagsChanged = JSON.stringify(tags) !== JSON.stringify(originalTags);
    
    if (!tagsChanged) {
        // No changes made, just exit edit mode without API call
        editBtn.dataset.skipRestore = "true";
        editBtn.click();
        return;
    }
    
    try {
        // Save tags metadata
        await saveModelMetadata(filePath, { tags: tags });
        
        // Set flag to skip restoring original tags when exiting edit mode
        editBtn.dataset.skipRestore = "true";
        
        // Update the compact tags display
        const compactTagsContainer = document.querySelector('.model-tags-container');
        if (compactTagsContainer) {
            // Generate new compact tags HTML
            const compactTagsDisplay = compactTagsContainer.querySelector('.model-tags-compact');
            
            if (compactTagsDisplay) {
                // Clear current tags
                compactTagsDisplay.innerHTML = '';
                
                // Add visible tags (up to 5)
                const visibleTags = tags.slice(0, 5);
                visibleTags.forEach(tag => {
                    const span = document.createElement('span');
                    span.className = 'model-tag-compact';
                    span.textContent = tag;
                    compactTagsDisplay.appendChild(span);
                });
                
                // Add more indicator if needed
                const remainingCount = Math.max(0, tags.length - 5);
                if (remainingCount > 0) {
                    const more = document.createElement('span');
                    more.className = 'model-tag-more';
                    more.dataset.count = remainingCount;
                    more.textContent = `+${remainingCount}`;
                    compactTagsDisplay.appendChild(more);
                }
            }
            
            // Update tooltip content
            const tooltipContent = compactTagsContainer.querySelector('.tooltip-content');
            if (tooltipContent) {
                tooltipContent.innerHTML = '';
                
                tags.forEach(tag => {
                    const span = document.createElement('span');
                    span.className = 'tooltip-tag';
                    span.textContent = tag;
                    tooltipContent.appendChild(span);
                });
            }
        }
        
        // Exit edit mode
        editBtn.click();
        
        showToast('Tags updated successfully', 'success');
    } catch (error) {
        console.error('Error saving tags:', error);
        showToast('Failed to update tags', 'error');
    }
}
