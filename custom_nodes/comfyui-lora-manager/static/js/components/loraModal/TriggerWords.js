/**
 * TriggerWords.js
 * Module that handles trigger word functionality for LoRA models
 */
import { showToast, copyToClipboard } from '../../utils/uiHelpers.js';
import { saveModelMetadata } from '../../api/loraApi.js';

/**
 * Fetch trained words for a model
 * @param {string} filePath - Path to the model file
 * @returns {Promise<Object>} - Object with trained words and class tokens
 */
async function fetchTrainedWords(filePath) {
    try {
        const response = await fetch(`/api/trained-words?file_path=${encodeURIComponent(filePath)}`);
        const data = await response.json();
        
        if (data.success) {
            return {
                trainedWords: data.trained_words || [], // Returns array of [word, frequency] pairs
                classTokens: data.class_tokens  // Can be null or a string
            };
        } else {
            throw new Error(data.error || 'Failed to fetch trained words');
        }
    } catch (error) {
        console.error('Error fetching trained words:', error);
        showToast('Could not load trained words', 'error');
        return { trainedWords: [], classTokens: null };
    }
}

/**
 * Create suggestion dropdown with trained words as tags
 * @param {Array} trainedWords - Array of [word, frequency] pairs
 * @param {string|null} classTokens - Class tokens from training
 * @param {Array} existingWords - Already added trigger words
 * @returns {HTMLElement} - Dropdown element
 */
function createSuggestionDropdown(trainedWords, classTokens, existingWords = []) {
    const dropdown = document.createElement('div');
    dropdown.className = 'metadata-suggestions-dropdown';
    
    // Create header
    const header = document.createElement('div');
    header.className = 'metadata-suggestions-header';
    
    // No suggestions case
    if ((!trainedWords || trainedWords.length === 0) && !classTokens) {
        header.innerHTML = '<span>No suggestions available</span>';
        dropdown.appendChild(header);
        dropdown.innerHTML += '<div class="no-suggestions">No trained words or class tokens found in this model. You can manually enter trigger words.</div>';
        return dropdown;
    }
    
    // Sort trained words by frequency (highest first) if available
    if (trainedWords && trainedWords.length > 0) {
        trainedWords.sort((a, b) => b[1] - a[1]);
    }
    
    // Add class tokens section if available
    if (classTokens) {
        // Add class tokens header
        const classTokensHeader = document.createElement('div');
        classTokensHeader.className = 'metadata-suggestions-header';
        classTokensHeader.innerHTML = `
            <span>Class Token</span>
            <small>Add to your prompt for best results</small>
        `;
        dropdown.appendChild(classTokensHeader);
        
        // Add class tokens container
        const classTokensContainer = document.createElement('div');
        classTokensContainer.className = 'class-tokens-container';
        
        // Create a special item for the class token
        const tokenItem = document.createElement('div');
        tokenItem.className = `metadata-suggestion-item class-token-item ${existingWords.includes(classTokens) ? 'already-added' : ''}`;
        tokenItem.title = `Class token: ${classTokens}`;
        tokenItem.innerHTML = `
            <span class="metadata-suggestion-text">${classTokens}</span>
            <div class="metadata-suggestion-meta">
                <span class="token-badge">Class Token</span>
                ${existingWords.includes(classTokens) ? 
                    '<span class="added-indicator"><i class="fas fa-check"></i></span>' : ''}
            </div>
        `;
        
        // Add click handler if not already added
        if (!existingWords.includes(classTokens)) {
            tokenItem.addEventListener('click', () => {
                // Automatically add this word
                addNewTriggerWord(classTokens);
                
                // Also populate the input field for potential editing
                const input = document.querySelector('.metadata-input');
                if (input) input.value = classTokens;
                
                // Focus on the input
                if (input) input.focus();
                
                // Update dropdown without removing it
                updateTrainedWordsDropdown();
            });
        }
        
        classTokensContainer.appendChild(tokenItem);
        dropdown.appendChild(classTokensContainer);
        
        // Add separator if we also have trained words
        if (trainedWords && trainedWords.length > 0) {
            const separator = document.createElement('div');
            separator.className = 'dropdown-separator';
            dropdown.appendChild(separator);
        }
    }
    
    // Add trained words header if we have any
    if (trainedWords && trainedWords.length > 0) {
        header.innerHTML = `
            <span>Word Suggestions</span>
            <small>${trainedWords.length} words found</small>
        `;
        dropdown.appendChild(header);
        
        // Create tag container for trained words
        const container = document.createElement('div');
        container.className = 'metadata-suggestions-container';
        
        // Add each trained word as a tag
        trainedWords.forEach(([word, frequency]) => {
            const isAdded = existingWords.includes(word);
            
            const item = document.createElement('div');
            item.className = `metadata-suggestion-item ${isAdded ? 'already-added' : ''}`;
            item.title = word; // Show full word on hover if truncated
            item.innerHTML = `
                <span class="metadata-suggestion-text">${word}</span>
                <div class="metadata-suggestion-meta">
                    <span class="trained-word-freq">${frequency}</span>
                    ${isAdded ? '<span class="added-indicator"><i class="fas fa-check"></i></span>' : ''}
                </div>
            `;
            
            if (!isAdded) {
                item.addEventListener('click', () => {
                    // Automatically add this word
                    addNewTriggerWord(word);
                    
                    // Also populate the input field for potential editing
                    const input = document.querySelector('.metadata-input');
                    if (input) input.value = word;
                    
                    // Focus on the input
                    if (input) input.focus();
                    
                    // Update dropdown without removing it
                    updateTrainedWordsDropdown();
                });
            }
            
            container.appendChild(item);
        });
        
        dropdown.appendChild(container);
    } else if (!classTokens) {
        // If we have neither class tokens nor trained words
        dropdown.innerHTML += '<div class="no-suggestions">No word suggestions found in this model. You can manually enter trigger words.</div>';
    }
    
    return dropdown;
}

/**
 * Render trigger words
 * @param {Array} words - Array of trigger words
 * @param {string} filePath - File path
 * @returns {string} HTML content
 */
export function renderTriggerWords(words, filePath) {
    if (!words.length) return `
        <div class="info-item full-width trigger-words">
            <div class="trigger-words-header">
                <label>Trigger Words</label>
                <button class="edit-trigger-words-btn metadata-edit-btn" data-file-path="${filePath}" title="Edit trigger words">
                    <i class="fas fa-pencil-alt"></i>
                </button>
            </div>
            <div class="trigger-words-content">
                <span class="no-trigger-words">No trigger word needed</span>
                <div class="trigger-words-tags" style="display:none;"></div>
            </div>
            <div class="metadata-edit-controls" style="display:none;">
                <button class="metadata-save-btn" title="Save changes">
                    <i class="fas fa-save"></i> Save
                </button>
            </div>
            <div class="metadata-add-form" style="display:none;">
                <input type="text" class="metadata-input" placeholder="Type to add or click suggestions below">
            </div>
        </div>
    `;
    
    return `
        <div class="info-item full-width trigger-words">
            <div class="trigger-words-header">
                <label>Trigger Words</label>
                <button class="edit-trigger-words-btn metadata-edit-btn" data-file-path="${filePath}" title="Edit trigger words">
                    <i class="fas fa-pencil-alt"></i>
                </button>
            </div>
            <div class="trigger-words-content">
                <div class="trigger-words-tags">
                    ${words.map(word => `
                        <div class="trigger-word-tag" data-word="${word}" onclick="copyTriggerWord('${word}')">
                            <span class="trigger-word-content">${word}</span>
                            <span class="trigger-word-copy">
                                <i class="fas fa-copy"></i>
                            </span>
                            <button class="metadata-delete-btn" style="display:none;" onclick="event.stopPropagation();">
                                <i class="fas fa-times"></i>
                            </button>
                        </div>
                    `).join('')}
                </div>
            </div>
            <div class="metadata-edit-controls" style="display:none;">
                <button class="metadata-save-btn" title="Save changes">
                    <i class="fas fa-save"></i> Save
                </button>
            </div>
            <div class="metadata-add-form" style="display:none;">
                <input type="text" class="metadata-input" placeholder="Type to add or click suggestions below">
            </div>
        </div>
    `;
}

/**
 * Set up trigger words edit mode
 */
export function setupTriggerWordsEditMode() {
    // Store trained words data
    let trainedWordsList = [];
    let classTokensValue = null;
    let isTrainedWordsLoaded = false;
    // Store original trigger words for restoring on cancel
    let originalTriggerWords = [];
    
    const editBtn = document.querySelector('.edit-trigger-words-btn');
    if (!editBtn) return;
    
    editBtn.addEventListener('click', async function() {
        const triggerWordsSection = this.closest('.trigger-words');
        const isEditMode = triggerWordsSection.classList.toggle('edit-mode');
        const filePath = this.dataset.filePath;
        
        // Toggle edit mode UI elements
        const triggerWordTags = triggerWordsSection.querySelectorAll('.trigger-word-tag');
        const editControls = triggerWordsSection.querySelector('.metadata-edit-controls');
        const addForm = triggerWordsSection.querySelector('.metadata-add-form');
        const noTriggerWords = triggerWordsSection.querySelector('.no-trigger-words');
        const tagsContainer = triggerWordsSection.querySelector('.trigger-words-tags');
        
        if (isEditMode) {
            this.innerHTML = '<i class="fas fa-times"></i>'; // Change to cancel icon
            this.title = "Cancel editing";
            
            // Store original trigger words for potential restoration
            originalTriggerWords = Array.from(triggerWordTags).map(tag => tag.dataset.word);
            
            // Show edit controls and input form
            editControls.style.display = 'flex';
            addForm.style.display = 'flex';
            
            // If we have no trigger words yet, hide the "No trigger word needed" text
            // and show the empty tags container
            if (noTriggerWords) {
                noTriggerWords.style.display = 'none';
                if (tagsContainer) tagsContainer.style.display = 'flex';
            }
            
            // Disable click-to-copy and show delete buttons
            triggerWordTags.forEach(tag => {
                tag.onclick = null;
                const copyIcon = tag.querySelector('.trigger-word-copy');
                const deleteBtn = tag.querySelector('.metadata-delete-btn');
                
                if (copyIcon) copyIcon.style.display = 'none';
                if (deleteBtn) {
                    deleteBtn.style.display = 'block';
                    
                    // Re-attach event listener to ensure it works every time
                    // First remove any existing listeners to avoid duplication
                    deleteBtn.removeEventListener('click', deleteTriggerWord);
                    deleteBtn.addEventListener('click', deleteTriggerWord);
                }
            });
            
            // Load trained words and display dropdown when entering edit mode
            // Add loading indicator
            const loadingIndicator = document.createElement('div');
            loadingIndicator.className = 'metadata-loading';
            loadingIndicator.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading suggestions...';
            addForm.appendChild(loadingIndicator);
            
            // Get currently added trigger words
            const currentTags = triggerWordsSection.querySelectorAll('.trigger-word-tag');
            const existingWords = Array.from(currentTags).map(tag => tag.dataset.word);
            
            // Asynchronously load trained words if not already loaded
            if (!isTrainedWordsLoaded) {
                const result = await fetchTrainedWords(filePath);
                trainedWordsList = result.trainedWords;
                classTokensValue = result.classTokens;
                isTrainedWordsLoaded = true;
            }
            
            // Remove loading indicator
            loadingIndicator.remove();
            
            // Create and display suggestion dropdown
            const dropdown = createSuggestionDropdown(trainedWordsList, classTokensValue, existingWords);
            addForm.appendChild(dropdown);
            
            // Focus the input
            addForm.querySelector('input').focus();
            
        } else {
            this.innerHTML = '<i class="fas fa-pencil-alt"></i>'; // Change back to edit icon
            this.title = "Edit trigger words";
            
            // Hide edit controls and input form
            editControls.style.display = 'none';
            addForm.style.display = 'none';
            
            // Check if we're exiting edit mode due to "Save" or "Cancel"
            if (!this.dataset.skipRestore) {
                // If canceling, restore original trigger words
                restoreOriginalTriggerWords(triggerWordsSection, originalTriggerWords);
            } else {
                // If saving, reset UI state on current trigger words
                resetTriggerWordsUIState(triggerWordsSection);
                // Reset the skip restore flag
                delete this.dataset.skipRestore;
            }
            
            // If we have no trigger words, show the "No trigger word needed" text
            // and hide the empty tags container
            const currentTags = triggerWordsSection.querySelectorAll('.trigger-word-tag');
            if (noTriggerWords && currentTags.length === 0) {
                noTriggerWords.style.display = '';
                if (tagsContainer) tagsContainer.style.display = 'none';
            }
            
            // Remove dropdown if present
            const dropdown = triggerWordsSection.querySelector('.metadata-suggestions-dropdown');
            if (dropdown) dropdown.remove();
        }
    });
    
    // Set up input for adding trigger words
    const triggerWordInput = document.querySelector('.metadata-input');
    
    if (triggerWordInput) {
        // Add keydown event to input
        triggerWordInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                addNewTriggerWord(this.value);
                this.value = ''; // Clear input after adding
            }
        });
    }
    
    // Set up save button
    const saveBtn = document.querySelector('.metadata-save-btn');
    if (saveBtn) {
        saveBtn.addEventListener('click', saveTriggerWords);
    }
    
    // Set up delete buttons
    document.querySelectorAll('.metadata-delete-btn').forEach(btn => {
        // Remove any existing listeners to avoid duplication
        btn.removeEventListener('click', deleteTriggerWord);
        btn.addEventListener('click', deleteTriggerWord);
    });
}

/**
 * Delete trigger word event handler
 * @param {Event} e - Click event
 */
function deleteTriggerWord(e) {
    e.stopPropagation();
    const tag = this.closest('.trigger-word-tag');
    tag.remove();
    
    // Update status of items in the trained words dropdown
    updateTrainedWordsDropdown();
}

/**
 * Reset UI state for trigger words after saving
 * @param {HTMLElement} section - The trigger words section
 */
function resetTriggerWordsUIState(section) {
    const triggerWordTags = section.querySelectorAll('.trigger-word-tag');
    
    triggerWordTags.forEach(tag => {
        const word = tag.dataset.word;
        const copyIcon = tag.querySelector('.trigger-word-copy');
        const deleteBtn = tag.querySelector('.metadata-delete-btn');
        
        // Restore click-to-copy functionality
        tag.onclick = () => copyTriggerWord(word);
        
        // Show copy icon, hide delete button
        if (copyIcon) copyIcon.style.display = '';
        if (deleteBtn) deleteBtn.style.display = 'none';
    });
}

/**
 * Restore original trigger words when canceling edit
 * @param {HTMLElement} section - The trigger words section
 * @param {Array} originalWords - Original trigger words
 */
function restoreOriginalTriggerWords(section, originalWords) {
    const tagsContainer = section.querySelector('.trigger-words-tags');
    const noTriggerWords = section.querySelector('.no-trigger-words');
    
    if (!tagsContainer) return;
    
    // Clear current tags
    tagsContainer.innerHTML = '';
    
    if (originalWords.length === 0) {
        if (noTriggerWords) noTriggerWords.style.display = '';
        tagsContainer.style.display = 'none';
        return;
    }
    
    // Hide "no trigger words" message
    if (noTriggerWords) noTriggerWords.style.display = 'none';
    tagsContainer.style.display = 'flex';
    
    // Recreate original tags
    originalWords.forEach(word => {
        const tag = document.createElement('div');
        tag.className = 'trigger-word-tag';
        tag.dataset.word = word;
        tag.onclick = () => copyTriggerWord(word);
        tag.innerHTML = `
            <span class="trigger-word-content">${word}</span>
            <span class="trigger-word-copy">
                <i class="fas fa-copy"></i>
            </span>
            <button class="metadata-delete-btn" style="display:none;" onclick="event.stopPropagation();">
                <i class="fas fa-times"></i>
            </button>
        `;
        tagsContainer.appendChild(tag);
    });
}

/**
 * Add a new trigger word
 * @param {string} word - Trigger word to add
 */
function addNewTriggerWord(word) {
    word = word.trim();
    if (!word) return;
    
    const triggerWordsSection = document.querySelector('.trigger-words');
    let tagsContainer = document.querySelector('.trigger-words-tags');
    
    // Ensure tags container exists and is visible
    if (tagsContainer) {
        tagsContainer.style.display = 'flex';
    } else {
        // Create tags container if it doesn't exist
        const contentDiv = triggerWordsSection.querySelector('.trigger-words-content');
        if (contentDiv) {
            tagsContainer = document.createElement('div');
            tagsContainer.className = 'trigger-words-tags';
            contentDiv.appendChild(tagsContainer);
        }
    }
    
    if (!tagsContainer) return;
    
    // Hide "no trigger words" message if it exists
    const noTriggerWordsMsg = triggerWordsSection.querySelector('.no-trigger-words');
    if (noTriggerWordsMsg) {
        noTriggerWordsMsg.style.display = 'none';
    }
    
    // Validation: Check length
    if (word.split(/\s+/).length > 30) {
        showToast('Trigger word should not exceed 30 words', 'error');
        return;
    }
    
    // Validation: Check total number
    const currentTags = tagsContainer.querySelectorAll('.trigger-word-tag');
    if (currentTags.length >= 30) {
        showToast('Maximum 30 trigger words allowed', 'error');
        return;
    }
    
    // Validation: Check for duplicates
    const existingWords = Array.from(currentTags).map(tag => tag.dataset.word);
    if (existingWords.includes(word)) {
        showToast('This trigger word already exists', 'error');
        return;
    }
    
    // Create new tag
    const newTag = document.createElement('div');
    newTag.className = 'trigger-word-tag';
    newTag.dataset.word = word;
    newTag.innerHTML = `
        <span class="trigger-word-content">${word}</span>
        <span class="trigger-word-copy" style="display:none;">
            <i class="fas fa-copy"></i>
        </span>
        <button class="metadata-delete-btn" onclick="event.stopPropagation();">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add event listener to delete button
    const deleteBtn = newTag.querySelector('.metadata-delete-btn');
    deleteBtn.addEventListener('click', deleteTriggerWord);
    
    tagsContainer.appendChild(newTag);
    
    // Update status of items in the trained words dropdown
    updateTrainedWordsDropdown();
}

/**
 * Update status of items in the trained words dropdown
 */
function updateTrainedWordsDropdown() {
    const dropdown = document.querySelector('.metadata-suggestions-dropdown');
    if (!dropdown) return;
    
    // Get all current trigger words
    const currentTags = document.querySelectorAll('.trigger-word-tag');
    const existingWords = Array.from(currentTags).map(tag => tag.dataset.word);
    
    // Update status of each item in dropdown
    dropdown.querySelectorAll('.metadata-suggestion-item').forEach(item => {
        const wordText = item.querySelector('.metadata-suggestion-text').textContent;
        const isAdded = existingWords.includes(wordText);
        
        if (isAdded) {
            item.classList.add('already-added');
            
            // Add indicator if it doesn't exist
            let indicator = item.querySelector('.added-indicator');
            if (!indicator) {
                const meta = item.querySelector('.metadata-suggestion-meta');
                indicator = document.createElement('span');
                indicator.className = 'added-indicator';
                indicator.innerHTML = '<i class="fas fa-check"></i>';
                meta.appendChild(indicator);
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
                    const word = item.querySelector('.metadata-suggestion-text').textContent;
                    addNewTriggerWord(word);
                    
                    // Also populate the input field
                    const input = document.querySelector('.metadata-input');
                    if (input) input.value = word;
                    
                    // Focus the input
                    if (input) input.focus();
                };
            }
        }
    });
}

/**
 * Save trigger words
 */
async function saveTriggerWords() {
    const editBtn = document.querySelector('.edit-trigger-words-btn');
    const filePath = editBtn.dataset.filePath;
    const triggerWordsSection = editBtn.closest('.trigger-words');
    const triggerWordTags = triggerWordsSection.querySelectorAll('.trigger-word-tag');
    const words = Array.from(triggerWordTags).map(tag => tag.dataset.word);
    
    try {
        // Special format for updating nested civitai.trainedWords
        await saveModelMetadata(filePath, {
            civitai: { trainedWords: words }
        });
        
        // Set flag to skip restoring original words when exiting edit mode
        editBtn.dataset.skipRestore = "true";
        
        // Exit edit mode without restoring original trigger words
        editBtn.click();
        
        // If we saved an empty array and there's a no-trigger-words element, show it
        const noTriggerWords = triggerWordsSection.querySelector('.no-trigger-words');
        const tagsContainer = triggerWordsSection.querySelector('.trigger-words-tags');
        if (words.length === 0 && noTriggerWords) {
            noTriggerWords.style.display = '';
            if (tagsContainer) tagsContainer.style.display = 'none';
        }
        
        showToast('Trigger words updated successfully', 'success');
    } catch (error) {
        console.error('Error saving trigger words:', error);
        showToast('Failed to update trigger words', 'error');
    }
}

/**
 * Copy a trigger word to clipboard
 * @param {string} word - Word to copy
 */
window.copyTriggerWord = async function(word) {
    try {
        await copyToClipboard(word, 'Trigger word copied');
    } catch (err) {
        console.error('Copy failed:', err);
        showToast('Copy failed', 'error');
    }
};