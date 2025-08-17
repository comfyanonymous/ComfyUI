import { showToast } from '../../utils/uiHelpers.js';

/**
 * ModelDescription.js
 * Handles model description related functionality - General version
 */

/**
 * Set up tab switching functionality
 */
export function setupTabSwitching() {
    const tabButtons = document.querySelectorAll('.showcase-tabs .tab-btn');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all tabs
            document.querySelectorAll('.showcase-tabs .tab-btn').forEach(btn => 
                btn.classList.remove('active')
            );
            document.querySelectorAll('.tab-content .tab-pane').forEach(tab => 
                tab.classList.remove('active')
            );
            
            // Add active class to clicked tab
            button.classList.add('active');
            const tabId = `${button.dataset.tab}-tab`;
            document.getElementById(tabId).classList.add('active');
            
            // If switching to description tab, make sure content is properly sized
            if (button.dataset.tab === 'description') {
                const descriptionContent = document.querySelector('.model-description-content');
                if (descriptionContent) {
                    const hasContent = descriptionContent.innerHTML.trim() !== '';
                    document.querySelector('.model-description-loading')?.classList.add('hidden');
                    
                    // If no content, show a message
                    if (!hasContent) {
                        descriptionContent.innerHTML = '<div class="no-description">No model description available</div>';
                        descriptionContent.classList.remove('hidden');
                    }
                }
            }
        });
    });
}

/**
 * Set up model description editing functionality
 * @param {string} filePath - File path
 */
export function setupModelDescriptionEditing(filePath) {
    const descContent = document.querySelector('.model-description-content');
    const descContainer = document.querySelector('.model-description-container');
    if (!descContent || !descContainer) return;

    // Add edit button if not present
    let editBtn = descContainer.querySelector('.edit-model-description-btn');
    if (!editBtn) {
        editBtn = document.createElement('button');
        editBtn.className = 'edit-model-description-btn';
        editBtn.title = 'Edit model description';
        editBtn.innerHTML = '<i class="fas fa-pencil-alt"></i>';
        descContainer.insertBefore(editBtn, descContent);
    }

    // Show edit button on hover
    descContainer.addEventListener('mouseenter', () => {
        editBtn.classList.add('visible');
    });
    descContainer.addEventListener('mouseleave', () => {
        if (!descContainer.classList.contains('editing')) {
            editBtn.classList.remove('visible');
        }
    });

    // Handle edit button click
    editBtn.addEventListener('click', () => {
        descContainer.classList.add('editing');
        descContent.setAttribute('contenteditable', 'true');
        descContent.dataset.originalValue = descContent.innerHTML.trim();
        descContent.focus();

        // Place cursor at the end
        const range = document.createRange();
        const sel = window.getSelection();
        range.selectNodeContents(descContent);
        range.collapse(false);
        sel.removeAllRanges();
        sel.addRange(range);

        editBtn.classList.add('visible');
    });

    // Keyboard events
    descContent.addEventListener('keydown', function(e) {
        if (!this.getAttribute('contenteditable')) return;
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.blur();
        } else if (e.key === 'Escape') {
            e.preventDefault();
            this.innerHTML = this.dataset.originalValue;
            exitEditMode();
        }
    });

    // Save on blur
    descContent.addEventListener('blur', async function() {
        if (!this.getAttribute('contenteditable')) return;
        const newValue = this.innerHTML.trim();
        const originalValue = this.dataset.originalValue;
        if (newValue === originalValue) {
            exitEditMode();
            return;
        }
        if (!newValue) {
            this.innerHTML = originalValue;
            showToast('Description cannot be empty', 'error');
            exitEditMode();
            return;
        }
        try {
            // Save to backend
            const { getModelApiClient } = await import('../../api/modelApiFactory.js');
            await getModelApiClient().saveModelMetadata(filePath, { modelDescription: newValue });
            showToast('Model description updated', 'success');
        } catch (err) {
            this.innerHTML = originalValue;
            showToast('Failed to update model description', 'error');
        } finally {
            exitEditMode();
        }
    });

    function exitEditMode() {
        descContent.removeAttribute('contenteditable');
        descContainer.classList.remove('editing');
        editBtn.classList.remove('visible');
    }
}