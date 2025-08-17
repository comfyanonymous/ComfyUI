// Recipe Card Component
import { showToast, copyToClipboard, sendLoraToWorkflow } from '../utils/uiHelpers.js';
import { modalManager } from '../managers/ModalManager.js';
import { getCurrentPageState } from '../state/index.js';
import { state } from '../state/index.js';
import { NSFW_LEVELS } from '../utils/constants.js';

class RecipeCard {
    constructor(recipe, clickHandler) {
        this.recipe = recipe;
        this.clickHandler = clickHandler;
        this.element = this.createCardElement();
        
        // Store reference to this instance on the DOM element for updates
        this.element._recipeCardInstance = this;
    }
    
    createCardElement() {
        const card = document.createElement('div');
        card.className = 'model-card';
        card.dataset.filepath = this.recipe.file_path;
        card.dataset.title = this.recipe.title;
        card.dataset.nsfwLevel = this.recipe.preview_nsfw_level || 0;
        card.dataset.created = this.recipe.created_date;
        card.dataset.id = this.recipe.id || '';
        
        // Get base model
        const baseModel = this.recipe.base_model || '';
        
        // Ensure loras array exists
        const loras = this.recipe.loras || [];
        const lorasCount = loras.length;
        
        // Check if all LoRAs are available in the library
        const missingLorasCount = loras.filter(lora => !lora.inLibrary && !lora.isDeleted).length;
        const allLorasAvailable = missingLorasCount === 0 && lorasCount > 0;
        
        // Ensure file_url exists, fallback to file_path if needed
        const imageUrl = this.recipe.file_url || 
                         (this.recipe.file_path ? `/loras_static/root1/preview/${this.recipe.file_path.split('/').pop()}` : 
                         '/loras_static/images/no-preview.png');

        // Check if in duplicates mode
        const pageState = getCurrentPageState();
        const isDuplicatesMode = pageState.duplicatesMode;

        // NSFW blur logic - similar to LoraCard
        const nsfwLevel = this.recipe.preview_nsfw_level !== undefined ? this.recipe.preview_nsfw_level : 0;
        const shouldBlur = state.settings.blurMatureContent && nsfwLevel > NSFW_LEVELS.PG13;
        
        if (shouldBlur) {
            card.classList.add('nsfw-content');
        }

        // Determine NSFW warning text based on level
        let nsfwText = "Mature Content";
        if (nsfwLevel >= NSFW_LEVELS.XXX) {
            nsfwText = "XXX-rated Content";
        } else if (nsfwLevel >= NSFW_LEVELS.X) {
            nsfwText = "X-rated Content";
        } else if (nsfwLevel >= NSFW_LEVELS.R) {
            nsfwText = "R-rated Content";
        }

        card.innerHTML = `
            <div class="card-preview ${shouldBlur ? 'blurred' : ''}">
                <img src="${imageUrl}" alt="${this.recipe.title}">
                ${!isDuplicatesMode ? `
                <div class="card-header">
                    ${shouldBlur ? 
                      `<button class="toggle-blur-btn" title="Toggle blur">
                          <i class="fas fa-eye"></i>
                      </button>` : ''}
                    ${baseModel ? `<span class="base-model-label ${shouldBlur ? 'with-toggle' : ''}" title="${baseModel}">${baseModel}</span>` : ''}
                    <div class="card-actions">
                        <i class="fas fa-share-alt" title="Share Recipe"></i>
                        <i class="fas fa-paper-plane" title="Send Recipe to Workflow (Click: Append, Shift+Click: Replace)"></i>
                        <i class="fas fa-trash" title="Delete Recipe"></i>
                    </div>
                </div>
                ` : ''}
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
                        <span class="model-name">${this.recipe.title}</span>
                    </div>
                    ${!isDuplicatesMode ? `
                    <div class="lora-count ${allLorasAvailable ? 'ready' : (lorasCount > 0 ? 'missing' : '')}" 
                         title="${this.getLoraStatusTitle(lorasCount, missingLorasCount)}">
                        <i class="fas fa-layer-group"></i> ${lorasCount}
                    </div>
                    ` : ''}
                </div>
            </div>
        `;
        
        this.attachEventListeners(card, isDuplicatesMode, shouldBlur);
        return card;
    }
    
    getLoraStatusTitle(totalCount, missingCount) {
        if (totalCount === 0) return "No LoRAs in this recipe";
        if (missingCount === 0) return "All LoRAs available - Ready to use";
        return `${missingCount} of ${totalCount} LoRAs missing`;
    }
    
    attachEventListeners(card, isDuplicatesMode, shouldBlur) {
        // Add blur toggle functionality if content should be blurred
        if (shouldBlur) {
            const toggleBtn = card.querySelector('.toggle-blur-btn');
            const showBtn = card.querySelector('.show-content-btn');
            
            if (toggleBtn) {
                toggleBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.toggleBlurContent(card);
                });
            }
            
            if (showBtn) {
                showBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.showBlurredContent(card);
                });
            }
        }

        // Recipe card click event - only attach if not in duplicates mode
        if (!isDuplicatesMode) {
            card.addEventListener('click', () => {
                this.clickHandler(this.recipe);
            });
            
            // Share button click event - prevent propagation to card
            card.querySelector('.fa-share-alt')?.addEventListener('click', (e) => {
                e.stopPropagation();
                this.shareRecipe();
            });
            
            // Send button click event - prevent propagation to card
            card.querySelector('.fa-paper-plane')?.addEventListener('click', (e) => {
                e.stopPropagation();
                this.sendRecipeToWorkflow(e.shiftKey);
            });
            
            // Delete button click event - prevent propagation to card
            card.querySelector('.fa-trash')?.addEventListener('click', (e) => {
                e.stopPropagation();
                this.showDeleteConfirmation();
            });
        }
    }
    
    toggleBlurContent(card) {
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

    showBlurredContent(card) {
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

    sendRecipeToWorkflow(replaceMode = false) {
        try {
            // Get recipe ID
            const recipeId = this.recipe.id;
            if (!recipeId) {
                showToast('Cannot send recipe: Missing recipe ID', 'error');
                return;
            }

            fetch(`/api/recipe/${recipeId}/syntax`)
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.syntax) {
                        return sendLoraToWorkflow(data.syntax, replaceMode, 'recipe');
                    } else {
                        throw new Error(data.error || 'No syntax returned');
                    }
                })
                .catch(err => {
                    console.error('Failed to send recipe to workflow: ', err);
                    showToast('Failed to send recipe to workflow', 'error');
                });
        } catch (error) {
            console.error('Error sending recipe to workflow:', error);
            showToast('Error sending recipe to workflow', 'error');
        }
    }
    
    showDeleteConfirmation() {
        try {
            // Get recipe ID
            const recipeId = this.recipe.id;
            const filePath = this.recipe.file_path;
            if (!recipeId) {
                showToast('Cannot delete recipe: Missing recipe ID', 'error');
                return;
            }
            
            // Create delete modal content
            const deleteModalContent = `
                <div class="modal-content delete-modal-content">
                    <h2>Delete Recipe</h2>
                    <p class="delete-message">Are you sure you want to delete this recipe?</p>
                    <div class="delete-model-info">
                        <div class="delete-preview">
                            <img src="${this.recipe.file_url || '/loras_static/images/no-preview.png'}" alt="${this.recipe.title}">
                        </div>
                        <div class="delete-info">
                            <h3>${this.recipe.title}</h3>
                            <p>This action cannot be undone.</p>
                        </div>
                    </div>
                    <p class="delete-note">Note: Deleting this recipe will not affect the LoRA files used in it.</p>
                    <div class="modal-actions">
                        <button class="cancel-btn" onclick="closeDeleteModal()">Cancel</button>
                        <button class="delete-btn" onclick="confirmDelete()">Delete</button>
                    </div>
                </div>
            `;
            
            // Show the modal with custom content and setup callbacks
            modalManager.showModal('deleteModal', deleteModalContent, () => {
                // This is the onClose callback
                const deleteModal = document.getElementById('deleteModal');
                const deleteBtn = deleteModal.querySelector('.delete-btn');
                deleteBtn.textContent = 'Delete';
                deleteBtn.disabled = false;
            });
            
            // Set up the delete and cancel buttons with proper event handlers
            const deleteModal = document.getElementById('deleteModal');
            const cancelBtn = deleteModal.querySelector('.cancel-btn');
            const deleteBtn = deleteModal.querySelector('.delete-btn');
            
            // Store recipe ID in the modal for the delete confirmation handler
            deleteModal.dataset.recipeId = recipeId;
            deleteModal.dataset.filePath = filePath;
            
            // Update button event handlers
            cancelBtn.onclick = () => modalManager.closeModal('deleteModal');
            deleteBtn.onclick = () => this.confirmDeleteRecipe();
            
        } catch (error) {
            console.error('Error showing delete confirmation:', error);
            showToast('Error showing delete confirmation', 'error');
        }
    }

    confirmDeleteRecipe() {
        const deleteModal = document.getElementById('deleteModal');
        const recipeId = deleteModal.dataset.recipeId;
        
        if (!recipeId) {
            showToast('Cannot delete recipe: Missing recipe ID', 'error');
            modalManager.closeModal('deleteModal');
            return;
        }
        
        // Show loading state
        const deleteBtn = deleteModal.querySelector('.delete-btn');
        const originalText = deleteBtn.textContent;
        deleteBtn.textContent = 'Deleting...';
        deleteBtn.disabled = true;
        
        // Call API to delete the recipe
        fetch(`/api/recipe/${recipeId}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to delete recipe');
            }
            return response.json();
        })
        .then(data => {
            showToast('Recipe deleted successfully', 'success');
            
            state.virtualScroller.removeItemByFilePath(deleteModal.dataset.filePath);
            
            modalManager.closeModal('deleteModal');
        })
        .catch(error => {
            console.error('Error deleting recipe:', error);
            showToast('Error deleting recipe: ' + error.message, 'error');
            
            // Reset button state
            deleteBtn.textContent = originalText;
            deleteBtn.disabled = false;
        });
    }

    shareRecipe() {
        try {
            // Get recipe ID
            const recipeId = this.recipe.id;
            if (!recipeId) {
                showToast('Cannot share recipe: Missing recipe ID', 'error');
                return;
            }
            
            // Show loading toast
            showToast('Preparing recipe for sharing...', 'info');
            
            // Call the API to process the image with metadata
            fetch(`/api/recipe/${recipeId}/share`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Failed to prepare recipe for sharing');
                    }
                    return response.json();
                })
                .then(data => {
                    if (!data.success) {
                        throw new Error(data.error || 'Unknown error');
                    }
                    
                    // Create a temporary anchor element for download
                    const downloadLink = document.createElement('a');
                    downloadLink.href = data.download_url;
                    downloadLink.download = data.filename;
                    
                    // Append to body, click and remove
                    document.body.appendChild(downloadLink);
                    downloadLink.click();
                    document.body.removeChild(downloadLink);
                    
                    showToast('Recipe download started', 'success');
                })
                .catch(error => {
                    console.error('Error sharing recipe:', error);
                    showToast('Error sharing recipe: ' + error.message, 'error');
                });
        } catch (error) {
            console.error('Error sharing recipe:', error);
            showToast('Error preparing recipe for sharing', 'error');
        }
    }
}

export { RecipeCard };