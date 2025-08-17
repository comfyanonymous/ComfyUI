import { showToast } from '../../utils/uiHelpers.js';

export class RecipeDataManager {
    constructor(importManager) {
        this.importManager = importManager;
    }

    showRecipeDetailsStep() {
        this.importManager.stepManager.showStep('detailsStep');
        
        // Set default recipe name from prompt or image filename
        const recipeName = document.getElementById('recipeName');
        
        // Check if we have recipe metadata from a shared recipe
        if (this.importManager.recipeData && this.importManager.recipeData.from_recipe_metadata) {
            // Use title from recipe metadata
            if (this.importManager.recipeData.title) {
                recipeName.value = this.importManager.recipeData.title;
                this.importManager.recipeName = this.importManager.recipeData.title;
            }
            
            // Use tags from recipe metadata
            if (this.importManager.recipeData.tags && Array.isArray(this.importManager.recipeData.tags)) {
                this.importManager.recipeTags = [...this.importManager.recipeData.tags];
                this.updateTagsDisplay();
            }
        } else if (this.importManager.recipeData && 
                  this.importManager.recipeData.gen_params && 
                  this.importManager.recipeData.gen_params.prompt) {
            // Use the first 10 words from the prompt as the default recipe name
            const promptWords = this.importManager.recipeData.gen_params.prompt.split(' ');
            const truncatedPrompt = promptWords.slice(0, 10).join(' ');
            recipeName.value = truncatedPrompt;
            this.importManager.recipeName = truncatedPrompt;
            
            // Set up click handler to select all text for easy editing
            if (!recipeName.hasSelectAllHandler) {
                recipeName.addEventListener('click', function() {
                    this.select();
                });
                recipeName.hasSelectAllHandler = true;
            }
        } else if (this.importManager.recipeImage && !recipeName.value) {
            // Fallback to image filename if no prompt is available
            const fileName = this.importManager.recipeImage.name.split('.')[0];
            recipeName.value = fileName;
            this.importManager.recipeName = fileName;
        }
        
        // Always set up click handler for easy editing if not already set
        if (!recipeName.hasSelectAllHandler) {
            recipeName.addEventListener('click', function() {
                this.select();
            });
            recipeName.hasSelectAllHandler = true;
        }
        
        // Display the uploaded image in the preview
        const imagePreview = document.getElementById('recipeImagePreview');
        if (imagePreview) {
            if (this.importManager.recipeImage) {
                // For file upload mode
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.innerHTML = `<img src="${e.target.result}" alt="Recipe preview">`;
                };
                reader.readAsDataURL(this.importManager.recipeImage);
            } else if (this.importManager.recipeData && this.importManager.recipeData.image_base64) {
                // For URL mode - use the base64 image data returned from the backend
                imagePreview.innerHTML = `<img src="data:image/jpeg;base64,${this.importManager.recipeData.image_base64}" alt="Recipe preview">`;
            } else if (this.importManager.importMode === 'url') {
                // Fallback for URL mode if no base64 data
                const urlInput = document.getElementById('imageUrlInput');
                if (urlInput && urlInput.value) {
                    imagePreview.innerHTML = `<img src="${urlInput.value}" alt="Recipe preview" crossorigin="anonymous">`;
                }
            }
        }

        // Update LoRA count information
        const totalLoras = this.importManager.recipeData.loras.length;
        const existingLoras = this.importManager.recipeData.loras.filter(lora => lora.existsLocally).length;
        const loraCountInfo = document.getElementById('loraCountInfo');
        if (loraCountInfo) {
            loraCountInfo.textContent = `(${existingLoras}/${totalLoras} in library)`;
        }
        
        // Display LoRAs list
        const lorasList = document.getElementById('lorasList');
        if (lorasList) {
            lorasList.innerHTML = this.importManager.recipeData.loras.map(lora => {
                const existsLocally = lora.existsLocally;
                const isDeleted = lora.isDeleted;
                const isEarlyAccess = lora.isEarlyAccess;
                const localPath = lora.localPath || '';
                
                // Create status badge based on LoRA status
                let statusBadge;
                if (isDeleted) {
                    statusBadge = `<div class="deleted-badge">
                        <i class="fas fa-exclamation-circle"></i> Deleted from Civitai
                    </div>`;
                } else {
                    statusBadge = existsLocally ? 
                        `<div class="local-badge">
                            <i class="fas fa-check"></i> In Library
                            <div class="local-path">${localPath}</div>
                        </div>` :
                        `<div class="missing-badge">
                            <i class="fas fa-exclamation-triangle"></i> Not in Library
                        </div>`;
                }

                // Early access badge (shown additionally with other badges)
                let earlyAccessBadge = '';
                if (isEarlyAccess) {
                    // Format the early access end date if available
                    let earlyAccessInfo = 'This LoRA requires early access payment to download.';
                    if (lora.earlyAccessEndsAt) {
                        try {
                            const endDate = new Date(lora.earlyAccessEndsAt);
                            const formattedDate = endDate.toLocaleDateString();
                            earlyAccessInfo += ` Early access ends on ${formattedDate}.`;
                        } catch (e) {
                            console.warn('Failed to format early access date', e);
                        }
                    }
                    
                    earlyAccessBadge = `<div class="early-access-badge">
                        <i class="fas fa-clock"></i> Early Access
                        <div class="early-access-info">${earlyAccessInfo} Verify that you have purchased early access before downloading.</div>
                    </div>`;
                }

                // Format size if available
                const sizeDisplay = lora.size ? 
                    `<div class="size-badge">${this.importManager.formatFileSize(lora.size)}</div>` : '';

                return `
                    <div class="lora-item ${existsLocally ? 'exists-locally' : isDeleted ? 'is-deleted' : 'missing-locally'} ${isEarlyAccess ? 'is-early-access' : ''}">
                        <div class="lora-thumbnail">
                            <img src="${lora.thumbnailUrl || '/loras_static/images/no-preview.png'}" alt="LoRA preview">
                        </div>
                        <div class="lora-content">
                            <div class="lora-header">
                                <h3>${lora.name}</h3>
                                <div class="badge-container">
                                    ${statusBadge}
                                    ${earlyAccessBadge}
                                </div>
                            </div>
                            ${lora.version ? `<div class="lora-version">${lora.version}</div>` : ''}
                            <div class="lora-info">
                                ${lora.baseModel ? `<div class="base-model">${lora.baseModel}</div>` : ''}
                                ${sizeDisplay}
                                <div class="weight-badge">Weight: ${lora.weight || 1.0}</div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        // Check for early access loras and show warning if any exist
        const earlyAccessLoras = this.importManager.recipeData.loras.filter(lora => 
            lora.isEarlyAccess && !lora.existsLocally && !lora.isDeleted);
        if (earlyAccessLoras.length > 0) {
            // Show a warning about early access loras
            const warningMessage = `
                <div class="early-access-warning">
                    <div class="warning-icon"><i class="fas fa-clock"></i></div>
                    <div class="warning-content">
                        <div class="warning-title">${earlyAccessLoras.length} LoRA(s) require Early Access</div>
                        <div class="warning-text">
                            These LoRAs require a payment to access. Download will fail if you haven't purchased access.
                            You may need to log in to your Civitai account in browser settings.
                        </div>
                    </div>
                </div>
            `;
            
            // Show the warning message
            const buttonsContainer = document.querySelector('#detailsStep .modal-actions');
            if (buttonsContainer) {
                // Remove existing warning if any
                const existingWarning = document.getElementById('earlyAccessWarning');
                if (existingWarning) {
                    existingWarning.remove();
                }
                
                // Add new warning
                const warningContainer = document.createElement('div');
                warningContainer.id = 'earlyAccessWarning';
                warningContainer.innerHTML = warningMessage;
                buttonsContainer.parentNode.insertBefore(warningContainer, buttonsContainer);
            }
        }
        
        // Check for duplicate recipes and display warning if found
        this.checkAndDisplayDuplicates();
        
        // Update Next button state based on missing LoRAs and duplicates
        this.updateNextButtonState();
    }
    
    checkAndDisplayDuplicates() {
        // Check if we have duplicate recipes
        if (this.importManager.recipeData && 
            this.importManager.recipeData.matching_recipes && 
            this.importManager.recipeData.matching_recipes.length > 0) {
            
            // Store duplicates in the importManager for later use
            this.importManager.duplicateRecipes = this.importManager.recipeData.matching_recipes;
            
            // Create duplicate warning container
            const duplicateContainer = document.getElementById('duplicateRecipesContainer') || 
                this.createDuplicateContainer();
                
            // Format date helper function
            const formatDate = (timestamp) => {
                try {
                    const date = new Date(timestamp * 1000);
                    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
                } catch (e) {
                    return 'Unknown date';
                }
            };
            
            // Generate the HTML for duplicate recipes warning
            duplicateContainer.innerHTML = `
                <div class="duplicate-warning">
                    <div class="warning-icon"><i class="fas fa-clone"></i></div>
                    <div class="warning-content">
                        <div class="warning-title">
                            ${this.importManager.duplicateRecipes.length} identical ${this.importManager.duplicateRecipes.length === 1 ? 'recipe' : 'recipes'} found in your library
                        </div>
                        <div class="warning-text">
                            These recipes contain the same LoRAs with identical weights.
                            <button id="toggleDuplicatesList" class="toggle-duplicates-btn">
                                Show duplicates <i class="fas fa-chevron-down"></i>
                            </button>
                        </div>
                    </div>
                </div>
                <div class="duplicate-recipes-list collapsed">
                    ${this.importManager.duplicateRecipes.map((recipe) => `
                        <div class="duplicate-recipe-card">
                            <div class="duplicate-recipe-preview">
                                <img src="${recipe.file_url}" alt="Recipe preview">
                                <div class="duplicate-recipe-title">${recipe.title}</div>
                            </div>
                            <div class="duplicate-recipe-details">
                                <div class="duplicate-recipe-date">
                                    <i class="fas fa-calendar-alt"></i> ${formatDate(recipe.modified)}
                                </div>
                                <div class="duplicate-recipe-lora-count">
                                    <i class="fas fa-layer-group"></i> ${recipe.lora_count} LoRAs
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
            
            // Show the duplicate container
            duplicateContainer.style.display = 'block';
            
            // Add click event for the toggle button
            const toggleButton = document.getElementById('toggleDuplicatesList');
            if (toggleButton) {
                toggleButton.addEventListener('click', () => {
                    const list = duplicateContainer.querySelector('.duplicate-recipes-list');
                    if (list) {
                        list.classList.toggle('collapsed');
                        const icon = toggleButton.querySelector('i');
                        if (icon) {
                            if (list.classList.contains('collapsed')) {
                                toggleButton.innerHTML = `Show duplicates <i class="fas fa-chevron-down"></i>`;
                            } else {
                                toggleButton.innerHTML = `Hide duplicates <i class="fas fa-chevron-up"></i>`;
                            }
                        }
                    }
                });
            }
        } else {
            // No duplicates, hide the container if it exists
            const duplicateContainer = document.getElementById('duplicateRecipesContainer');
            if (duplicateContainer) {
                duplicateContainer.style.display = 'none';
            }
            
            // Reset duplicate tracking
            this.importManager.duplicateRecipes = [];
        }
    }
    
    createDuplicateContainer() {
        // Find where to insert the duplicate container
        const lorasListContainer = document.querySelector('.input-group:has(#lorasList)');
        
        if (!lorasListContainer) return null;
        
        // Create container
        const duplicateContainer = document.createElement('div');
        duplicateContainer.id = 'duplicateRecipesContainer';
        duplicateContainer.className = 'duplicate-recipes-container';
        
        // Insert before the LoRA list
        lorasListContainer.parentNode.insertBefore(duplicateContainer, lorasListContainer);
        
        return duplicateContainer;
    }
    
    updateNextButtonState() {
        const nextButton = document.querySelector('#detailsStep .primary-btn');
        const actionsContainer = document.querySelector('#detailsStep .modal-actions');
        if (!nextButton || !actionsContainer) return;
        
        // Always clean up previous warnings and buttons first
        const existingWarning = document.getElementById('deletedLorasWarning');
        if (existingWarning) {
            existingWarning.remove();
        }
        
        // Remove any existing "import anyway" button
        const importAnywayBtn = document.getElementById('importAnywayBtn');
        if (importAnywayBtn) {
            importAnywayBtn.remove();
        }
        
        // Count deleted LoRAs
        const deletedLoras = this.importManager.recipeData.loras.filter(lora => lora.isDeleted).length;
        
        // If we have deleted LoRAs, show a warning
        if (deletedLoras > 0) {
            // Create a new warning container above the buttons
            const buttonsContainer = document.querySelector('#detailsStep .modal-actions') || nextButton.parentNode;
            const warningContainer = document.createElement('div');
            warningContainer.id = 'deletedLorasWarning';
            warningContainer.className = 'deleted-loras-warning';
            
            // Create warning message
            warningContainer.innerHTML = `
                <div class="warning-icon"><i class="fas fa-exclamation-triangle"></i></div>
                <div class="warning-content">
                    <div class="warning-title">${deletedLoras} LoRA(s) have been deleted from Civitai</div>
                    <div class="warning-text">These LoRAs cannot be downloaded. If you continue, they will remain in the recipe but won't be included when used.</div>
                </div>
            `;
            
            // Insert before the buttons container
            buttonsContainer.parentNode.insertBefore(warningContainer, buttonsContainer);
        }
        
        // Check for duplicates but don't change button actions
        const missingNotDeleted = this.importManager.recipeData.loras.filter(
            lora => !lora.existsLocally && !lora.isDeleted
        ).length;
        
        // Standard button behavior regardless of duplicates
        nextButton.classList.remove('warning-btn');
        
        if (missingNotDeleted > 0) {
            nextButton.textContent = 'Download Missing LoRAs';
        } else {
            nextButton.textContent = 'Save Recipe';
        }
    }

    addTag() {
        const tagInput = document.getElementById('tagInput');
        const tag = tagInput.value.trim();
        
        if (!tag) return;
        
        if (!this.importManager.recipeTags.includes(tag)) {
            this.importManager.recipeTags.push(tag);
            this.updateTagsDisplay();
        }
        
        tagInput.value = '';
    }
    
    removeTag(tag) {
        this.importManager.recipeTags = this.importManager.recipeTags.filter(t => t !== tag);
        this.updateTagsDisplay();
    }
    
    updateTagsDisplay() {
        const tagsContainer = document.getElementById('tagsContainer');
        
        if (this.importManager.recipeTags.length === 0) {
            tagsContainer.innerHTML = '<div class="empty-tags">No tags added</div>';
            return;
        }
        
        tagsContainer.innerHTML = this.importManager.recipeTags.map(tag => `
            <div class="recipe-tag">
                ${tag}
                <i class="fas fa-times" onclick="importManager.removeTag('${tag}')"></i>
            </div>
        `).join('');
    }

    proceedFromDetails() {
        // Validate recipe name
        if (!this.importManager.recipeName) {
            showToast('Please enter a recipe name', 'error');
            return;
        }
        
        // Automatically mark all deleted LoRAs as excluded
        if (this.importManager.recipeData && this.importManager.recipeData.loras) {
            this.importManager.recipeData.loras.forEach(lora => {
                if (lora.isDeleted) {
                    lora.exclude = true;
                }
            });
        }
        
        // Update missing LoRAs list to exclude deleted LoRAs
        this.importManager.missingLoras = this.importManager.recipeData.loras.filter(lora => 
            !lora.existsLocally && !lora.isDeleted);
        
        // If we have downloadable missing LoRAs, go to location step
        if (this.importManager.missingLoras.length > 0) {
            // Store only downloadable LoRAs for the download step
            this.importManager.downloadableLoRAs = this.importManager.missingLoras;
            this.importManager.proceedToLocation();
        } else {
            // Otherwise, save the recipe directly
            this.importManager.saveRecipe();
        }
    }
}
