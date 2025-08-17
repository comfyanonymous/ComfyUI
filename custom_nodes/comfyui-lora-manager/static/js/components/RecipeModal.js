// Recipe Modal Component
import { showToast, copyToClipboard } from '../utils/uiHelpers.js';
import { state } from '../state/index.js';
import { setSessionItem, removeSessionItem } from '../utils/storageHelpers.js';
import { updateRecipeMetadata } from '../api/recipeApi.js';

class RecipeModal {
    constructor() {
        this.init();
    }
    
    init() {
        this.setupCopyButtons();
        // Set up tooltip positioning handlers after DOM is ready
        document.addEventListener('DOMContentLoaded', () => {
            this.setupTooltipPositioning();
        });
        
        // Set up document click handler to close edit fields
        document.addEventListener('click', (event) => {
            // Handle title edit
            const titleEditor = document.getElementById('recipeTitleEditor');
            if (titleEditor && titleEditor.classList.contains('active') && 
                !titleEditor.contains(event.target) && 
                !event.target.closest('.edit-icon')) {
                this.saveTitleEdit();
            }
            
            // Handle tags edit
            const tagsEditor = document.getElementById('recipeTagsEditor');
            if (tagsEditor && tagsEditor.classList.contains('active') && 
                !tagsEditor.contains(event.target) && 
                !event.target.closest('.edit-icon')) {
                this.saveTagsEdit();
            }

            // Handle reconnect input
            const reconnectContainers = document.querySelectorAll('.lora-reconnect-container');
            reconnectContainers.forEach(container => {
                if (container.classList.contains('active') && 
                    !container.contains(event.target) && 
                    !event.target.closest('.deleted-badge.reconnectable')) {
                    this.hideReconnectInput(container);
                }
            });
        });
    }
    
    // Add tooltip positioning handler to ensure correct positioning of fixed tooltips
    setupTooltipPositioning() {
        document.addEventListener('mouseover', (event) => {
            // Check if we're hovering over a local-badge
            if (event.target.closest('.local-badge')) {
                const badge = event.target.closest('.local-badge');
                const tooltip = badge.querySelector('.local-path');
                
                if (tooltip) {
                    // Get badge position
                    const badgeRect = badge.getBoundingClientRect();
                    
                    // Position the tooltip
                    tooltip.style.top = (badgeRect.bottom + 4) + 'px';
                    tooltip.style.left = (badgeRect.right - tooltip.offsetWidth) + 'px';
                }
            }
            
            // Add tooltip positioning for missing badge
            if (event.target.closest('.recipe-status.missing')) {
                const badge = event.target.closest('.recipe-status.missing');
                const tooltip = badge.querySelector('.missing-tooltip');
                
                if (tooltip) {
                    // Get badge position
                    const badgeRect = badge.getBoundingClientRect();
                    
                    // Position the tooltip
                    tooltip.style.top = (badgeRect.bottom + 4) + 'px';
                    tooltip.style.left = (badgeRect.left) + 'px';
                }
            }
        }, true);
    }
    
    showRecipeDetails(recipe) {
        // Store the full recipe for editing
        this.currentRecipe = recipe;
        
        // Set modal title with edit icon
        const modalTitle = document.getElementById('recipeModalTitle');
        if (modalTitle) {
            modalTitle.innerHTML = `
                <div class="editable-content">
                    <span class="content-text">${recipe.title || 'Recipe Details'}</span>
                    <button class="edit-icon" title="Edit recipe name"><i class="fas fa-pencil-alt"></i></button>
                </div>
                <div id="recipeTitleEditor" class="content-editor">
                    <input type="text" class="title-input" value="${recipe.title || ''}">
                </div>
            `;
            
            // Add event listener for title editing
            const editIcon = modalTitle.querySelector('.edit-icon');
            editIcon.addEventListener('click', () => this.showTitleEditor());
            
            // Add key event listener for Enter key
            const titleInput = modalTitle.querySelector('.title-input');
            titleInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.saveTitleEdit();
                } else if (e.key === 'Escape') {
                    e.preventDefault();
                    this.cancelTitleEdit();
                }
            });
        }
        
        // Store the recipe ID for copy syntax API call
        this.recipeId = recipe.id;
        this.filePath = recipe.file_path;
        
        // Set recipe tags if they exist
        const tagsCompactElement = document.getElementById('recipeTagsCompact');
        const tagsTooltipContent = document.getElementById('recipeTagsTooltipContent');
        
        if (tagsCompactElement) {
            // Add tags container with edit functionality
            tagsCompactElement.innerHTML = `
                <div class="editable-content tags-content">
                    <div class="tags-display"></div>
                    <button class="edit-icon" title="Edit tags"><i class="fas fa-pencil-alt"></i></button>
                </div>
                <div id="recipeTagsEditor" class="content-editor tags-editor">
                    <input type="text" class="tags-input" placeholder="Enter tags separated by commas">
                </div>
            `;
            
            const tagsDisplay = tagsCompactElement.querySelector('.tags-display');
            
            if (recipe.tags && recipe.tags.length > 0) {
                // Limit displayed tags to 5, show a "+X more" button if needed
                const maxVisibleTags = 5;
                const visibleTags = recipe.tags.slice(0, maxVisibleTags);
                const remainingTags = recipe.tags.length > maxVisibleTags ? recipe.tags.slice(maxVisibleTags) : [];
                
                // Add visible tags
                visibleTags.forEach(tag => {
                    const tagElement = document.createElement('div');
                    tagElement.className = 'recipe-tag-compact';
                    tagElement.textContent = tag;
                    tagsDisplay.appendChild(tagElement);
                });
                
                // Add "more" button if needed
                if (remainingTags.length > 0) {
                    const moreButton = document.createElement('div');
                    moreButton.className = 'recipe-tag-more';
                    moreButton.textContent = `+${remainingTags.length} more`;
                    tagsDisplay.appendChild(moreButton);
                    
                    // Add tooltip functionality
                    moreButton.addEventListener('mouseenter', () => {
                        document.getElementById('recipeTagsTooltip').classList.add('visible');
                    });
                    
                    moreButton.addEventListener('mouseleave', () => {
                        setTimeout(() => {
                            if (!document.getElementById('recipeTagsTooltip').matches(':hover')) {
                                document.getElementById('recipeTagsTooltip').classList.remove('visible');
                            }
                        }, 300);
                    });
                    
                    document.getElementById('recipeTagsTooltip').addEventListener('mouseleave', () => {
                        document.getElementById('recipeTagsTooltip').classList.remove('visible');
                    });
                    
                    // Add all tags to tooltip
                    if (tagsTooltipContent) {
                        tagsTooltipContent.innerHTML = '';
                        recipe.tags.forEach(tag => {
                            const tooltipTag = document.createElement('div');
                            tooltipTag.className = 'tooltip-tag';
                            tooltipTag.textContent = tag;
                            tagsTooltipContent.appendChild(tooltipTag);
                        });
                    }
                }
            } else {
                tagsDisplay.innerHTML = '<div class="no-tags">No tags</div>';
            }
            
            // Add event listeners for tags editing
            const editTagsIcon = tagsCompactElement.querySelector('.edit-icon');
            const tagsInput = tagsCompactElement.querySelector('.tags-input');
            
            // Set current tags in the input
            if (recipe.tags && recipe.tags.length > 0) {
                tagsInput.value = recipe.tags.join(', ');
            }
            
            editTagsIcon.addEventListener('click', () => this.showTagsEditor());
            
            // Add key event listener for Enter key
            tagsInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.saveTagsEdit();
                } else if (e.key === 'Escape') {
                    e.preventDefault();
                    this.cancelTagsEdit();
                }
            });
        }
        
        // Set recipe image
        const modalImage = document.getElementById('recipeModalImage');
        if (modalImage) {
            // Ensure file_url exists, fallback to file_path if needed
            const imageUrl = recipe.file_url || 
                            (recipe.file_path ? `/loras_static/root1/preview/${recipe.file_path.split('/').pop()}` : 
                            '/loras_static/images/no-preview.png');
            
            // Check if the file is a video (mp4)
            const isVideo = imageUrl.toLowerCase().endsWith('.mp4');
            
            // Replace the image element with appropriate media element
            const mediaContainer = modalImage.parentElement;
            mediaContainer.innerHTML = '';
            
            if (isVideo) {
                const videoElement = document.createElement('video');
                videoElement.id = 'recipeModalVideo';
                videoElement.src = imageUrl;
                videoElement.controls = true;
                videoElement.autoplay = false;
                videoElement.loop = true;
                videoElement.muted = true;
                videoElement.className = 'recipe-preview-media';
                videoElement.alt = recipe.title || 'Recipe Preview';
                mediaContainer.appendChild(videoElement);
            } else {
                const imgElement = document.createElement('img');
                imgElement.id = 'recipeModalImage';
                imgElement.src = imageUrl;
                imgElement.className = 'recipe-preview-media';
                imgElement.alt = recipe.title || 'Recipe Preview';
                mediaContainer.appendChild(imgElement);
            }

            // Add source URL container if the recipe has a source_path
            const sourceUrlContainer = document.createElement('div');
            sourceUrlContainer.className = 'source-url-container';
            const hasSourceUrl = recipe.source_path && recipe.source_path.trim().length > 0;
            const sourceUrl = hasSourceUrl ? recipe.source_path : '';
            const isValidUrl = hasSourceUrl && (sourceUrl.startsWith('http://') || sourceUrl.startsWith('https://'));
            
            sourceUrlContainer.innerHTML = `
                <div class="source-url-content">
                    <span class="source-url-icon"><i class="fas fa-link"></i></span>
                    <span class="source-url-text" title="${isValidUrl ? 'Click to open source URL' : 'No valid URL'}">${
                        hasSourceUrl ? sourceUrl : 'No source URL'
                    }</span>
                </div>
                <button class="source-url-edit-btn" title="Edit source URL">
                    <i class="fas fa-pencil-alt"></i>
                </button>
            `;
            
            // Add source URL editor
            const sourceUrlEditor = document.createElement('div');
            sourceUrlEditor.className = 'source-url-editor';
            sourceUrlEditor.innerHTML = `
                <input type="text" class="source-url-input" placeholder="Enter source URL (e.g., https://civitai.com/...)" value="${sourceUrl}">
                <div class="source-url-actions">
                    <button class="source-url-cancel-btn">Cancel</button>
                    <button class="source-url-save-btn">Save</button>
                </div>
            `;
            
            // Append both containers to the media container
            mediaContainer.appendChild(sourceUrlContainer);
            mediaContainer.appendChild(sourceUrlEditor);
            
            // Set up event listeners for source URL functionality
            setTimeout(() => {
                this.setupSourceUrlHandlers();
            }, 50);
        }
        
        // Set generation parameters
        const promptElement = document.getElementById('recipePrompt');
        const negativePromptElement = document.getElementById('recipeNegativePrompt');
        const otherParamsElement = document.getElementById('recipeOtherParams');
        
        if (recipe.gen_params) {
            // Set prompt
            if (promptElement && recipe.gen_params.prompt) {
                promptElement.textContent = recipe.gen_params.prompt;
            } else if (promptElement) {
                promptElement.textContent = 'No prompt information available';
            }
            
            // Set negative prompt
            if (negativePromptElement && recipe.gen_params.negative_prompt) {
                negativePromptElement.textContent = recipe.gen_params.negative_prompt;
            } else if (negativePromptElement) {
                negativePromptElement.textContent = 'No negative prompt information available';
            }
            
            // Set other parameters
            if (otherParamsElement) {
                // Clear previous params
                otherParamsElement.innerHTML = '';
                
                // Add all other parameters except prompt and negative_prompt
                const excludedParams = ['prompt', 'negative_prompt'];
                
                for (const [key, value] of Object.entries(recipe.gen_params)) {
                    if (!excludedParams.includes(key) && value !== undefined && value !== null) {
                        const paramTag = document.createElement('div');
                        paramTag.className = 'param-tag';
                        paramTag.innerHTML = `
                            <span class="param-name">${key}:</span>
                            <span class="param-value">${value}</span>
                        `;
                        otherParamsElement.appendChild(paramTag);
                    }
                }
                
                // If no other params, show a message
                if (otherParamsElement.children.length === 0) {
                    otherParamsElement.innerHTML = '<div class="no-params">No additional parameters available</div>';
                }
            }
        } else {
            // No generation parameters available
            if (promptElement) promptElement.textContent = 'No prompt information available';
            if (negativePromptElement) promptElement.textContent = 'No negative prompt information available';
            if (otherParamsElement) otherParamsElement.innerHTML = '<div class="no-params">No parameters available</div>';
        }
        
        // Set LoRAs list and count
        const lorasListElement = document.getElementById('recipeLorasList');
        const lorasCountElement = document.getElementById('recipeLorasCount');
        
        // Check all LoRAs status
        let allLorasAvailable = true;
        let missingLorasCount = 0;
        let deletedLorasCount = 0;
        
        if (recipe.loras && recipe.loras.length > 0) {
            recipe.loras.forEach(lora => {
                if (lora.isDeleted) {
                    deletedLorasCount++;
                } else if (!lora.inLibrary) {
                    allLorasAvailable = false;
                    missingLorasCount++;
                }
            });
        }
        
        // Set LoRAs count and status
        if (lorasCountElement && recipe.loras) {
            const totalCount = recipe.loras.length;
            
            // Create status indicator based on LoRA states
            let statusHTML = '';
            if (totalCount > 0) {
                if (allLorasAvailable && deletedLorasCount === 0) {
                    // All LoRAs are available
                    statusHTML = `<div class="recipe-status ready"><i class="fas fa-check-circle"></i> Ready to use</div>`;
                } else if (missingLorasCount > 0) {
                    // Some LoRAs are missing (prioritize showing missing over deleted)
                    statusHTML = `<div class="recipe-status missing">
                        <i class="fas fa-exclamation-triangle"></i> ${missingLorasCount} missing
                        <div class="missing-tooltip">Click to download missing LoRAs</div>
                    </div>`;
                } else if (deletedLorasCount > 0 && missingLorasCount === 0) {
                    // Some LoRAs are deleted but none are missing
                    statusHTML = `<div class="recipe-status partial"><i class="fas fa-info-circle"></i> ${deletedLorasCount} deleted</div>`;
                }
            }
            
            lorasCountElement.innerHTML = `<i class="fas fa-layer-group"></i> ${totalCount} LoRAs ${statusHTML}`;
            
            // Add event listeners for buttons and status indicators
            setTimeout(() => {
                // Set up click handler for View LoRAs button
                const viewRecipeLorasBtn = document.getElementById('viewRecipeLorasBtn');
                if (viewRecipeLorasBtn) {
                    viewRecipeLorasBtn.addEventListener('click', () => this.navigateToLorasPage());
                }
                
                // Add click handler for missing LoRAs status
                const missingStatus = document.querySelector('.recipe-status.missing');
                if (missingStatus && missingLorasCount > 0) {
                    missingStatus.classList.add('clickable');
                    missingStatus.addEventListener('click', () => this.showDownloadMissingLorasModal());
                }
            }, 100);
        }
        
        if (lorasListElement && recipe.loras && recipe.loras.length > 0) {
            lorasListElement.innerHTML = recipe.loras.map(lora => {
                const existsLocally = lora.inLibrary;
                const isDeleted = lora.isDeleted;
                const localPath = lora.localPath || '';
                
                // Create status badge based on LoRA state
                let localStatus;
                if (existsLocally) {
                    localStatus = `
                        <div class="local-badge">
                            <i class="fas fa-check"></i> In Library
                            <div class="local-path">${localPath}</div>
                        </div>`;
                } else if (isDeleted) {
                    localStatus = `
                        <div class="deleted-badge reconnectable" data-lora-index="${recipe.loras.indexOf(lora)}">
                            <span class="badge-text"><i class="fas fa-trash-alt"></i> Deleted</span>
                            <div class="reconnect-tooltip">Click to reconnect with a local LoRA</div>
                        </div>`;
                } else {
                    localStatus = `
                        <div class="missing-badge">
                            <i class="fas fa-exclamation-triangle"></i> Not in Library
                        </div>`;
                }

                // Check if preview is a video
                const isPreviewVideo = lora.preview_url && lora.preview_url.toLowerCase().endsWith('.mp4');
                const previewMedia = isPreviewVideo ?
                    `<video class="thumbnail-video" autoplay loop muted playsinline>
                        <source src="${lora.preview_url}" type="video/mp4">
                     </video>` :
                    `<img src="${lora.preview_url || '/loras_static/images/no-preview.png'}" alt="LoRA preview">`;

                // Determine CSS class based on LoRA state
                let loraItemClass = 'recipe-lora-item';
                if (existsLocally) {
                    loraItemClass += ' exists-locally';
                } else if (isDeleted) {
                    loraItemClass += ' is-deleted';
                } else {
                    loraItemClass += ' missing-locally';
                }

                return `
                    <div class="${loraItemClass}" data-lora-index="${recipe.loras.indexOf(lora)}">
                        <div class="recipe-lora-thumbnail">
                            ${previewMedia}
                        </div>
                        <div class="recipe-lora-content">
                            <div class="recipe-lora-header">
                                <h4>${lora.modelName}</h4>
                                <div class="badge-container">${localStatus}</div>
                            </div>
                            <div class="recipe-lora-info">
                                ${lora.modelVersionName ? `<div class="recipe-lora-version">${lora.modelVersionName}</div>` : ''}
                                <div class="recipe-lora-weight">Weight: ${lora.strength || 1.0}</div>
                                ${lora.baseModel ? `<div class="base-model">${lora.baseModel}</div>` : ''}
                            </div>
                            <div class="lora-reconnect-container" data-lora-index="${recipe.loras.indexOf(lora)}">
                                <div class="reconnect-instructions">
                                    <p>Enter LoRA Syntax or Name to Reconnect:</p>
                                    <small>Example: <code>&lt;lora:Boris_Vallejo_BV_flux_D:1&gt;</code> or just <code>Boris_Vallejo_BV_flux_D</code></small>
                                </div>
                                <div class="reconnect-form">
                                    <input type="text" class="reconnect-input" placeholder="Enter LoRA name or syntax">
                                    <div class="reconnect-actions">
                                        <button class="reconnect-cancel-btn">Cancel</button>
                                        <button class="reconnect-confirm-btn">Reconnect</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
            
            // Add event listeners for reconnect functionality
            setTimeout(() => {
                this.setupReconnectButtons();
                this.setupLoraItemsClickable();
            }, 100);
            
            // Generate recipe syntax for copy button (this is now a placeholder, actual syntax will be fetched from the API)
            this.recipeLorasSyntax = '';
            
        } else if (lorasListElement) {
            lorasListElement.innerHTML = '<div class="no-loras">No LoRAs associated with this recipe</div>';
            this.recipeLorasSyntax = '';
        }
        
        // Show the modal
        modalManager.showModal('recipeModal');
    }
    
    // Title editing methods
    showTitleEditor() {
        const titleContainer = document.getElementById('recipeModalTitle');
        if (titleContainer) {
            titleContainer.querySelector('.editable-content').classList.add('hide');
            const editor = titleContainer.querySelector('#recipeTitleEditor');
            editor.classList.add('active');
            const input = editor.querySelector('input');
            input.focus();
            input.select();
        }
    }
    
    saveTitleEdit() {
        const titleContainer = document.getElementById('recipeModalTitle');
        if (titleContainer) {
            const editor = titleContainer.querySelector('#recipeTitleEditor');
            const input = editor.querySelector('input');
            const newTitle = input.value.trim();
            
            // Check if title changed
            if (newTitle && newTitle !== this.currentRecipe.title) {
                // Update title in the UI
                titleContainer.querySelector('.content-text').textContent = newTitle;
                
                // Update the recipe on the server
                updateRecipeMetadata(this.filePath, { title: newTitle })
                    .then(data => {
                        // Show success toast
                        showToast('Recipe name updated successfully', 'success');
                        
                        // Update the current recipe object
                        this.currentRecipe.title = newTitle;
                    })
                    .catch(error => {
                        // Error is handled in the API function
                        // Reset the UI if needed
                        titleContainer.querySelector('.content-text').textContent = this.currentRecipe.title || '';
                    });
            }
            
            // Hide editor
            editor.classList.remove('active');
            titleContainer.querySelector('.editable-content').classList.remove('hide');
        }
    }
    
    cancelTitleEdit() {
        const titleContainer = document.getElementById('recipeModalTitle');
        if (titleContainer) {
            // Reset input value
            const editor = titleContainer.querySelector('#recipeTitleEditor');
            const input = editor.querySelector('input');
            input.value = this.currentRecipe.title || '';
            
            // Hide editor
            editor.classList.remove('active');
            titleContainer.querySelector('.editable-content').classList.remove('hide');
        }
    }
    
    // Tags editing methods
    showTagsEditor() {
        const tagsContainer = document.getElementById('recipeTagsCompact');
        if (tagsContainer) {
            tagsContainer.querySelector('.editable-content').classList.add('hide');
            const editor = tagsContainer.querySelector('#recipeTagsEditor');
            editor.classList.add('active');
            const input = editor.querySelector('input');
            input.focus();
        }
    }
    
    saveTagsEdit() {
        const tagsContainer = document.getElementById('recipeTagsCompact');
        if (tagsContainer) {
            const editor = tagsContainer.querySelector('#recipeTagsEditor');
            const input = editor.querySelector('input');
            const tagsText = input.value.trim();
            
            // Parse tags
            let newTags = [];
            if (tagsText) {
                newTags = tagsText.split(',')
                    .map(tag => tag.trim())
                    .filter(tag => tag.length > 0);
            }
            
            // Check if tags changed
            const oldTags = this.currentRecipe.tags || [];
            const tagsChanged = 
                newTags.length !== oldTags.length || 
                newTags.some((tag, index) => tag !== oldTags[index]);
            
            if (tagsChanged) {
                // Update the recipe on the server
                updateRecipeMetadata(this.filePath, { tags: newTags })
                    .then(data => {
                        // Show success toast
                        showToast('Recipe tags updated successfully', 'success');
                        
                        // Update the current recipe object
                        this.currentRecipe.tags = newTags;
                        
                        // Update tags in the UI
                        this.updateTagsDisplay(tagsContainer, newTags);
                    })
                    .catch(error => {
                        // Error is handled in the API function
                    });
            }
            
            // Hide editor
            editor.classList.remove('active');
            tagsContainer.querySelector('.editable-content').classList.remove('hide');
        }
    }
    
    // Helper method to update tags display
    updateTagsDisplay(tagsContainer, tags) {
        const tagsDisplay = tagsContainer.querySelector('.tags-display');
        tagsDisplay.innerHTML = '';
        
        if (tags.length > 0) {
            // Limit displayed tags to 5, show a "+X more" button if needed
            const maxVisibleTags = 5;
            const visibleTags = tags.slice(0, maxVisibleTags);
            const remainingTags = tags.length > maxVisibleTags ? tags.slice(maxVisibleTags) : [];
            
            // Add visible tags
            visibleTags.forEach(tag => {
                const tagElement = document.createElement('div');
                tagElement.className = 'recipe-tag-compact';
                tagElement.textContent = tag;
                tagsDisplay.appendChild(tagElement);
            });
            
            // Add "more" button if needed
            if (remainingTags.length > 0) {
                const moreButton = document.createElement('div');
                moreButton.className = 'recipe-tag-more';
                moreButton.textContent = `+${remainingTags.length} more`;
                tagsDisplay.appendChild(moreButton);
                
                // Update tooltip content
                const tooltipContent = document.getElementById('recipeTagsTooltipContent');
                if (tooltipContent) {
                    tooltipContent.innerHTML = '';
                    tags.forEach(tag => {
                        const tooltipTag = document.createElement('div');
                        tooltipTag.className = 'tooltip-tag';
                        tooltipTag.textContent = tag;
                        tooltipContent.appendChild(tooltipTag);
                    });
                }
                
                // Re-add tooltip functionality
                moreButton.addEventListener('mouseenter', () => {
                    document.getElementById('recipeTagsTooltip').classList.add('visible');
                });
                
                moreButton.addEventListener('mouseleave', () => {
                    setTimeout(() => {
                        if (!document.getElementById('recipeTagsTooltip').matches(':hover')) {
                            document.getElementById('recipeTagsTooltip').classList.remove('visible');
                        }
                    }, 300);
                });
            }
        } else {
            tagsDisplay.innerHTML = '<div class="no-tags">No tags</div>';
        }
    }
    
    cancelTagsEdit() {
        const tagsContainer = document.getElementById('recipeTagsCompact');
        if (tagsContainer) {
            // Reset input value
            const editor = tagsContainer.querySelector('#recipeTagsEditor');
            const input = editor.querySelector('input');
            input.value = this.currentRecipe.tags ? this.currentRecipe.tags.join(', ') : '';
            
            // Hide editor
            editor.classList.remove('active');
            tagsContainer.querySelector('.editable-content').classList.remove('hide');
        }
    }
    
    // Setup source URL handlers
    setupSourceUrlHandlers() {
        const sourceUrlContainer = document.querySelector('.source-url-container');
        const sourceUrlEditor = document.querySelector('.source-url-editor');
        const sourceUrlText = sourceUrlContainer.querySelector('.source-url-text');
        const sourceUrlEditBtn = sourceUrlContainer.querySelector('.source-url-edit-btn');
        const sourceUrlCancelBtn = sourceUrlEditor.querySelector('.source-url-cancel-btn');
        const sourceUrlSaveBtn = sourceUrlEditor.querySelector('.source-url-save-btn');
        const sourceUrlInput = sourceUrlEditor.querySelector('.source-url-input');
        
        // Show editor on edit button click
        sourceUrlEditBtn.addEventListener('click', () => {
            sourceUrlContainer.classList.add('hide');
            sourceUrlEditor.classList.add('active');
            sourceUrlInput.focus();
        });
        
        // Cancel editing
        sourceUrlCancelBtn.addEventListener('click', () => {
            sourceUrlEditor.classList.remove('active');
            sourceUrlContainer.classList.remove('hide');
            sourceUrlInput.value = this.currentRecipe.source_path || '';
        });
        
        // Save new source URL
        sourceUrlSaveBtn.addEventListener('click', () => {
            const newSourceUrl = sourceUrlInput.value.trim();
            if (newSourceUrl !== this.currentRecipe.source_path) {
                // Update the recipe on the server
                updateRecipeMetadata(this.filePath, { source_path: newSourceUrl })
                    .then(data => {
                        // Show success toast
                        showToast('Source URL updated successfully', 'success');
                        
                        // Update source URL in the UI
                        sourceUrlText.textContent = newSourceUrl || 'No source URL';
                        sourceUrlText.title = newSourceUrl && (newSourceUrl.startsWith('http://') || 
                                             newSourceUrl.startsWith('https://')) ? 
                                             'Click to open source URL' : 'No valid URL';
                        
                        // Update the current recipe object
                        this.currentRecipe.source_path = newSourceUrl;
                    })
                    .catch(error => {
                        // Error is handled in the API function
                    });
            }
            
            // Hide editor
            sourceUrlEditor.classList.remove('active');
            sourceUrlContainer.classList.remove('hide');
        });
        
        // Open source URL in a new tab if it's valid
        sourceUrlText.addEventListener('click', () => {
            const url = sourceUrlText.textContent.trim();
            if (url.startsWith('http://') || url.startsWith('https://')) {
                window.open(url, '_blank');
            }
        });
    }
    
    // Setup copy buttons for prompts and recipe syntax
    setupCopyButtons() {
        const copyPromptBtn = document.getElementById('copyPromptBtn');
        const copyNegativePromptBtn = document.getElementById('copyNegativePromptBtn');
        const copyRecipeSyntaxBtn = document.getElementById('copyRecipeSyntaxBtn');
        
        if (copyPromptBtn) {
            copyPromptBtn.addEventListener('click', () => {
                const promptText = document.getElementById('recipePrompt').textContent;
                this.copyToClipboard(promptText, 'Prompt copied to clipboard');
            });
        }
        
        if (copyNegativePromptBtn) {
            copyNegativePromptBtn.addEventListener('click', () => {
                const negativePromptText = document.getElementById('recipeNegativePrompt').textContent;
                this.copyToClipboard(negativePromptText, 'Negative prompt copied to clipboard');
            });
        }
        
        if (copyRecipeSyntaxBtn) {
            copyRecipeSyntaxBtn.addEventListener('click', () => {
                // Use backend API to get recipe syntax
                this.fetchAndCopyRecipeSyntax();
            });
        }
    }
    
    // Fetch recipe syntax from backend and copy to clipboard
    async fetchAndCopyRecipeSyntax() {
        if (!this.recipeId) {
            showToast('No recipe ID available', 'error');
            return;
        }
        
        try {
            // Fetch recipe syntax from backend
            const response = await fetch(`/api/recipe/${this.recipeId}/syntax`);
            
            if (!response.ok) {
                throw new Error(`Failed to get recipe syntax: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.success && data.syntax) {
                // Use the centralized copyToClipboard utility function
                await copyToClipboard(data.syntax, 'Recipe syntax copied to clipboard');
            } else {
                throw new Error(data.error || 'No syntax returned from server');
            }
        } catch (error) {
            console.error('Error fetching recipe syntax:', error);
            showToast(`Error copying recipe syntax: ${error.message}`, 'error');
        }
    }
    
    // Helper method to copy text to clipboard
    copyToClipboard(text, successMessage) {
        copyToClipboard(text, successMessage);
    }

    // Add new method to handle downloading missing LoRAs
    async showDownloadMissingLorasModal() {
        console.log("currentRecipe", this.currentRecipe);
        // Get missing LoRAs from the current recipe
        const missingLoras = this.currentRecipe.loras.filter(lora => !lora.inLibrary);
        console.log("missingLoras", missingLoras);
        
        if (missingLoras.length === 0) {
            showToast('No missing LoRAs to download', 'info');
            return;
        }

        try {
            state.loadingManager.showSimpleLoading('Getting version info for missing LoRAs...');

            // Get version info for each missing LoRA by calling the appropriate API endpoint
            const missingLorasWithVersionInfoPromises = missingLoras.map(async lora => {
                let endpoint;
                
                // Determine which endpoint to use based on available data
                if (lora.modelVersionId) {
                    endpoint = `/api/loras/civitai/model/version/${lora.modelVersionId}`;
                } else if (lora.hash) {
                    endpoint = `/api/loras/civitai/model/hash/${lora.hash}`;
                } else {
                    console.error("Missing both hash and modelVersionId for lora:", lora);
                    return null;
                }
                
                const response = await fetch(endpoint);
                const versionInfo = await response.json();
                
                // Return original lora data combined with version info
                return {
                    ...lora,
                    civitaiInfo: versionInfo
                };
            });
            
            // Wait for all API calls to complete
            const lorasWithVersionInfo = await Promise.all(missingLorasWithVersionInfoPromises);
            console.log("Loras with version info:", lorasWithVersionInfo);
            
            // Filter out null values (failed requests)
            const validLoras = lorasWithVersionInfo.filter(lora => lora !== null);
            
            if (validLoras.length === 0) {
                showToast('Failed to get information for missing LoRAs', 'error');
                return;
            }
            
            // Close the recipe modal first
            modalManager.closeModal('recipeModal');
            
            // Prepare data for import manager using the retrieved information
            const recipeData = {
                loras: validLoras.map(lora => {
                    const civitaiInfo = lora.civitaiInfo;
                    const modelFile = civitaiInfo.files ? 
                        civitaiInfo.files.find(file => file.type === 'Model') : null;
                    
                    return {
                        // Basic lora info
                        name: civitaiInfo.model?.name || lora.name,
                        version: civitaiInfo.name || '',
                        strength: lora.strength || 1.0,
                        
                        // Model identifiers
                        hash: modelFile?.hashes?.SHA256?.toLowerCase() || lora.hash,
                        id: civitaiInfo.id || lora.modelVersionId,
                        
                        // Metadata
                        thumbnailUrl: civitaiInfo.images?.[0]?.url || '',
                        baseModel: civitaiInfo.baseModel || '',
                        downloadUrl: civitaiInfo.downloadUrl || '',
                        size: modelFile ? (modelFile.sizeKB * 1024) : 0,
                        file_name: modelFile ? modelFile.name.split('.')[0] : '',
                        
                        // Status flags
                        existsLocally: false,
                        isDeleted: civitaiInfo.error === "Model not found",
                        isEarlyAccess: !!civitaiInfo.earlyAccessEndsAt,
                        earlyAccessEndsAt: civitaiInfo.earlyAccessEndsAt || ''
                    };
                })
            };
            
            console.log("recipeData for import:", recipeData);
            
            // Call ImportManager's download missing LoRAs method
            window.importManager.downloadMissingLoras(recipeData, this.currentRecipe.id);
        } catch (error) {
            console.error("Error downloading missing LoRAs:", error);
            showToast('Error preparing LoRAs for download', 'error');
        } finally {
            state.loadingManager.hide();
        }
    }

    // New methods for reconnecting LoRAs
    setupReconnectButtons() {
        // Add event listeners to all deleted badges
        const deletedBadges = document.querySelectorAll('.deleted-badge.reconnectable');
        deletedBadges.forEach(badge => {
            badge.addEventListener('mouseenter', () => {
                badge.querySelector('.badge-text').innerHTML = 'Reconnect';
            });
            
            badge.addEventListener('mouseleave', () => {
                badge.querySelector('.badge-text').innerHTML = '<i class="fas fa-trash-alt"></i> Deleted';
            });
            
            badge.addEventListener('click', (e) => {
                const loraIndex = badge.getAttribute('data-lora-index');
                this.showReconnectInput(loraIndex);
            });
        });
        
        // Add event listeners to reconnect cancel buttons
        const cancelButtons = document.querySelectorAll('.reconnect-cancel-btn');
        cancelButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const container = button.closest('.lora-reconnect-container');
                this.hideReconnectInput(container);
            });
        });
        
        // Add event listeners to reconnect confirm buttons
        const confirmButtons = document.querySelectorAll('.reconnect-confirm-btn');
        confirmButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const container = button.closest('.lora-reconnect-container');
                const input = container.querySelector('.reconnect-input');
                const loraIndex = container.getAttribute('data-lora-index');
                this.reconnectLora(loraIndex, input.value);
            });
        });
        
        // Add keydown handlers to reconnect inputs
        const reconnectInputs = document.querySelectorAll('.reconnect-input');
        reconnectInputs.forEach(input => {
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    const container = input.closest('.lora-reconnect-container');
                    const loraIndex = container.getAttribute('data-lora-index');
                    this.reconnectLora(loraIndex, input.value);
                } else if (e.key === 'Escape') {
                    const container = input.closest('.lora-reconnect-container');
                    this.hideReconnectInput(container);
                }
            });
        });
    }
    
    showReconnectInput(loraIndex) {
        // Hide any currently active reconnect containers
        document.querySelectorAll('.lora-reconnect-container.active').forEach(active => {
            active.classList.remove('active');
        });
        
        // Show the reconnect container for this lora
        const container = document.querySelector(`.lora-reconnect-container[data-lora-index="${loraIndex}"]`);
        if (container) {
            container.classList.add('active');
            const input = container.querySelector('.reconnect-input');
            input.focus();
        }
    }
    
    hideReconnectInput(container) {
        if (container && container.classList.contains('active')) {
            container.classList.remove('active');
            const input = container.querySelector('.reconnect-input');
            if (input) input.value = '';
        }
    }
    
    async reconnectLora(loraIndex, inputValue) {
        if (!inputValue || !inputValue.trim()) {
            showToast('Please enter a LoRA name or syntax', 'error');
            return;
        }
        
        try {
            // Parse input value to extract file_name
            let loraSyntaxMatch = inputValue.match(/<lora:([^:>]+)(?::[^>]+)?>/);
            let fileName = loraSyntaxMatch ? loraSyntaxMatch[1] : inputValue.trim();
            
            // Remove .safetensors extension if present
            fileName = fileName.replace(/\.safetensors$/, '');
            
            state.loadingManager.showSimpleLoading('Reconnecting LoRA...');
            
            // Call API to reconnect the LoRA
            const response = await fetch('/api/recipe/lora/reconnect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    recipe_id: this.recipeId,
                    lora_index: loraIndex,
                    target_name: fileName
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Hide the reconnect input
                const container = document.querySelector(`.lora-reconnect-container[data-lora-index="${loraIndex}"]`);
                this.hideReconnectInput(container);
                
                // Update the current recipe with the updated lora data
                this.currentRecipe.loras[loraIndex] = result.updated_lora;
                
                // Show success message
                showToast('LoRA reconnected successfully', 'success');
                
                // Refresh modal to show updated content
                setTimeout(() => {
                    this.showRecipeDetails(this.currentRecipe);
                }, 500);

                state.virtualScroller.updateSingleItem(this.currentRecipe.file_path, {
                    loras: this.currentRecipe.loras
                });
            } else {
                showToast(`Error: ${result.error}`, 'error');
            }
        } catch (error) {
            console.error('Error reconnecting LoRA:', error);
            showToast(`Error reconnecting LoRA: ${error.message}`, 'error');
        } finally {
            state.loadingManager.hide();
        }
    }

    // New method to navigate to the LoRAs page
    navigateToLorasPage(specificLoraIndex = null) {
        // Close the current modal
        modalManager.closeModal('recipeModal');
        
        // Clear any previous filters first
        removeSessionItem('recipe_to_lora_filterLoraHash');
        removeSessionItem('recipe_to_lora_filterLoraHashes');
        removeSessionItem('filterRecipeName');
        removeSessionItem('viewLoraDetail');
        
        if (specificLoraIndex !== null) {
            // If a specific LoRA index is provided, navigate to view just that one LoRA
            const lora = this.currentRecipe.loras[specificLoraIndex];
            if (lora && lora.hash) {
                // Set session storage to open the LoRA modal directly
                setSessionItem('recipe_to_lora_filterLoraHash', lora.hash.toLowerCase());
                setSessionItem('viewLoraDetail', 'true');
                setSessionItem('filterRecipeName', this.currentRecipe.title);
            }
        } else {
            // If no specific LoRA index is provided, show all LoRAs from this recipe
            // Collect all hashes from the recipe's LoRAs
            const loraHashes = this.currentRecipe.loras
                .filter(lora => lora.hash)
                .map(lora => lora.hash.toLowerCase());
                
            if (loraHashes.length > 0) {
                // Store the LoRA hashes and recipe name in sessionStorage
                setSessionItem('recipe_to_lora_filterLoraHashes', JSON.stringify(loraHashes));
                setSessionItem('filterRecipeName', this.currentRecipe.title);
            }
        }
        
        // Navigate to the LoRAs page
        window.location.href = '/loras';
    }

    // New method to make LoRA items clickable
    setupLoraItemsClickable() {
        const loraItems = document.querySelectorAll('.recipe-lora-item');
        loraItems.forEach(item => {
            // Get the lora index from the data attribute
            const loraIndex = parseInt(item.dataset.loraIndex);
            
            item.addEventListener('click', (e) => {
                // If the click is on the reconnect container or badge, don't navigate
                if (e.target.closest('.lora-reconnect-container') || 
                    e.target.closest('.deleted-badge') ||
                    e.target.closest('.reconnect-tooltip')) {
                    return;
                }
                
                // Navigate to the LoRAs page with the specific LoRA index
                this.navigateToLorasPage(loraIndex);
            });
        });
    }
}

export { RecipeModal };