import { BaseContextMenu } from './BaseContextMenu.js';
import { ModelContextMenuMixin } from './ModelContextMenuMixin.js';
import { showToast, copyToClipboard, sendLoraToWorkflow } from '../../utils/uiHelpers.js';
import { setSessionItem, removeSessionItem } from '../../utils/storageHelpers.js';
import { updateRecipeMetadata } from '../../api/recipeApi.js';
import { state } from '../../state/index.js';

export class RecipeContextMenu extends BaseContextMenu {
    constructor() {
        super('recipeContextMenu', '.model-card');
        this.nsfwSelector = document.getElementById('nsfwLevelSelector');
        this.modelType = 'recipe';
        
        // Initialize NSFW Level Selector events
        if (this.nsfwSelector) {
            this.initNSFWSelector();
        }
    }

    // Use the updateRecipeMetadata implementation from recipeApi
    async saveModelMetadata(filePath, data) {
        return updateRecipeMetadata(filePath, data);
    }

    // Override resetAndReload for recipe context
    async resetAndReload() {
        const { resetAndReload } = await import('../../api/recipeApi.js');
        return resetAndReload();
    }
    
    showMenu(x, y, card) {
        // Call the parent method first to handle basic positioning
        super.showMenu(x, y, card);
        
        // Get recipe data to check for missing LoRAs
        const recipeId = card.dataset.id;
        const missingLorasItem = this.menu.querySelector('.download-missing-item');
        
        if (recipeId && missingLorasItem) {
            // Check if this card has missing LoRAs
            const loraCountElement = card.querySelector('.lora-count');
            const hasMissingLoras = loraCountElement && loraCountElement.classList.contains('missing');
            
            // Show/hide the download missing LoRAs option based on missing status
            if (hasMissingLoras) {
                missingLorasItem.style.display = 'flex';
            } else {
                missingLorasItem.style.display = 'none';
            }
        }
    }
    
    handleMenuAction(action) {
        // First try to handle with common actions from ModelContextMenuMixin
        if (ModelContextMenuMixin.handleCommonMenuActions.call(this, action)) {
            return;
        }

        // Handle recipe-specific actions
        const recipeId = this.currentCard.dataset.id;
        
        switch(action) {
            case 'details':
                // Show recipe details
                this.currentCard.click();
                break;
            case 'copy':
                // Copy recipe syntax to clipboard
                this.copyRecipeSyntax();
                break;
            case 'sendappend':
                // Send recipe to workflow (append mode)
                this.sendRecipeToWorkflow(false);
                break;
            case 'sendreplace':
                // Send recipe to workflow (replace mode)
                this.sendRecipeToWorkflow(true);
                break;
            case 'share':
                // Share recipe
                this.currentCard.querySelector('.fa-share-alt')?.click();
                break;
            case 'delete':
                // Delete recipe
                this.currentCard.querySelector('.fa-trash')?.click();
                break;
            case 'viewloras':
                // View all LoRAs in the recipe
                this.viewRecipeLoRAs(recipeId);
                break;
            case 'download-missing':
                // Download missing LoRAs
                this.downloadMissingLoRAs(recipeId);
                break;
        }
    }
    
    // New method to copy recipe syntax to clipboard
    copyRecipeSyntax() {
        const recipeId = this.currentCard.dataset.id;
        if (!recipeId) {
            showToast('Cannot copy recipe: Missing recipe ID', 'error');
            return;
        }

        fetch(`/api/recipe/${recipeId}/syntax`)
            .then(response => response.json())
            .then(data => {
                if (data.success && data.syntax) {
                    copyToClipboard(data.syntax, 'Recipe syntax copied to clipboard');
                } else {
                    throw new Error(data.error || 'No syntax returned');
                }
            })
            .catch(err => {
                console.error('Failed to copy recipe syntax: ', err);
                showToast('Failed to copy recipe syntax', 'error');
            });
    }
    
    // New method to send recipe to workflow
    sendRecipeToWorkflow(replaceMode) {
        const recipeId = this.currentCard.dataset.id;
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
    }
    
    // View all LoRAs in the recipe
    viewRecipeLoRAs(recipeId) {
        if (!recipeId) {
            showToast('Cannot view LoRAs: Missing recipe ID', 'error');
            return;
        }
        
        // First get the recipe details to access its LoRAs
        fetch(`/api/recipe/${recipeId}`)
            .then(response => response.json())
            .then(recipe => {
                // Clear any previous filters first
                removeSessionItem('recipe_to_lora_filterLoraHash');
                removeSessionItem('recipe_to_lora_filterLoraHashes');
                removeSessionItem('filterRecipeName');
                removeSessionItem('viewLoraDetail');
                
                // Collect all hashes from the recipe's LoRAs
                const loraHashes = recipe.loras
                    .filter(lora => lora.hash)
                    .map(lora => lora.hash.toLowerCase());
                    
                if (loraHashes.length > 0) {
                    // Store the LoRA hashes and recipe name in session storage
                    setSessionItem('recipe_to_lora_filterLoraHashes', JSON.stringify(loraHashes));
                    setSessionItem('filterRecipeName', recipe.title);
                    
                    // Navigate to the LoRAs page
                    window.location.href = '/loras';
                } else {
                    showToast('No LoRAs found in this recipe', 'info');
                }
            })
            .catch(error => {
                console.error('Error loading recipe LoRAs:', error);
                showToast('Error loading recipe LoRAs: ' + error.message, 'error');
            });
    }
    
    // Download missing LoRAs
    async downloadMissingLoRAs(recipeId) {
        if (!recipeId) {
            showToast('Cannot download LoRAs: Missing recipe ID', 'error');
            return;
        }
        
        try {
            // First get the recipe details
            const response = await fetch(`/api/recipe/${recipeId}`);
            const recipe = await response.json();
            
            // Get missing LoRAs
            const missingLoras = recipe.loras.filter(lora => !lora.inLibrary && !lora.isDeleted);
            
            if (missingLoras.length === 0) {
                showToast('No missing LoRAs to download', 'info');
                return;
            }
            
            // Show loading toast
            state.loadingManager.showSimpleLoading('Getting version info for missing LoRAs...');
            
            // Get version info for each missing LoRA
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
                
                const versionResponse = await fetch(endpoint);
                const versionInfo = await versionResponse.json();
                
                // Return original lora data combined with version info
                return {
                    ...lora,
                    civitaiInfo: versionInfo
                };
            });
            
            // Wait for all API calls to complete
            const lorasWithVersionInfo = await Promise.all(missingLorasWithVersionInfoPromises);
            
            // Filter out null values (failed requests)
            const validLoras = lorasWithVersionInfo.filter(lora => lora !== null);
            
            if (validLoras.length === 0) {
                showToast('Failed to get information for missing LoRAs', 'error');
                return;
            }
            
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
                        modelVersionId: civitaiInfo.id || lora.modelVersionId,
                        
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
            
            // Call ImportManager's download missing LoRAs method
            window.importManager.downloadMissingLoras(recipeData, recipeId);
        } catch (error) {
            console.error('Error downloading missing LoRAs:', error);
            showToast('Error preparing LoRAs for download: ' + error.message, 'error');
        } finally {
            if (state.loadingManager) {
                state.loadingManager.hide();
            }
        }
    }
}

// Mix in shared methods from ModelContextMenuMixin
Object.assign(RecipeContextMenu.prototype, ModelContextMenuMixin);