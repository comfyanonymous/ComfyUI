/**
 * RecipeTab - Handles the recipes tab in the Lora Modal
 */
import { showToast, copyToClipboard } from '../../utils/uiHelpers.js';
import { setSessionItem, removeSessionItem } from '../../utils/storageHelpers.js';

/**
 * Loads recipes that use the specified Lora and renders them in the tab
 * @param {string} loraName - The display name of the Lora
 * @param {string} sha256 - The SHA256 hash of the Lora
 */
export function loadRecipesForLora(loraName, sha256) {
    const recipeTab = document.getElementById('recipes-tab');
    if (!recipeTab) return;
    
    // Show loading state
    recipeTab.innerHTML = `
        <div class="recipes-loading">
            <i class="fas fa-spinner fa-spin"></i> Loading recipes...
        </div>
    `;
    
    // Fetch recipes that use this Lora by hash
    fetch(`/api/recipes/for-lora?hash=${encodeURIComponent(sha256.toLowerCase())}`)
        .then(response => response.json())
        .then(data => {
            if (!data.success) {
                throw new Error(data.error || 'Failed to load recipes');
            }
            
            renderRecipes(recipeTab, data.recipes, loraName, sha256);
        })
        .catch(error => {
            console.error('Error loading recipes for Lora:', error);
            recipeTab.innerHTML = `
                <div class="recipes-error">
                    <i class="fas fa-exclamation-circle"></i>
                    <p>Failed to load recipes. Please try again later.</p>
                </div>
            `;
        });
}

/**
 * Renders the recipe cards in the tab
 * @param {HTMLElement} tabElement - The tab element to render into
 * @param {Array} recipes - Array of recipe objects
 * @param {string} loraName - The display name of the Lora
 * @param {string} loraHash - The hash of the Lora
 */
function renderRecipes(tabElement, recipes, loraName, loraHash) {
    if (!recipes || recipes.length === 0) {
        tabElement.innerHTML = `
            <div class="recipes-empty">
                <i class="fas fa-book-open"></i>
                <p>No recipes found that use this Lora.</p>
            </div>
        `;
        
        return;
    }

    // Create header with count and view all button
    const headerElement = document.createElement('div');
    headerElement.className = 'recipes-header';
    headerElement.innerHTML = `
        <h3>Found ${recipes.length} recipe${recipes.length > 1 ? 's' : ''} using this Lora</h3>
        <button class="view-all-btn" title="View all in Recipes page">
            <i class="fas fa-external-link-alt"></i> View All in Recipes
        </button>
    `;
    
    // Add click handler for "View All" button
    headerElement.querySelector('.view-all-btn').addEventListener('click', () => {
        navigateToRecipesPage(loraName, loraHash);
    });
    
    // Create grid container for recipe cards
    const cardGrid = document.createElement('div');
    cardGrid.className = 'card-grid';
    
    // Create recipe cards matching the structure in recipes.html
    recipes.forEach(recipe => {
        // Get basic info
        const baseModel = recipe.base_model || '';
        const loras = recipe.loras || [];
        const lorasCount = loras.length;
        const missingLorasCount = loras.filter(lora => !lora.inLibrary && !lora.isDeleted).length;
        const allLorasAvailable = missingLorasCount === 0 && lorasCount > 0;
        
        // Ensure file_url exists, fallback to file_path if needed
        const imageUrl = recipe.file_url || 
                         (recipe.file_path ? `/loras_static/root1/preview/${recipe.file_path.split('/').pop()}` : 
                         '/loras_static/images/no-preview.png');
        
        // Create card element matching the structure in recipes.html
        const card = document.createElement('div');
        card.className = 'lora-card';
        card.dataset.filePath = recipe.file_path || '';
        card.dataset.title = recipe.title || '';
        card.dataset.created = recipe.created_date || '';
        card.dataset.id = recipe.id || '';
        
        card.innerHTML = `
            <div class="card-preview">
                <img src="${imageUrl}" alt="${recipe.title}" loading="lazy">
                <div class="card-header">
                    ${baseModel ? `<span class="base-model-label" title="${baseModel}">${baseModel}</span>` : ''}
                    <div class="card-actions">
                        <i class="fas fa-copy" title="Copy Recipe Syntax"></i>
                    </div>
                </div>
                <div class="card-footer">
                    <div class="model-info">
                        <span class="model-name">${recipe.title}</span>
                    </div>
                    <div class="lora-count ${allLorasAvailable ? 'ready' : (lorasCount > 0 ? 'missing' : '')}" 
                         title="${getLoraStatusTitle(lorasCount, missingLorasCount)}">
                        <i class="fas fa-layer-group"></i> ${lorasCount}
                    </div>
                </div>
            </div>
        `;
        
        // Add event listeners for action buttons
        card.querySelector('.fa-copy').addEventListener('click', (e) => {
            e.stopPropagation();
            copyRecipeSyntax(recipe.id);
        });
        
        // Add click handler for the entire card
        card.addEventListener('click', () => {
            navigateToRecipeDetails(recipe.id);
        });
        
        // Add card to grid
        cardGrid.appendChild(card);
    });
    
    // Clear loading indicator and append content
    tabElement.innerHTML = '';
    tabElement.appendChild(headerElement);
    tabElement.appendChild(cardGrid);
}

/**
 * Returns a descriptive title for the LoRA status indicator
 * @param {number} totalCount - Total number of LoRAs in recipe
 * @param {number} missingCount - Number of missing LoRAs
 * @returns {string} Status title text
 */
function getLoraStatusTitle(totalCount, missingCount) {
    if (totalCount === 0) return "No LoRAs in this recipe";
    if (missingCount === 0) return "All LoRAs available - Ready to use";
    return `${missingCount} of ${totalCount} LoRAs missing`;
}

/**
 * Copies recipe syntax to clipboard
 * @param {string} recipeId - The recipe ID
 */
function copyRecipeSyntax(recipeId) {
    if (!recipeId) {
        showToast('Cannot copy recipe syntax: Missing recipe ID', 'error');
        return;
    }

    fetch(`/api/recipe/${recipeId}/syntax`)
        .then(response => response.json())
        .then(data => {
            if (data.success && data.syntax) {
                return copyToClipboard(data.syntax, 'Recipe syntax copied to clipboard');
            } else {
                throw new Error(data.error || 'No syntax returned');
            }
        })
        .catch(err => {
            console.error('Failed to copy: ', err);
            showToast('Failed to copy recipe syntax', 'error');
        });
}

/**
 * Navigates to the recipes page with filter for the current Lora
 * @param {string} loraName - The Lora display name to filter by
 * @param {string} loraHash - The hash of the Lora to filter by
 * @param {boolean} createNew - Whether to open the create recipe dialog
 */
function navigateToRecipesPage(loraName, loraHash) {
    // Close the current modal
    if (window.modalManager) {
        modalManager.closeModal('loraModal');
    }
    
    // Clear any previous filters first
    removeSessionItem('lora_to_recipe_filterLoraName');
    removeSessionItem('lora_to_recipe_filterLoraHash');
    removeSessionItem('viewRecipeId');
    
    // Store the LoRA name and hash filter in sessionStorage
    setSessionItem('lora_to_recipe_filterLoraName', loraName);
    setSessionItem('lora_to_recipe_filterLoraHash', loraHash);
    
    // Directly navigate to recipes page
    window.location.href = '/loras/recipes';
}

/**
 * Navigates directly to a specific recipe's details
 * @param {string} recipeId - The recipe ID to view
 */
function navigateToRecipeDetails(recipeId) {
    // Close the current modal
    if (window.modalManager) {
        modalManager.closeModal('loraModal');
    }
    
    // Clear any previous filters first
    removeSessionItem('filterLoraName');
    removeSessionItem('filterLoraHash');
    removeSessionItem('viewRecipeId');
    
    // Store the recipe ID in sessionStorage to load on recipes page
    setSessionItem('viewRecipeId', recipeId);
    
    // Directly navigate to recipes page
    window.location.href = '/loras/recipes';
}