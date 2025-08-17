/**
 * Utility functions to update checkpoint cards after modal edits
 */

/**
 * Update the recipe card after metadata edits in the modal
 * @param {string} recipeId - ID of the recipe to update
 * @param {Object} updates - Object containing the updates (title, tags, source_path)
 */
export function updateRecipeCard(recipeId, updates) {
    // Find the card with matching recipe ID
    const recipeCard = document.querySelector(`.model-card[data-id="${recipeId}"]`);
    if (!recipeCard) return;

    // Get the recipe card component instance
    const recipeCardInstance = recipeCard._recipeCardInstance;
    
    // Update card dataset and visual elements based on the updates object
    Object.entries(updates).forEach(([key, value]) => {
        // Update dataset
        recipeCard.dataset[key] = value;

        // Update visual elements based on the property
        switch(key) {
            case 'title':
                // Update the title in the recipe object
                if (recipeCardInstance && recipeCardInstance.recipe) {
                    recipeCardInstance.recipe.title = value;
                }
                
                // Update the title shown in the card
                const modelNameElement = recipeCard.querySelector('.model-name');
                if (modelNameElement) modelNameElement.textContent = value;
                break;
                
            case 'tags':
                // Update tags in the recipe object (not displayed on card UI)
                if (recipeCardInstance && recipeCardInstance.recipe) {
                    recipeCardInstance.recipe.tags = value;
                }
                
                // Store in dataset as JSON string
                try {
                    if (typeof value === 'string') {
                        recipeCard.dataset.tags = value;
                    } else {
                        recipeCard.dataset.tags = JSON.stringify(value);
                    }
                } catch (e) {
                    console.error('Failed to update recipe tags:', e);
                }
                break;
                
            case 'source_path':
                // Update source_path in the recipe object (not displayed on card UI)
                if (recipeCardInstance && recipeCardInstance.recipe) {
                    recipeCardInstance.recipe.source_path = value;
                }
                break;
        }
    });
    
    return recipeCard; // Return the updated card element for chaining
}