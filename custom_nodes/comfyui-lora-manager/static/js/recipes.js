// Recipe manager module
import { appCore } from './core.js';
import { ImportManager } from './managers/ImportManager.js';
import { RecipeModal } from './components/RecipeModal.js';
import { getCurrentPageState, state } from './state/index.js';
import { getSessionItem, removeSessionItem } from './utils/storageHelpers.js';
import { RecipeContextMenu } from './components/ContextMenu/index.js';
import { DuplicatesManager } from './components/DuplicatesManager.js';
import { refreshVirtualScroll } from './utils/infiniteScroll.js';
import { refreshRecipes } from './api/recipeApi.js';

class RecipeManager {
    constructor() {
        // Get page state
        this.pageState = getCurrentPageState();
        
        // Initialize ImportManager
        this.importManager = new ImportManager();
        
        // Initialize RecipeModal
        this.recipeModal = new RecipeModal();
        
        // Initialize DuplicatesManager
        this.duplicatesManager = new DuplicatesManager(this);
        
        // Add state tracking for infinite scroll
        this.pageState.isLoading = false;
        this.pageState.hasMore = true;
        
        // Custom filter state - move to pageState for compatibility with virtual scrolling
        this.pageState.customFilter = {
            active: false,
            loraName: null,
            loraHash: null,
            recipeId: null
        };
    }
    
    async initialize() {
        // Initialize event listeners
        this.initEventListeners();
        
        // Set default search options if not already defined
        this._initSearchOptions();
        
        // Initialize context menu
        new RecipeContextMenu();
        
        // Check for custom filter parameters in session storage
        this._checkCustomFilter();
        
        // Expose necessary functions to the page
        this._exposeGlobalFunctions();
        
        // Initialize common page features
        appCore.initializePageFeatures();
    }
    
    _initSearchOptions() {
        // Ensure recipes search options are properly initialized
        if (!this.pageState.searchOptions) {
            this.pageState.searchOptions = {
                title: true,       // Recipe title
                tags: true,        // Recipe tags
                loraName: true,    // LoRA file name
                loraModel: true    // LoRA model name
            };
        }
    }
    
    _exposeGlobalFunctions() {
        // Only expose what's needed for the page
        window.recipeManager = this;
        window.importManager = this.importManager;
    }
    
    _checkCustomFilter() {
        // Check for Lora filter
        const filterLoraName = getSessionItem('lora_to_recipe_filterLoraName');
        const filterLoraHash = getSessionItem('lora_to_recipe_filterLoraHash');
        
        // Check for specific recipe ID
        const viewRecipeId = getSessionItem('viewRecipeId');
        
        // Set custom filter if any parameter is present
        if (filterLoraName || filterLoraHash || viewRecipeId) {
            this.pageState.customFilter = {
                active: true,
                loraName: filterLoraName,
                loraHash: filterLoraHash,
                recipeId: viewRecipeId
            };
            
            // Show custom filter indicator
            this._showCustomFilterIndicator();
        }
    }
    
    _showCustomFilterIndicator() {
        const indicator = document.getElementById('customFilterIndicator');
        const textElement = document.getElementById('customFilterText');
        
        if (!indicator || !textElement) return;
        
        // Update text based on filter type
        let filterText = '';
        
        if (this.pageState.customFilter.recipeId) {
            filterText = 'Viewing specific recipe';
        } else if (this.pageState.customFilter.loraName) {
            // Format with Lora name
            const loraName = this.pageState.customFilter.loraName;
            const displayName = loraName.length > 25 ? 
                loraName.substring(0, 22) + '...' : 
                loraName;
                
            filterText = `<span>Recipes using: <span class="lora-name">${displayName}</span></span>`;
        } else {
            filterText = 'Filtered recipes';
        }
        
        // Update indicator text and show it
        textElement.innerHTML = filterText;
        // Add title attribute to show the lora name as a tooltip
        if (this.pageState.customFilter.loraName) {
            textElement.setAttribute('title', this.pageState.customFilter.loraName);
        }
        indicator.classList.remove('hidden');
        
        // Add pulse animation
        const filterElement = indicator.querySelector('.filter-active');
        if (filterElement) {
            filterElement.classList.add('animate');
            setTimeout(() => filterElement.classList.remove('animate'), 600);
        }
        
        // Add click handler for clear filter button
        const clearFilterBtn = indicator.querySelector('.clear-filter');
        if (clearFilterBtn) {
            clearFilterBtn.addEventListener('click', (e) => {
                e.stopPropagation();  // Prevent button click from triggering
                this._clearCustomFilter();
            });
        }
    }
    
    _clearCustomFilter() {
        // Reset custom filter
        this.pageState.customFilter = {
            active: false,
            loraName: null,
            loraHash: null,
            recipeId: null
        };
        
        // Hide indicator
        const indicator = document.getElementById('customFilterIndicator');
        if (indicator) {
            indicator.classList.add('hidden');
        }
        
        // Clear any session storage items
        removeSessionItem('lora_to_recipe_filterLoraName');
        removeSessionItem('lora_to_recipe_filterLoraHash');
        removeSessionItem('viewRecipeId');
        
        // Reset and refresh the virtual scroller
        refreshVirtualScroll();
    }
    
    initEventListeners() {
        // Sort select
        const sortSelect = document.getElementById('sortSelect');
        if (sortSelect) {
            sortSelect.addEventListener('change', () => {
                this.pageState.sortBy = sortSelect.value;
                refreshVirtualScroll();
            });
        }
    }
    
    // This method is kept for compatibility but now uses virtual scrolling
    async loadRecipes(resetPage = true) {
        // Skip loading if in duplicates mode
        const pageState = getCurrentPageState();
        if (pageState.duplicatesMode) {
            return;
        }
        
        if (resetPage) {
            refreshVirtualScroll();
        }
    }
    
    /**
     * Refreshes the recipe list by first rebuilding the cache and then loading recipes
     */
    async refreshRecipes() {
        return refreshRecipes();
    }
    
    showRecipeDetails(recipe) {
        this.recipeModal.showRecipeDetails(recipe);
    }
    
    // Duplicate detection and management methods
    async findDuplicateRecipes() {
        return await this.duplicatesManager.findDuplicates();
    }
    
    selectLatestDuplicates() {
        this.duplicatesManager.selectLatestDuplicates();
    }
    
    deleteSelectedDuplicates() {
        this.duplicatesManager.deleteSelectedDuplicates();
    }

    confirmDeleteDuplicates() {
        this.duplicatesManager.confirmDeleteDuplicates();
    }
    
    exitDuplicateMode() {
        // Clear the grid first to prevent showing old content temporarily
        const recipeGrid = document.getElementById('recipeGrid');
        if (recipeGrid) {
            recipeGrid.innerHTML = '';
        }
        
        this.duplicatesManager.exitDuplicateMode();
    }
}

// Initialize components
document.addEventListener('DOMContentLoaded', async () => {
    // Initialize core application
    await appCore.initialize();
    
    // Initialize recipe manager
    const recipeManager = new RecipeManager();
    await recipeManager.initialize();
});

// Export for use in other modules
export { RecipeManager };