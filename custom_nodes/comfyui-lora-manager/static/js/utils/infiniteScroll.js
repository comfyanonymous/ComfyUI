import { state, getCurrentPageState } from '../state/index.js';
import { VirtualScroller } from './VirtualScroller.js';
import { createModelCard, setupModelCardEventDelegation } from '../components/shared/ModelCard.js';
import { getModelApiClient } from '../api/modelApiFactory.js';
import { showToast } from './uiHelpers.js';

// Function to dynamically import the appropriate card creator based on page type
async function getCardCreator(pageType) {
    if (pageType === 'recipes') {
        // Import the RecipeCard module
        const { RecipeCard } = await import('../components/RecipeCard.js');
        
        // Return a wrapper function that creates a recipe card element
        return (recipe) => {
            const recipeCard = new RecipeCard(recipe, (recipe) => {
                if (window.recipeManager) {
                    window.recipeManager.showRecipeDetails(recipe);
                }
            });
            return recipeCard.element;
        };
    }

    // For other page types, use the shared ModelCard creator
    return (model) => createModelCard(model, pageType);

}

// Function to get the appropriate data fetcher based on page type
async function getDataFetcher(pageType) {
    if (pageType === 'loras' || pageType === 'embeddings' || pageType === 'checkpoints') {
        return (page = 1, pageSize = 100) => getModelApiClient().fetchModelsPage(page, pageSize);
    } else if (pageType === 'recipes') {
        // Import the recipeApi module and use the fetchRecipesPage function
        const { fetchRecipesPage } = await import('../api/recipeApi.js');
        return fetchRecipesPage;
    }
    return null;
}

export async function initializeInfiniteScroll(pageType = 'loras') {
    // Clean up any existing virtual scroller
    if (state.virtualScroller) {
        state.virtualScroller.dispose();
        state.virtualScroller = null;
    }

    // Set the current page type
    state.currentPageType = pageType;
    
    // Get the current page state
    const pageState = getCurrentPageState();
    
    // Skip initializing if in duplicates mode (for recipes page)
    if (pageType === 'recipes' && pageState.duplicatesMode) {
        console.log('Skipping virtual scroll initialization - duplicates mode is active');
        return;
    }

    // Use virtual scrolling for all page types
    await initializeVirtualScroll(pageType);
    
    // Setup event delegation for model cards based on page type
    setupModelCardEventDelegation(pageType);
}

async function initializeVirtualScroll(pageType) {
    // Determine the grid ID based on page type
    let gridId;
    
    switch (pageType) {
        case 'recipes':
            gridId = 'recipeGrid';
            break;
        case 'checkpoints':
        case 'loras':
        default:
            gridId = 'modelGrid';
            break;
    }

    const grid = document.getElementById(gridId);
    
    if (!grid) {
        console.warn(`Grid with ID "${gridId}" not found for virtual scroll`);
        return;
    }
    
    // Change this line to get the actual scrolling container
    const scrollContainer = document.querySelector('.page-content');
    const gridContainer = scrollContainer.querySelector('.container');
    
    if (!gridContainer) {
        console.warn('Grid container element not found for virtual scroll');
        return;
    }
    
    try {
        // Get the card creator and data fetcher for this page type
        const createCardFn = await getCardCreator(pageType);
        const fetchDataFn = await getDataFetcher(pageType);
        
        if (!createCardFn || !fetchDataFn) {
            throw new Error(`Required components not available for ${pageType} page`);
        }
        
        // Initialize virtual scroller with renamed container elements
        state.virtualScroller = new VirtualScroller({
            gridElement: grid,
            containerElement: gridContainer,
            scrollContainer: scrollContainer,
            createItemFn: createCardFn,
            fetchItemsFn: fetchDataFn,
            pageSize: 100,
            rowGap: 20,
            containerPaddingTop: 4,
            containerPaddingBottom: 4,
            enableDataWindowing: false // Explicitly set to false to disable data windowing
        });
        
        // Initialize the virtual scroller
        await state.virtualScroller.initialize();
        
        // Add grid class for CSS styling
        grid.classList.add('virtual-scroll');
        
        // Setup keyboard navigation
        setupKeyboardNavigation();
        
    } catch (error) {
        console.error(`Error initializing virtual scroller for ${pageType}:`, error);
        showToast(`Failed to initialize ${pageType} page. Please reload.`, 'error');
        
        // Fallback: show a message in the grid
        grid.innerHTML = `
            <div class="placeholder-message">
                <h3>Failed to initialize ${pageType}</h3>
                <p>There was an error loading this page. Please try reloading.</p>
            </div>
        `;
    }
}

// Add keyboard navigation setup function
function setupKeyboardNavigation() {
    // Keep track of the last keypress time to prevent multiple rapid triggers
    let lastKeyTime = 0;
    const keyDelay = 300; // ms between allowed keypresses
    
    // Store the event listener reference so we can remove it later if needed
    const keyboardNavHandler = (event) => {
        // Only handle keyboard events when not in form elements
        if (event.target.matches('input, textarea, select')) return;
        
        // Prevent rapid keypresses
        const now = Date.now();
        if (now - lastKeyTime < keyDelay) return;
        lastKeyTime = now;
        
        // Handle navigation keys
        if (event.key === 'PageUp') {
            event.preventDefault();
            if (state.virtualScroller) {
                state.virtualScroller.handlePageUpDown('up');
            }
        } else if (event.key === 'PageDown') {
            event.preventDefault();
            if (state.virtualScroller) {
                state.virtualScroller.handlePageUpDown('down');
            }
        } else if (event.key === 'Home') {
            event.preventDefault();
            if (state.virtualScroller) {
                state.virtualScroller.scrollToTop();
            }
        } else if (event.key === 'End') {
            event.preventDefault();
            if (state.virtualScroller) {
                state.virtualScroller.scrollToBottom();
            }
        }
    };
    
    // Add the event listener
    document.addEventListener('keydown', keyboardNavHandler);
    
    // Store the handler in state for potential cleanup
    state.keyboardNavHandler = keyboardNavHandler;
}

// Add cleanup function to remove keyboard navigation when needed
export function cleanupKeyboardNavigation() {
    if (state.keyboardNavHandler) {
        document.removeEventListener('keydown', state.keyboardNavHandler);
        state.keyboardNavHandler = null;
    }
}

// Export a method to refresh the virtual scroller when filters change
export function refreshVirtualScroll() {
    if (state.virtualScroller) {
        state.virtualScroller.reset();
        state.virtualScroller.initialize();
    }
}