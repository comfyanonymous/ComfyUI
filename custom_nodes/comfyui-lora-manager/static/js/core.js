// Core application functionality
import { state } from './state/index.js';
import { LoadingManager } from './managers/LoadingManager.js';
import { modalManager } from './managers/ModalManager.js';
import { updateService } from './managers/UpdateService.js';
import { HeaderManager } from './components/Header.js';
import { settingsManager } from './managers/SettingsManager.js';
import { moveManager } from './managers/MoveManager.js';
import { bulkManager } from './managers/BulkManager.js';
import { exampleImagesManager } from './managers/ExampleImagesManager.js';
import { helpManager } from './managers/HelpManager.js';
import { bannerService } from './managers/BannerService.js';
import { showToast, initTheme, initBackToTop } from './utils/uiHelpers.js';
import { initializeInfiniteScroll } from './utils/infiniteScroll.js';
import { migrateStorageItems } from './utils/storageHelpers.js';

// Core application class
export class AppCore {
    constructor() {
        this.initialized = false;
    }
    
    // Initialize core functionality
    async initialize() {
        if (this.initialized) return;

        console.log('AppCore: Initializing...');
        
        // Initialize managers
        state.loadingManager = new LoadingManager();
        modalManager.initialize();
        updateService.initialize();
        bannerService.initialize();
        window.modalManager = modalManager;
        window.settingsManager = settingsManager;
        window.exampleImagesManager = exampleImagesManager;
        window.helpManager = helpManager;
        window.moveManager = moveManager;
        window.bulkManager = bulkManager;
        
        // Initialize UI components
        window.headerManager = new HeaderManager();
        initTheme();
        initBackToTop();

        // Initialize the bulk manager
        bulkManager.initialize();
        
        // Initialize the example images manager
        exampleImagesManager.initialize();
        // Initialize the help manager
        helpManager.initialize();

        const cardInfoDisplay = state.global.settings.cardInfoDisplay || 'always';
        document.body.classList.toggle('hover-reveal', cardInfoDisplay === 'hover');
        
        // Mark as initialized
        this.initialized = true;
        
        // Return the core instance for chaining
        return this;
    }
    
    // Get the current page type
    getPageType() {
        const body = document.body;
        return body.dataset.page || 'unknown';
    }
    
    // Show toast messages
    showToast(message, type = 'info') {
        showToast(message, type);
    }
    
    // Initialize common UI features based on page type
    initializePageFeatures() {
        const pageType = this.getPageType();
        
        // Initialize virtual scroll for pages that need it
        if (['loras', 'recipes', 'checkpoints', 'embeddings'].includes(pageType)) {
            initializeInfiniteScroll(pageType);
        }
        
        return this;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    // Migrate localStorage items to use the namespace prefix
    migrateStorageItems();
});

// Create and export a singleton instance
export const appCore = new AppCore();