import { appCore } from './core.js';
import { confirmDelete, closeDeleteModal, confirmExclude, closeExcludeModal } from './utils/modalUtils.js';
import { createPageControls } from './components/controls/index.js';
import { EmbeddingContextMenu } from './components/ContextMenu/index.js';
import { ModelDuplicatesManager } from './components/ModelDuplicatesManager.js';
import { MODEL_TYPES } from './api/apiConfig.js';

// Initialize the Embeddings page
class EmbeddingsPageManager {
    constructor() {
        // Initialize page controls
        this.pageControls = createPageControls(MODEL_TYPES.EMBEDDING);
        
        // Initialize the ModelDuplicatesManager
        this.duplicatesManager = new ModelDuplicatesManager(this, MODEL_TYPES.EMBEDDING);
        
        // Expose only necessary functions to global scope
        this._exposeRequiredGlobalFunctions();
    }
    
    _exposeRequiredGlobalFunctions() {
        // Minimal set of functions that need to remain global
        window.confirmDelete = confirmDelete;
        window.closeDeleteModal = closeDeleteModal;
        window.confirmExclude = confirmExclude;
        window.closeExcludeModal = closeExcludeModal;
        
        // Expose duplicates manager
        window.modelDuplicatesManager = this.duplicatesManager;
    }
    
    async initialize() {
        // Initialize page-specific components
        this.pageControls.restoreFolderFilter();
        this.pageControls.initFolderTagsVisibility();
        
        // Initialize context menu
        new EmbeddingContextMenu();
        
        // Initialize common page features
        appCore.initializePageFeatures();
        
        console.log('Embeddings Manager initialized');
    }
}

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    // Initialize core application
    await appCore.initialize();
    
    // Initialize embeddings page
    const embeddingsPage = new EmbeddingsPageManager();
    await embeddingsPage.initialize();
});
