// EmbeddingsControls.js - Specific implementation for the Embeddings page
import { PageControls } from './PageControls.js';
import { getModelApiClient, resetAndReload } from '../../api/modelApiFactory.js';
import { showToast } from '../../utils/uiHelpers.js';
import { downloadManager } from '../../managers/DownloadManager.js';

/**
 * EmbeddingsControls class - Extends PageControls for Embedding-specific functionality
 */
export class EmbeddingsControls extends PageControls {
    constructor() {
        // Initialize with 'embeddings' page type
        super('embeddings');
        
        // Register API methods specific to the Embeddings page
        this.registerEmbeddingsAPI();
    }
    
    /**
     * Register Embedding-specific API methods
     */
    registerEmbeddingsAPI() {
        const embeddingsAPI = {
            // Core API functions
            loadMoreModels: async (resetPage = false, updateFolders = false) => {
                return await getModelApiClient().loadMoreWithVirtualScroll(resetPage, updateFolders);
            },
            
            resetAndReload: async (updateFolders = false) => {
                return await resetAndReload(updateFolders);
            },
            
            refreshModels: async (fullRebuild = false) => {
                return await getModelApiClient().refreshModels(fullRebuild);
            },
            
            // Add fetch from Civitai functionality for embeddings
            fetchFromCivitai: async () => {
                return await getModelApiClient().fetchCivitaiMetadata();
            },
            
            // Add show download modal functionality
            showDownloadModal: () => {
                downloadManager.showDownloadModal();
            },
            
            // No clearCustomFilter implementation is needed for embeddings
            // as custom filters are currently only used for LoRAs
            clearCustomFilter: async () => {
                showToast('No custom filter to clear', 'info');
            }
        };
        
        // Register the API
        this.registerAPI(embeddingsAPI);
    }
}
