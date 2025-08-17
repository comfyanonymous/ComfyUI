import { state } from '../state/index.js';

/**
 * API Configuration
 * Centralized configuration for all model types and their endpoints
 */

// Model type definitions
export const MODEL_TYPES = {
    LORA: 'loras',
    CHECKPOINT: 'checkpoints',
    EMBEDDING: 'embeddings' // Future model type
};

// Base API configuration for each model type
export const MODEL_CONFIG = {
    [MODEL_TYPES.LORA]: {
        displayName: 'LoRA',
        singularName: 'lora',
        defaultPageSize: 100,
        supportsLetterFilter: true,
        supportsBulkOperations: true,
        supportsMove: true,
        templateName: 'loras.html'
    },
    [MODEL_TYPES.CHECKPOINT]: {
        displayName: 'Checkpoint',
        singularName: 'checkpoint',
        defaultPageSize: 100,
        supportsLetterFilter: false,
        supportsBulkOperations: true,
        supportsMove: true,
        templateName: 'checkpoints.html'
    },
    [MODEL_TYPES.EMBEDDING]: {
        displayName: 'Embedding',
        singularName: 'embedding',
        defaultPageSize: 100,
        supportsLetterFilter: true,
        supportsBulkOperations: true,
        supportsMove: true,
        templateName: 'embeddings.html'
    }
};

/**
 * Generate API endpoints for a given model type
 * @param {string} modelType - The model type (e.g., 'loras', 'checkpoints')
 * @returns {Object} Object containing all API endpoints for the model type
 */
export function getApiEndpoints(modelType) {
    if (!Object.values(MODEL_TYPES).includes(modelType)) {
        throw new Error(`Invalid model type: ${modelType}`);
    }
    
    return {
        // Base CRUD operations
        list: `/api/${modelType}/list`,
        delete: `/api/${modelType}/delete`,
        exclude: `/api/${modelType}/exclude`,
        rename: `/api/${modelType}/rename`,
        save: `/api/${modelType}/save-metadata`,
        
        // Bulk operations
        bulkDelete: `/api/${modelType}/bulk-delete`,

        // Move operations (now common for all model types that support move)
        moveModel: `/api/${modelType}/move_model`,
        moveBulk: `/api/${modelType}/move_models_bulk`,
        
        // CivitAI integration
        fetchCivitai: `/api/${modelType}/fetch-civitai`,
        fetchAllCivitai: `/api/${modelType}/fetch-all-civitai`,
        relinkCivitai: `/api/${modelType}/relink-civitai`,
        civitaiVersions: `/api/${modelType}/civitai/versions`,
        
        // Preview management
        replacePreview: `/api/${modelType}/replace-preview`,
        
        // Query operations
        scan: `/api/${modelType}/scan`,
        topTags: `/api/${modelType}/top-tags`,
        baseModels: `/api/${modelType}/base-models`,
        roots: `/api/${modelType}/roots`,
        folders: `/api/${modelType}/folders`,
        folderTree: `/api/${modelType}/folder-tree`,
        unifiedFolderTree: `/api/${modelType}/unified-folder-tree`,
        duplicates: `/api/${modelType}/find-duplicates`,
        conflicts: `/api/${modelType}/find-filename-conflicts`,
        verify: `/api/${modelType}/verify-duplicates`,
        
        // Model-specific endpoints (will be merged with specific configs)
        specific: {}
    };
}

/**
 * Model-specific endpoint configurations
 */
export const MODEL_SPECIFIC_ENDPOINTS = {
    [MODEL_TYPES.LORA]: {
        letterCounts: `/api/${MODEL_TYPES.LORA}/letter-counts`,
        notes: `/api/${MODEL_TYPES.LORA}/get-notes`,
        triggerWords: `/api/${MODEL_TYPES.LORA}/get-trigger-words`,
        previewUrl: `/api/${MODEL_TYPES.LORA}/preview-url`,
        civitaiUrl: `/api/${MODEL_TYPES.LORA}/civitai-url`,
        modelDescription: `/api/${MODEL_TYPES.LORA}/model-description`,
        getTriggerWordsPost: `/api/${MODEL_TYPES.LORA}/get_trigger_words`,
        civitaiModelByVersion: `/api/${MODEL_TYPES.LORA}/civitai/model/version`,
        civitaiModelByHash: `/api/${MODEL_TYPES.LORA}/civitai/model/hash`,
    },
    [MODEL_TYPES.CHECKPOINT]: {
        info: `/api/${MODEL_TYPES.CHECKPOINT}/info`,
        checkpoints_roots: `/api/${MODEL_TYPES.CHECKPOINT}/checkpoints_roots`,
        unet_roots: `/api/${MODEL_TYPES.CHECKPOINT}/unet_roots`,
    },
    [MODEL_TYPES.EMBEDDING]: {
    }
};

/**
 * Get complete API configuration for a model type
 * @param {string} modelType - The model type
 * @returns {Object} Complete API configuration
 */
export function getCompleteApiConfig(modelType) {
    const baseEndpoints = getApiEndpoints(modelType);
    const specificEndpoints = MODEL_SPECIFIC_ENDPOINTS[modelType] || {};
    const config = MODEL_CONFIG[modelType];
    
    return {
        modelType,
        config,
        endpoints: {
            ...baseEndpoints,
            specific: specificEndpoints
        }
    };
}

/**
 * Validate if a model type is supported
 * @param {string} modelType - The model type to validate
 * @returns {boolean} True if valid, false otherwise
 */
export function isValidModelType(modelType) {
    return Object.values(MODEL_TYPES).includes(modelType);
}

/**
 * Get model type from current page or explicit parameter
 * @param {string} [explicitType] - Explicitly provided model type
 * @returns {string} The model type
 */
export function getCurrentModelType(explicitType = null) {
    if (explicitType && isValidModelType(explicitType)) {
        return explicitType;
    }

    return state.currentPageType || MODEL_TYPES.LORA;
}

// Download API endpoints (shared across all model types)
export const DOWNLOAD_ENDPOINTS = {
    download: '/api/download-model',
    downloadGet: '/api/download-model-get',
    cancelGet: '/api/cancel-download-get',
    progress: '/api/download-progress'
};

// WebSocket endpoints
export const WS_ENDPOINTS = {
    fetchProgress: '/ws/fetch-progress'
};
