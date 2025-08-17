// Create the new hierarchical state structure
import { getStorageItem, getMapFromStorage } from '../utils/storageHelpers.js';
import { MODEL_TYPES } from '../api/apiConfig.js';

// Load settings from localStorage or use defaults
const savedSettings = getStorageItem('settings', {
    blurMatureContent: true,
    show_only_sfw: false,
    cardInfoDisplay: 'always'
});

// Load preview versions from localStorage for each model type
const loraPreviewVersions = getMapFromStorage('lora_preview_versions');
const checkpointPreviewVersions = getMapFromStorage('checkpoint_preview_versions');
const embeddingPreviewVersions = getMapFromStorage('embedding_preview_versions');

export const state = {
    // Global state
    global: {
        settings: savedSettings,
        loadingManager: null,
        observer: null,
    },
    
    // Page-specific states
    pages: {
        [MODEL_TYPES.LORA]: {
            currentPage: 1,
            isLoading: false,
            hasMore: true,
            sortBy: 'name',
            activeFolder: null,
            activeLetterFilter: null,
            previewVersions: loraPreviewVersions,
            searchManager: null,
            searchOptions: {
                filename: true,
                modelname: true,
                tags: false,
                creator: false,
                recursive: false
            },
            filters: {
                baseModel: [],
                tags: []
            },
            bulkMode: false,
            selectedLoras: new Set(),
            loraMetadataCache: new Map(),
            showFavoritesOnly: false,
            duplicatesMode: false,
        },
        
        recipes: {
            currentPage: 1,
            isLoading: false,
            hasMore: true,
            sortBy: 'date',
            searchManager: null,
            searchOptions: {
                title: true,
                tags: true,
                loraName: true,
                loraModel: true
            },
            filters: {
                baseModel: [],
                tags: [],
                search: ''
            },
            pageSize: 20,
            showFavoritesOnly: false,
            duplicatesMode: false,
        },
        
        [MODEL_TYPES.CHECKPOINT]: {
            currentPage: 1,
            isLoading: false,
            hasMore: true,
            sortBy: 'name',
            activeFolder: null,
            previewVersions: checkpointPreviewVersions,
            searchManager: null,
            searchOptions: {
                filename: true,
                modelname: true,
                creator: false,
                recursive: false
            },
            filters: {
                baseModel: [],
                tags: []
            },
            modelType: 'checkpoint', // 'checkpoint' or 'diffusion_model'
            bulkMode: false,
            selectedModels: new Set(),
            metadataCache: new Map(),
            showFavoritesOnly: false,
            duplicatesMode: false,
        },
        
        [MODEL_TYPES.EMBEDDING]: {
            currentPage: 1,
            isLoading: false,
            hasMore: true,
            sortBy: 'name',
            activeFolder: null,
            activeLetterFilter: null,
            previewVersions: embeddingPreviewVersions,
            searchManager: null,
            searchOptions: {
                filename: true,
                modelname: true,
                tags: false,
                creator: false,
                recursive: false
            },
            filters: {
                baseModel: [],
                tags: []
            },
            bulkMode: false,
            selectedModels: new Set(),
            metadataCache: new Map(),
            showFavoritesOnly: false,
            duplicatesMode: false,
        }
    },
    
    // Current active page - use MODEL_TYPES constants
    currentPageType: MODEL_TYPES.LORA,
    
    // Backward compatibility - proxy properties
    get currentPage() { return this.pages[this.currentPageType].currentPage; },
    set currentPage(value) { this.pages[this.currentPageType].currentPage = value; },
    
    get isLoading() { return this.pages[this.currentPageType].isLoading; },
    set isLoading(value) { this.pages[this.currentPageType].isLoading = value; },
    
    get hasMore() { return this.pages[this.currentPageType].hasMore; },
    set hasMore(value) { this.pages[this.currentPageType].hasMore = value; },
    
    get sortBy() { return this.pages[this.currentPageType].sortBy; },
    set sortBy(value) { this.pages[this.currentPageType].sortBy = value; },
    
    get activeFolder() { return this.pages[this.currentPageType].activeFolder; },
    set activeFolder(value) { this.pages[this.currentPageType].activeFolder = value; },
    
    get loadingManager() { return this.global.loadingManager; },
    set loadingManager(value) { this.global.loadingManager = value; },
    
    get observer() { return this.global.observer; },
    set observer(value) { this.global.observer = value; },
    
    get previewVersions() { return this.pages.loras.previewVersions; },
    set previewVersions(value) { this.pages.loras.previewVersions = value; },
    
    get searchManager() { return this.pages[this.currentPageType].searchManager; },
    set searchManager(value) { this.pages[this.currentPageType].searchManager = value; },
    
    get searchOptions() { return this.pages[this.currentPageType].searchOptions; },
    set searchOptions(value) { this.pages[this.currentPageType].searchOptions = value; },
    
    get filters() { return this.pages[this.currentPageType].filters; },
    set filters(value) { this.pages[this.currentPageType].filters = value; },
    
    get bulkMode() { 
        const currentType = this.currentPageType;
        if (currentType === MODEL_TYPES.LORA) {
            return this.pages.loras.bulkMode;
        } else {
            return this.pages[currentType].bulkMode;
        }
    },
    set bulkMode(value) { 
        const currentType = this.currentPageType;
        if (currentType === MODEL_TYPES.LORA) {
            this.pages.loras.bulkMode = value;
        } else {
            this.pages[currentType].bulkMode = value;
        }
    },
    
    get selectedLoras() { return this.pages.loras.selectedLoras; },
    set selectedLoras(value) { this.pages.loras.selectedLoras = value; },
    
    get selectedModels() { 
        const currentType = this.currentPageType;
        if (currentType === MODEL_TYPES.LORA) {
            return this.pages.loras.selectedLoras;
        } else {
            return this.pages[currentType].selectedModels;
        }
    },
    set selectedModels(value) { 
        const currentType = this.currentPageType;
        if (currentType === MODEL_TYPES.LORA) {
            this.pages.loras.selectedLoras = value;
        } else {
            this.pages[currentType].selectedModels = value;
        }
    },
    
    get loraMetadataCache() { return this.pages.loras.loraMetadataCache; },
    set loraMetadataCache(value) { this.pages.loras.loraMetadataCache = value; },
    
    get settings() { return this.global.settings; },
    set settings(value) { this.global.settings = value; }
};

// Get the current page state
export function getCurrentPageState() {
    return state.pages[state.currentPageType];
}

// Set the current page type
export function setCurrentPageType(pageType) {
    if (state.pages[pageType]) {
        state.currentPageType = pageType;
        return true;
    }
    console.warn(`Unknown page type: ${pageType}`);
    return false;
}

// Initialize page state when a page loads
export function initPageState(pageType) {
    if (setCurrentPageType(pageType)) {
        console.log(`Initialized state for page: ${pageType}`);
        return getCurrentPageState();
    }
    return null;
}