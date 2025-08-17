// PageControls.js - Manages controls for both LoRAs and Checkpoints pages
import { getCurrentPageState, setCurrentPageType } from '../../state/index.js';
import { getStorageItem, setStorageItem, getSessionItem, setSessionItem } from '../../utils/storageHelpers.js';
import { showToast } from '../../utils/uiHelpers.js';

/**
 * PageControls class - Unified control management for model pages
 */
export class PageControls {
    constructor(pageType) {
        // Set the current page type in state
        setCurrentPageType(pageType);
        
        // Store the page type
        this.pageType = pageType;
        
        // Get the current page state
        this.pageState = getCurrentPageState();
        
        // Initialize state based on page type
        this.initializeState();
        
        // Store API methods
        this.api = null;
        
        // Initialize event listeners
        this.initEventListeners();
        
        // Initialize favorites filter button state
        this.initFavoritesFilter();
        
        console.log(`PageControls initialized for ${pageType} page`);
    }
    
    /**
     * Initialize state based on page type
     */
    initializeState() {
        // Set default values
        this.pageState.pageSize = 100;
        this.pageState.isLoading = false;
        this.pageState.hasMore = true;
        
        // Set default sort based on page type
        this.pageState.sortBy = this.pageType === 'loras' ? 'name:asc' : 'name:asc';
        
        // Load sort preference
        this.loadSortPreference();
    }
    
    /**
     * Register API methods for the page
     * @param {Object} api - API methods for the page
     */
    registerAPI(api) {
        this.api = api;
        console.log(`API methods registered for ${this.pageType} page`);
    }
    
    /**
     * Initialize event listeners for controls
     */
    initEventListeners() {
        // Sort select handler
        const sortSelect = document.getElementById('sortSelect');
        if (sortSelect) {
            sortSelect.value = this.pageState.sortBy;
            sortSelect.addEventListener('change', async (e) => {
                this.pageState.sortBy = e.target.value;
                this.saveSortPreference(e.target.value);
                await this.resetAndReload();
            });
        }
        
        // Use event delegation for folder tags - this is the key fix
        const folderTagsContainer = document.querySelector('.folder-tags-container');
        if (folderTagsContainer) {
            folderTagsContainer.addEventListener('click', (e) => {
                const tag = e.target.closest('.tag');
                if (tag) {
                    this.handleFolderClick(tag);
                }
            });
        }
        
        // Refresh button handler
        const refreshBtn = document.querySelector('[data-action="refresh"]');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshModels(false)); // Regular refresh (incremental)
        }
        
        // Initialize dropdown functionality
        this.initDropdowns();
        
        // Toggle folders button
        const toggleFoldersBtn = document.querySelector('.toggle-folders-btn');
        if (toggleFoldersBtn) {
            toggleFoldersBtn.addEventListener('click', () => this.toggleFolderTags());
        }
        
        // Clear custom filter handler
        const clearFilterBtn = document.querySelector('.clear-filter');
        if (clearFilterBtn) {
            clearFilterBtn.addEventListener('click', () => this.clearCustomFilter());
        }
        
        // Page-specific event listeners
        this.initPageSpecificListeners();
    }
    
    /**
     * Initialize dropdown functionality
     */
    initDropdowns() {
        // Handle dropdown toggles
        const dropdownToggles = document.querySelectorAll('.dropdown-toggle');
        dropdownToggles.forEach(toggle => {
            toggle.addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent triggering parent button
                const dropdownGroup = toggle.closest('.dropdown-group');
                
                // Close all other open dropdowns first
                document.querySelectorAll('.dropdown-group.active').forEach(group => {
                    if (group !== dropdownGroup) {
                        group.classList.remove('active');
                    }
                });
                
                // Toggle current dropdown
                dropdownGroup.classList.toggle('active');
            });
        });
        
        // Handle quick refresh option
        const quickRefreshOption = document.querySelector('[data-action="quick-refresh"]');
        if (quickRefreshOption) {
            quickRefreshOption.addEventListener('click', (e) => {
                e.stopPropagation();
                this.refreshModels(false);
                // Close the dropdown
                document.querySelector('.dropdown-group.active')?.classList.remove('active');
            });
        }
        
        // Handle full rebuild option
        const fullRebuildOption = document.querySelector('[data-action="full-rebuild"]');
        if (fullRebuildOption) {
            fullRebuildOption.addEventListener('click', (e) => {
                e.stopPropagation();
                this.refreshModels(true);
                // Close the dropdown
                document.querySelector('.dropdown-group.active')?.classList.remove('active');
            });
        }
        
        // Close dropdowns when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.dropdown-group')) {
                document.querySelectorAll('.dropdown-group.active').forEach(group => {
                    group.classList.remove('active');
                });
            }
        });
    }
    
    /**
     * Initialize page-specific event listeners
     */
    initPageSpecificListeners() {
        // Fetch from Civitai button - available for both loras and checkpoints
        const fetchButton = document.querySelector('[data-action="fetch"]');
        if (fetchButton) {
            fetchButton.addEventListener('click', () => this.fetchFromCivitai());
        }
        
        const downloadButton = document.querySelector('[data-action="download"]');
        if (downloadButton) {
            downloadButton.addEventListener('click', () => this.showDownloadModal());
        }
        
        // Find duplicates button - available for both loras and checkpoints
        const duplicatesButton = document.querySelector('[data-action="find-duplicates"]');
        if (duplicatesButton) {
            duplicatesButton.addEventListener('click', () => this.findDuplicates());
        }
        
        if (this.pageType === 'loras') {
            // Bulk operations button - LoRAs only
            const bulkButton = document.querySelector('[data-action="bulk"]');
            if (bulkButton) {
                bulkButton.addEventListener('click', () => this.toggleBulkMode());
            }
        }
        
        // Favorites filter button handler
        const favoriteFilterBtn = document.getElementById('favoriteFilterBtn');
        if (favoriteFilterBtn) {
            favoriteFilterBtn.addEventListener('click', () => this.toggleFavoritesOnly());
        }
    }
    
    /**
     * Toggle folder selection
     * @param {HTMLElement} tagElement - The folder tag element that was clicked
     */
    handleFolderClick(tagElement) {
        const folder = tagElement.dataset.folder;
        const wasActive = tagElement.classList.contains('active');
        
        document.querySelectorAll('.folder-tags .tag').forEach(t => {
            t.classList.remove('active');
        });
        
        if (!wasActive) {
            tagElement.classList.add('active');
            this.pageState.activeFolder = folder;
            setStorageItem(`${this.pageType}_activeFolder`, folder);
        } else {
            this.pageState.activeFolder = null;
            setStorageItem(`${this.pageType}_activeFolder`, null);
        }
        
        this.resetAndReload();
    }
    
    /**
     * Restore folder filter from storage
     */
    restoreFolderFilter() {
        const activeFolder = getStorageItem(`${this.pageType}_activeFolder`);
        const folderTag = activeFolder && document.querySelector(`.tag[data-folder="${activeFolder}"]`);
        
        if (folderTag) {
            folderTag.classList.add('active');
            this.pageState.activeFolder = activeFolder;
            this.filterByFolder(activeFolder);
        }
    }
    
    /**
     * Filter displayed cards by folder
     * @param {string} folderPath - Folder path to filter by
     */
    filterByFolder(folderPath) {
        const cardSelector = this.pageType === 'loras' ? '.model-card' : '.checkpoint-card';
        document.querySelectorAll(cardSelector).forEach(card => {
            card.style.display = card.dataset.folder === folderPath ? '' : 'none';
        });
    }
    
    /**
     * Update the folder tags display with new folder list
     * @param {Array} folders - List of folder names
     */
    updateFolderTags(folders) {
        const folderTagsContainer = document.querySelector('.folder-tags');
        if (!folderTagsContainer) return;

        // Keep track of currently selected folder
        const currentFolder = this.pageState.activeFolder;

        // Create HTML for folder tags
        const tagsHTML = folders.map(folder => {
            const isActive = folder === currentFolder;
            return `<div class="tag ${isActive ? 'active' : ''}" data-folder="${folder}">${folder}</div>`;
        }).join('');

        // Update the container
        folderTagsContainer.innerHTML = tagsHTML;

        // Scroll active folder into view (no need to reattach click handlers)
        const activeTag = folderTagsContainer.querySelector(`.tag[data-folder="${currentFolder}"]`);
        if (activeTag) {
            activeTag.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }
    
    /**
     * Toggle visibility of folder tags
     */
    toggleFolderTags() {
        const folderTags = document.querySelector('.folder-tags');
        const toggleBtn = document.querySelector('.toggle-folders-btn i');
        
        if (folderTags) {
            folderTags.classList.toggle('collapsed');
            
            if (folderTags.classList.contains('collapsed')) {
                // Change icon to indicate folders are hidden
                toggleBtn.className = 'fas fa-folder-plus';
                toggleBtn.parentElement.title = 'Show folder tags';
                setStorageItem('folderTagsCollapsed', 'true');
            } else {
                // Change icon to indicate folders are visible
                toggleBtn.className = 'fas fa-folder-minus';
                toggleBtn.parentElement.title = 'Hide folder tags';
                setStorageItem('folderTagsCollapsed', 'false');
            }
        }
    }
    
    /**
     * Initialize folder tags visibility based on stored preference
     */
    initFolderTagsVisibility() {
        const isCollapsed = getStorageItem('folderTagsCollapsed');
        if (isCollapsed) {
            const folderTags = document.querySelector('.folder-tags');
            const toggleBtn = document.querySelector('.toggle-folders-btn i');
            if (folderTags) {
                folderTags.classList.add('collapsed');
            }
            if (toggleBtn) {
                toggleBtn.className = 'fas fa-folder-plus';
                toggleBtn.parentElement.title = 'Show folder tags';
            }
        } else {
            const toggleBtn = document.querySelector('.toggle-folders-btn i');
            if (toggleBtn) {
                toggleBtn.className = 'fas fa-folder-minus';
                toggleBtn.parentElement.title = 'Hide folder tags';
            }
        }
    }
    
    /**
     * Load sort preference from storage
     */
    loadSortPreference() {
        const savedSort = getStorageItem(`${this.pageType}_sort`);
        if (savedSort) {
            // Handle legacy format conversion
            const convertedSort = this.convertLegacySortFormat(savedSort);
            this.pageState.sortBy = convertedSort;
            const sortSelect = document.getElementById('sortSelect');
            if (sortSelect) {
                sortSelect.value = convertedSort;
            }
        }
    }
    
    /**
     * Convert legacy sort format to new format
     * @param {string} sortValue - The sort value to convert
     * @returns {string} - Converted sort value
     */
    convertLegacySortFormat(sortValue) {
        // Convert old format to new format with direction
        switch (sortValue) {
            case 'name':
                return 'name:asc';
            case 'date':
                return 'date:desc'; // Newest first is more intuitive default
            case 'size':
                return 'size:desc'; // Largest first is more intuitive default
            default:
                // If it's already in new format or unknown, return as is
                return sortValue.includes(':') ? sortValue : 'name:asc';
        }
    }
    
    /**
     * Save sort preference to storage
     * @param {string} sortValue - The sort value to save
     */
    saveSortPreference(sortValue) {
        setStorageItem(`${this.pageType}_sort`, sortValue);
    }
    
    /**
     * Open model page on Civitai
     * @param {string} modelName - Name of the model
     */
    openCivitai(modelName) {
        // Get card selector based on page type
        const cardSelector = this.pageType === 'loras' 
            ? `.model-card[data-name="${modelName}"]`
            : `.checkpoint-card[data-name="${modelName}"]`;
            
        const card = document.querySelector(cardSelector);
        if (!card) return;
        
        const metaData = JSON.parse(card.dataset.meta);
        const civitaiId = metaData.modelId;
        const versionId = metaData.id;
        
        // Build URL
        if (civitaiId) {
            let url = `https://civitai.com/models/${civitaiId}`;
            if (versionId) {
                url += `?modelVersionId=${versionId}`;
            }
            window.open(url, '_blank');
        } else {
            // If no ID, try searching by name
            window.open(`https://civitai.com/models?query=${encodeURIComponent(modelName)}`, '_blank');
        }
    }
    
    /**
     * Reset and reload the models list
     */
    async resetAndReload(updateFolders = false) {
        if (!this.api) {
            console.error('API methods not registered');
            return;
        }

        try {
            await this.api.resetAndReload(updateFolders);
        } catch (error) {
            console.error(`Error reloading ${this.pageType}:`, error);
            showToast(`Failed to reload ${this.pageType}: ${error.message}`, 'error');
        }
    }
    
    /**
     * Refresh models list
     * @param {boolean} fullRebuild - Whether to perform a full rebuild
     */
    async refreshModels(fullRebuild = false) {
        if (!this.api) {
            console.error('API methods not registered');
            return;
        }

        try {
            await this.api.refreshModels(fullRebuild);
        } catch (error) {
            console.error(`Error ${fullRebuild ? 'rebuilding' : 'refreshing'} ${this.pageType}:`, error);
            showToast(`Failed to ${fullRebuild ? 'rebuild' : 'refresh'} ${this.pageType}: ${error.message}`, 'error');
        }

        if (window.modelDuplicatesManager) {
            // Update duplicates badge after refresh
            window.modelDuplicatesManager.updateDuplicatesBadgeAfterRefresh();
        }
    }
    
    /**
     * Fetch metadata from Civitai (available for both LoRAs and Checkpoints)
     */
    async fetchFromCivitai() {
        if (!this.api) {
            console.error('API methods not registered');
            return;
        }
        
        try {
            await this.api.fetchFromCivitai();
        } catch (error) {
            console.error('Error fetching metadata:', error);
            showToast('Failed to fetch metadata: ' + error.message, 'error');
        }
    }
    
    /**
     * Show download modal
     */
    showDownloadModal() {
        this.api.showDownloadModal();
    }
    
    /**
     * Toggle bulk mode (LoRAs only)
     */
    toggleBulkMode() {
        if (this.pageType !== 'loras' || !this.api) {
            console.error('Bulk mode is only available for LoRAs');
            return;
        }
        
        this.api.toggleBulkMode();
    }
    
    /**
     * Clear custom filter
     */
    async clearCustomFilter() {
        if (!this.api) {
            console.error('API methods not registered');
            return;
        }
        
        try {
            await this.api.clearCustomFilter();
        } catch (error) {
            console.error('Error clearing custom filter:', error);
            showToast('Failed to clear custom filter: ' + error.message, 'error');
        }
    }
    
    /**
     * Initialize the favorites filter button state
     */
    initFavoritesFilter() {
        const favoriteFilterBtn = document.getElementById('favoriteFilterBtn');
        if (favoriteFilterBtn) {
            // Get current state from session storage with page-specific key
            const storageKey = `show_favorites_only_${this.pageType}`;
            const showFavoritesOnly = getSessionItem(storageKey, false);
            
            // Update button state
            if (showFavoritesOnly) {
                favoriteFilterBtn.classList.add('active');
            }
            
            // Update app state
            this.pageState.showFavoritesOnly = showFavoritesOnly;
        }
    }
    
    /**
     * Toggle favorites-only filter and reload models
     */
    async toggleFavoritesOnly() {
        const favoriteFilterBtn = document.getElementById('favoriteFilterBtn');
        
        // Toggle the filter state in storage
        const storageKey = `show_favorites_only_${this.pageType}`;
        const currentState = this.pageState.showFavoritesOnly;
        const newState = !currentState;
        
        // Update session storage
        setSessionItem(storageKey, newState);
        
        // Update state
        this.pageState.showFavoritesOnly = newState;
        
        // Update button appearance
        if (favoriteFilterBtn) {
            favoriteFilterBtn.classList.toggle('active', newState);
        }
        
        // Reload models with new filter
        await this.resetAndReload(true);
    }
    
    /**
     * Find duplicate models
     */
    findDuplicates() {
        if (window.modelDuplicatesManager) {
            // Change to toggle functionality
            window.modelDuplicatesManager.toggleDuplicateMode();
        } else {
            console.error('Model duplicates manager not available');
        }
    }
}