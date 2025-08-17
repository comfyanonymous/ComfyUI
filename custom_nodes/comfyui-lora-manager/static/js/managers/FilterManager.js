import { getCurrentPageState } from '../state/index.js';
import { showToast, updatePanelPositions } from '../utils/uiHelpers.js';
import { getModelApiClient } from '../api/modelApiFactory.js';
import { removeStorageItem, setStorageItem, getStorageItem } from '../utils/storageHelpers.js';

export class FilterManager {
    constructor(options = {}) {
        this.options = {
            ...options
        };
        
        this.currentPage = options.page || document.body.dataset.page || 'loras';
        const pageState = getCurrentPageState();
        
        this.filters = pageState.filters || {
            baseModel: [],
            tags: []
        };
        
        this.filterPanel = document.getElementById('filterPanel');
        this.filterButton = document.getElementById('filterButton');
        this.activeFiltersCount = document.getElementById('activeFiltersCount');
        this.tagsLoaded = false;
        
        this.initialize();
        
        // Store this instance in the state
        if (pageState) {
            pageState.filterManager = this;
        }
    }
    
    initialize() {
        // Create base model filter tags if they exist
        if (document.getElementById('baseModelTags')) {
            this.createBaseModelTags();
        }

        // Add click handler for filter button
        if (this.filterButton) {
            this.filterButton.addEventListener('click', () => {
                this.toggleFilterPanel();
            });
        }
        
        // Close filter panel when clicking outside
        document.addEventListener('click', (e) => {
            if (this.filterPanel && !this.filterPanel.contains(e.target) && 
                e.target !== this.filterButton &&
                !this.filterButton.contains(e.target) &&
                !this.filterPanel.classList.contains('hidden')) {
                this.closeFilterPanel();
            }
        });
        
        // Initialize active filters from localStorage if available
        this.loadFiltersFromStorage();
    }
    
    async loadTopTags() {
        try {
            // Show loading state
            const tagsContainer = document.getElementById('modelTagsFilter');
            if (!tagsContainer) return;
            
            tagsContainer.innerHTML = '<div class="tags-loading">Loading tags...</div>';
            
            // Determine the API endpoint based on the page type
            const tagsEndpoint = `/api/${this.currentPage}/top-tags?limit=20`;

            const response = await fetch(tagsEndpoint);
            if (!response.ok) throw new Error('Failed to fetch tags');
            
            const data = await response.json();
            if (data.success && data.tags) {
                this.createTagFilterElements(data.tags);
                
                // After creating tag elements, mark any previously selected ones
                this.updateTagSelections();
            } else {
                throw new Error('Invalid response format');
            }
        } catch (error) {
            console.error('Error loading top tags:', error);
            const tagsContainer = document.getElementById('modelTagsFilter');
            if (tagsContainer) {
                tagsContainer.innerHTML = '<div class="tags-error">Failed to load tags</div>';
            }
        }
    }
    
    createTagFilterElements(tags) {
        const tagsContainer = document.getElementById('modelTagsFilter');
        if (!tagsContainer) return;
        
        tagsContainer.innerHTML = '';
        
        if (!tags.length) {
            tagsContainer.innerHTML = `<div class="no-tags">No ${this.currentPage === 'recipes' ? 'recipe ' : ''}tags available</div>`;
            return;
        }
        
        tags.forEach(tag => {
            const tagEl = document.createElement('div');
            tagEl.className = 'filter-tag tag-filter';
            const tagName = tag.tag;
            tagEl.dataset.tag = tagName;
            tagEl.innerHTML = `${tagName} <span class="tag-count">${tag.count}</span>`;
            
            // Add click handler to toggle selection and automatically apply
            tagEl.addEventListener('click', async () => {
                tagEl.classList.toggle('active');
                
                if (tagEl.classList.contains('active')) {
                    if (!this.filters.tags.includes(tagName)) {
                        this.filters.tags.push(tagName);
                    }
                } else {
                    this.filters.tags = this.filters.tags.filter(t => t !== tagName);
                }
                
                this.updateActiveFiltersCount();
                
                // Auto-apply filter when tag is clicked
                await this.applyFilters(false);
            });
            
            tagsContainer.appendChild(tagEl);
        });
    }
    
    createBaseModelTags() {
        const baseModelTagsContainer = document.getElementById('baseModelTags');
        if (!baseModelTagsContainer) return;
        
        // Set the API endpoint based on current page
        const apiEndpoint = `/api/${this.currentPage}/base-models`;
        
        // Fetch base models
        fetch(apiEndpoint)
            .then(response => response.json())
            .then(data => {
                if (data.success && data.base_models) {
                    baseModelTagsContainer.innerHTML = '';
                    
                    data.base_models.forEach(model => {
                        const tag = document.createElement('div');
                        tag.className = `filter-tag base-model-tag`;
                        tag.dataset.baseModel = model.name;
                        tag.innerHTML = `${model.name} <span class="tag-count">${model.count}</span>`;
                        
                        // Add click handler to toggle selection and automatically apply
                        tag.addEventListener('click', async () => {
                            tag.classList.toggle('active');
                            
                            if (tag.classList.contains('active')) {
                                if (!this.filters.baseModel.includes(model.name)) {
                                    this.filters.baseModel.push(model.name);
                                }
                            } else {
                                this.filters.baseModel = this.filters.baseModel.filter(m => m !== model.name);
                            }
                            
                            this.updateActiveFiltersCount();
                            
                            // Auto-apply filter when tag is clicked
                            await this.applyFilters(false);
                        });
                        
                        baseModelTagsContainer.appendChild(tag);
                    });
                    
                    // Update selections based on stored filters
                    this.updateTagSelections();
                }
            })
            .catch(error => {
                console.error(`Error fetching base models for ${this.currentPage}:`, error);
                baseModelTagsContainer.innerHTML = '<div class="tags-error">Failed to load base models</div>';
            });
    }
    
    toggleFilterPanel() {       
        if (this.filterPanel) {
            const isHidden = this.filterPanel.classList.contains('hidden');
            
            if (isHidden) {
                // Update panel positions before showing
                updatePanelPositions();
                
                this.filterPanel.classList.remove('hidden');
                this.filterButton.classList.add('active');
                
                // Load tags if they haven't been loaded yet
                if (!this.tagsLoaded) {
                    this.loadTopTags();
                    this.tagsLoaded = true;
                }
            } else {
                this.closeFilterPanel();
            }
        }
    }
    
    closeFilterPanel() {
        if (this.filterPanel) {
            this.filterPanel.classList.add('hidden');
        }
        if (this.filterButton) {
            this.filterButton.classList.remove('active');
        }
    }
    
    updateTagSelections() {
        // Update base model tags
        const baseModelTags = document.querySelectorAll('.base-model-tag');
        baseModelTags.forEach(tag => {
            const baseModel = tag.dataset.baseModel;
            if (this.filters.baseModel.includes(baseModel)) {
                tag.classList.add('active');
            } else {
                tag.classList.remove('active');
            }
        });
        
        // Update model tags
        const modelTags = document.querySelectorAll('.tag-filter');
        modelTags.forEach(tag => {
            const tagName = tag.dataset.tag;
            if (this.filters.tags.includes(tagName)) {
                tag.classList.add('active');
            } else {
                tag.classList.remove('active');
            }
        });
    }
    
    updateActiveFiltersCount() {
        const totalActiveFilters = this.filters.baseModel.length + this.filters.tags.length;
        
        if (this.activeFiltersCount) {
            if (totalActiveFilters > 0) {
                this.activeFiltersCount.textContent = totalActiveFilters;
                this.activeFiltersCount.style.display = 'inline-flex';
            } else {
                this.activeFiltersCount.style.display = 'none';
            }
        }
    }
    
    async applyFilters(showToastNotification = true) {
        const pageState = getCurrentPageState();
        const storageKey = `${this.currentPage}_filters`;
        
        // Save filters to localStorage
        setStorageItem(storageKey, this.filters);
        
        // Update state with current filters
        pageState.filters = { ...this.filters };
        
        // Call the appropriate manager's load method based on page type
        if (this.currentPage === 'recipes' && window.recipeManager) {
            await window.recipeManager.loadRecipes(true);
        } else if (this.currentPage === 'loras' || this.currentPage === 'embeddings' || this.currentPage === 'checkpoints') {
            // For models page, reset the page and reload
            await getModelApiClient().loadMoreWithVirtualScroll(true, false);
        }
        
        // Update filter button to show active state
        if (this.hasActiveFilters()) {
            this.filterButton.classList.add('active');
            if (showToastNotification) {
                const baseModelCount = this.filters.baseModel.length;
                const tagsCount = this.filters.tags.length;
                
                let message = '';
                if (baseModelCount > 0 && tagsCount > 0) {
                    message = `Filtering by ${baseModelCount} base model${baseModelCount > 1 ? 's' : ''} and ${tagsCount} tag${tagsCount > 1 ? 's' : ''}`;
                } else if (baseModelCount > 0) {
                    message = `Filtering by ${baseModelCount} base model${baseModelCount > 1 ? 's' : ''}`;
                } else if (tagsCount > 0) {
                    message = `Filtering by ${tagsCount} tag${tagsCount > 1 ? 's' : ''}`;
                }
                
                showToast(message, 'success');
            }
        } else {
            this.filterButton.classList.remove('active');
            if (showToastNotification) {
                showToast('Filters cleared', 'info');
            }
        }
    }
    
    async clearFilters() {
        // Clear all filters
        this.filters = {
            baseModel: [],
            tags: []
        };
        
        // Update state
        const pageState = getCurrentPageState();
        pageState.filters = { ...this.filters };
        
        // Update UI
        this.updateTagSelections();
        this.updateActiveFiltersCount();
        
        // Remove from local Storage
        const storageKey = `${this.currentPage}_filters`;
        removeStorageItem(storageKey);
        
        // Update UI
        this.filterButton.classList.remove('active');
        
        // Reload data using the appropriate method for the current page
        if (this.currentPage === 'recipes' && window.recipeManager) {
            await window.recipeManager.loadRecipes(true);
        } else if (this.currentPage === 'loras' || this.currentPage === 'checkpoints' || this.currentPage === 'embeddings') {
            await getModelApiClient().loadMoreWithVirtualScroll(true, true);
        }
        
        showToast(`Filters cleared`, 'info');
    }
    
    loadFiltersFromStorage() {
        const storageKey = `${this.currentPage}_filters`;
        const savedFilters = getStorageItem(storageKey);
        
        if (savedFilters) {
            try {
                // Ensure backward compatibility with older filter format
                this.filters = {
                    baseModel: savedFilters.baseModel || [],
                    tags: savedFilters.tags || []
                };
                
                // Update state with loaded filters
                const pageState = getCurrentPageState();
                pageState.filters = { ...this.filters };

                this.updateTagSelections();
                this.updateActiveFiltersCount();
                
                if (this.hasActiveFilters()) {
                    this.filterButton.classList.add('active');
                }
            } catch (error) {
                console.error(`Error loading ${this.currentPage} filters from storage:`, error);
            }
        }
    }
    
    hasActiveFilters() {
        return this.filters.baseModel.length > 0 || this.filters.tags.length > 0;
    }
}
