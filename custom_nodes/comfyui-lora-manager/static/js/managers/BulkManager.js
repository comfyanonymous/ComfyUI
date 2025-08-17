import { state, getCurrentPageState } from '../state/index.js';
import { showToast, copyToClipboard, sendLoraToWorkflow } from '../utils/uiHelpers.js';
import { updateCardsForBulkMode } from '../components/shared/ModelCard.js';
import { modalManager } from './ModalManager.js';
import { moveManager } from './MoveManager.js';
import { getModelApiClient } from '../api/modelApiFactory.js';
import { MODEL_TYPES, MODEL_CONFIG } from '../api/apiConfig.js';

export class BulkManager {
    constructor() {
        this.bulkBtn = document.getElementById('bulkOperationsBtn');
        this.bulkPanel = document.getElementById('bulkOperationsPanel');
        this.isStripVisible = false;
        
        this.stripMaxThumbnails = 50;
        
        // Model type specific action configurations
        this.actionConfig = {
            [MODEL_TYPES.LORA]: {
                sendToWorkflow: true,
                copyAll: true,
                refreshAll: true,
                moveAll: true,
                deleteAll: true
            },
            [MODEL_TYPES.EMBEDDING]: {
                sendToWorkflow: false,
                copyAll: false,
                refreshAll: true,
                moveAll: true,
                deleteAll: true
            },
            [MODEL_TYPES.CHECKPOINT]: {
                sendToWorkflow: false,
                copyAll: false,
                refreshAll: true,
                moveAll: false,
                deleteAll: true
            }
        };
    }

    initialize() {
        this.setupEventListeners();
        this.setupGlobalKeyboardListeners();
    }

    setupEventListeners() {
        // Bulk operations button listeners
        const sendToWorkflowBtn = this.bulkPanel?.querySelector('[data-action="send-to-workflow"]');
        const copyAllBtn = this.bulkPanel?.querySelector('[data-action="copy-all"]');
        const refreshAllBtn = this.bulkPanel?.querySelector('[data-action="refresh-all"]');
        const moveAllBtn = this.bulkPanel?.querySelector('[data-action="move-all"]');
        const deleteAllBtn = this.bulkPanel?.querySelector('[data-action="delete-all"]');
        const clearBtn = this.bulkPanel?.querySelector('[data-action="clear"]');

        if (sendToWorkflowBtn) {
            sendToWorkflowBtn.addEventListener('click', () => this.sendAllModelsToWorkflow());
        }
        if (copyAllBtn) {
            copyAllBtn.addEventListener('click', () => this.copyAllModelsSyntax());
        }
        if (refreshAllBtn) {
            refreshAllBtn.addEventListener('click', () => this.refreshAllMetadata());
        }
        if (moveAllBtn) {
            moveAllBtn.addEventListener('click', () => {
                moveManager.showMoveModal('bulk');
            });
        }
        if (deleteAllBtn) {
            deleteAllBtn.addEventListener('click', () => this.showBulkDeleteModal());
        }
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearSelection());
        }

        // Selected count click listener
        const selectedCount = document.getElementById('selectedCount');
        if (selectedCount) {
            selectedCount.addEventListener('click', () => this.toggleThumbnailStrip());
        }
    }

    setupGlobalKeyboardListeners() {
        document.addEventListener('keydown', (e) => {
            if (modalManager.isAnyModalOpen()) {
                return;
            }

            const searchInput = document.getElementById('searchInput');
            if (searchInput && document.activeElement === searchInput) {
                return;
            }

            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'a') {
                e.preventDefault();
                if (!state.bulkMode) {
                    this.toggleBulkMode();
                    setTimeout(() => this.selectAllVisibleModels(), 50);
                } else {
                    this.selectAllVisibleModels();
                }
            } else if (e.key === 'Escape' && state.bulkMode) {
                this.toggleBulkMode();
            } else if (e.key.toLowerCase() === 'b') {
                this.toggleBulkMode();
            }
        });
    }

    toggleBulkMode() {
        state.bulkMode = !state.bulkMode;
        
        this.bulkBtn.classList.toggle('active', state.bulkMode);
        
        if (state.bulkMode) {
            this.bulkPanel.classList.remove('hidden');
            this.updateActionButtonsVisibility();
            setTimeout(() => {
                this.bulkPanel.classList.add('visible');
            }, 10);
        } else {
            this.bulkPanel.classList.remove('visible');
            setTimeout(() => {
                this.bulkPanel.classList.add('hidden');
            }, 400);
            this.hideThumbnailStrip();
        }
        
        updateCardsForBulkMode(state.bulkMode);
        
        if (!state.bulkMode) {
            this.clearSelection();
            
            // TODO:
            document.querySelectorAll('.model-card').forEach(card => {
                const actions = card.querySelectorAll('.card-actions, .card-button');
                actions.forEach(action => action.style.display = 'flex');
            });
        }
    }

    updateActionButtonsVisibility() {
        const currentModelType = state.currentPageType;
        const config = this.actionConfig[currentModelType];
        
        if (!config) return;

        // Update button visibility based on model type
        const sendToWorkflowBtn = this.bulkPanel?.querySelector('[data-action="send-to-workflow"]');
        const copyAllBtn = this.bulkPanel?.querySelector('[data-action="copy-all"]');
        const refreshAllBtn = this.bulkPanel?.querySelector('[data-action="refresh-all"]');
        const moveAllBtn = this.bulkPanel?.querySelector('[data-action="move-all"]');
        const deleteAllBtn = this.bulkPanel?.querySelector('[data-action="delete-all"]');

        if (sendToWorkflowBtn) {
            sendToWorkflowBtn.style.display = config.sendToWorkflow ? 'block' : 'none';
        }
        if (copyAllBtn) {
            copyAllBtn.style.display = config.copyAll ? 'block' : 'none';
        }
        if (refreshAllBtn) {
            refreshAllBtn.style.display = config.refreshAll ? 'block' : 'none';
        }
        if (moveAllBtn) {
            moveAllBtn.style.display = config.moveAll ? 'block' : 'none';
        }
        if (deleteAllBtn) {
            deleteAllBtn.style.display = config.deleteAll ? 'block' : 'none';
        }
    }

    clearSelection() {
        document.querySelectorAll('.model-card.selected').forEach(card => {
            card.classList.remove('selected');
        });
        state.selectedModels.clear();
        this.updateSelectedCount();
        this.hideThumbnailStrip();
    }

    updateSelectedCount() {
        const countElement = document.getElementById('selectedCount');
        const currentConfig = MODEL_CONFIG[state.currentPageType];
        const displayName = currentConfig?.displayName || 'Models';
        
        if (countElement) {
            countElement.textContent = `${state.selectedModels.size} ${displayName.toLowerCase()}(s) selected `;
            
            const existingCaret = countElement.querySelector('.dropdown-caret');
            if (existingCaret) {
                existingCaret.className = `fas fa-caret-${this.isStripVisible ? 'down' : 'up'} dropdown-caret`;
                existingCaret.style.visibility = state.selectedModels.size > 0 ? 'visible' : 'hidden';
            } else {
                const caretIcon = document.createElement('i');
                caretIcon.className = `fas fa-caret-${this.isStripVisible ? 'down' : 'up'} dropdown-caret`;
                caretIcon.style.visibility = state.selectedModels.size > 0 ? 'visible' : 'hidden';
                countElement.appendChild(caretIcon);
            }
        }
    }

    toggleCardSelection(card) {
        const filepath = card.dataset.filepath;
        const pageState = getCurrentPageState();
        
        if (card.classList.contains('selected')) {
            card.classList.remove('selected');
            state.selectedModels.delete(filepath);
        } else {
            card.classList.add('selected');
            state.selectedModels.add(filepath);
            
            // Cache the metadata for this model
            const metadataCache = this.getMetadataCache();
            metadataCache.set(filepath, {
                fileName: card.dataset.file_name,
                usageTips: card.dataset.usage_tips,
                previewUrl: this.getCardPreviewUrl(card),
                isVideo: this.isCardPreviewVideo(card),
                modelName: card.dataset.name
            });
        }
        
        this.updateSelectedCount();
        
        if (this.isStripVisible) {
            this.updateThumbnailStrip();
        }
    }

    getMetadataCache() {
        const currentType = state.currentPageType;
        const pageState = getCurrentPageState();
        
        // Initialize metadata cache if it doesn't exist
        if (currentType === MODEL_TYPES.LORA) {
            if (!state.loraMetadataCache) {
                state.loraMetadataCache = new Map();
            }
            return state.loraMetadataCache;
        } else {
            if (!pageState.metadataCache) {
                pageState.metadataCache = new Map();
            }
            return pageState.metadataCache;
        }
    }
    
    getCardPreviewUrl(card) {
        const img = card.querySelector('img');
        const video = card.querySelector('video source');
        return img ? img.src : (video ? video.src : '/loras_static/images/no-preview.png');
    }
    
    isCardPreviewVideo(card) {
        return card.querySelector('video') !== null;
    }

    applySelectionState() {
        if (!state.bulkMode) return;
        
        document.querySelectorAll('.model-card').forEach(card => {
            const filepath = card.dataset.filepath;
            if (state.selectedModels.has(filepath)) {
                card.classList.add('selected');
                
                const metadataCache = this.getMetadataCache();
                metadataCache.set(filepath, {
                    fileName: card.dataset.file_name,
                    usageTips: card.dataset.usage_tips,
                    previewUrl: this.getCardPreviewUrl(card),
                    isVideo: this.isCardPreviewVideo(card),
                    modelName: card.dataset.name
                });
            } else {
                card.classList.remove('selected');
            }
        });
        
        this.updateSelectedCount();
    }

    async copyAllModelsSyntax() {
        if (state.currentPageType !== MODEL_TYPES.LORA) {
            showToast('Copy syntax is only available for LoRAs', 'warning');
            return;
        }
        
        if (state.selectedModels.size === 0) {
            showToast('No LoRAs selected', 'warning');
            return;
        }
        
        const loraSyntaxes = [];
        const missingLoras = [];
        const metadataCache = this.getMetadataCache();
        
        for (const filepath of state.selectedModels) {
            const metadata = metadataCache.get(filepath);
            
            if (metadata) {
                const usageTips = JSON.parse(metadata.usageTips || '{}');
                const strength = usageTips.strength || 1;
                loraSyntaxes.push(`<lora:${metadata.fileName}:${strength}>`);
            } else {
                missingLoras.push(filepath);
            }
        }
        
        if (missingLoras.length > 0) {
            console.warn('Missing metadata for some selected loras:', missingLoras);
            showToast(`Missing data for ${missingLoras.length} LoRAs`, 'warning');
        }
        
        if (loraSyntaxes.length === 0) {
            showToast('No valid LoRAs to copy', 'error');
            return;
        }
        
        await copyToClipboard(loraSyntaxes.join(', '), `Copied ${loraSyntaxes.length} LoRA syntaxes to clipboard`);
    }
    
    async sendAllModelsToWorkflow() {
        if (state.currentPageType !== MODEL_TYPES.LORA) {
            showToast('Send to workflow is only available for LoRAs', 'warning');
            return;
        }
        
        if (state.selectedModels.size === 0) {
            showToast('No LoRAs selected', 'warning');
            return;
        }
        
        const loraSyntaxes = [];
        const missingLoras = [];
        const metadataCache = this.getMetadataCache();
        
        for (const filepath of state.selectedModels) {
            const metadata = metadataCache.get(filepath);
            
            if (metadata) {
                const usageTips = JSON.parse(metadata.usageTips || '{}');
                const strength = usageTips.strength || 1;
                loraSyntaxes.push(`<lora:${metadata.fileName}:${strength}>`);
            } else {
                missingLoras.push(filepath);
            }
        }
        
        if (missingLoras.length > 0) {
            console.warn('Missing metadata for some selected loras:', missingLoras);
            showToast(`Missing data for ${missingLoras.length} LoRAs`, 'warning');
        }
        
        if (loraSyntaxes.length === 0) {
            showToast('No valid LoRAs to send', 'error');
            return;
        }
        
        await sendLoraToWorkflow(loraSyntaxes.join(', '), false, 'lora');
    }
    
    showBulkDeleteModal() {
        if (state.selectedModels.size === 0) {
            showToast('No models selected', 'warning');
            return;
        }
        
        const countElement = document.getElementById('bulkDeleteCount');
        if (countElement) {
            countElement.textContent = state.selectedModels.size;
        }
        
        modalManager.showModal('bulkDeleteModal');
    }
    
    async confirmBulkDelete() {
        if (state.selectedModels.size === 0) {
            showToast('No models selected', 'warning');
            modalManager.closeModal('bulkDeleteModal');
            return;
        }
        
        modalManager.closeModal('bulkDeleteModal');
        
        try {
            const apiClient = getModelApiClient();
            const filePaths = Array.from(state.selectedModels);
            
            const result = await apiClient.bulkDeleteModels(filePaths);
            
            if (result.success) {
                const currentConfig = MODEL_CONFIG[state.currentPageType];
                showToast(`Successfully deleted ${result.deleted_count} ${currentConfig.displayName.toLowerCase()}(s)`, 'success');
                
                filePaths.forEach(path => {
                    state.virtualScroller.removeItemByFilePath(path);
                });
                this.clearSelection();

                if (window.modelDuplicatesManager) {
                    window.modelDuplicatesManager.updateDuplicatesBadgeAfterRefresh();
                }
            } else {
                showToast(`Error: ${result.error || 'Failed to delete models'}`, 'error');
            }
        } catch (error) {
            console.error('Error during bulk delete:', error);
            showToast('Failed to delete models', 'error');
        }
    }

    toggleThumbnailStrip() {
        if (state.selectedModels.size === 0) return;
        
        const existing = document.querySelector('.selected-thumbnails-strip');
        if (existing) {
            this.hideThumbnailStrip();
        } else {
            this.showThumbnailStrip();
        }
    }
    
    showThumbnailStrip() {
        const strip = document.createElement('div');
        strip.className = 'selected-thumbnails-strip';
        
        const thumbnailContainer = document.createElement('div');
        thumbnailContainer.className = 'thumbnails-container';
        strip.appendChild(thumbnailContainer);
        
        this.bulkPanel.parentNode.insertBefore(strip, this.bulkPanel);
        
        this.updateThumbnailStrip();
        
        this.isStripVisible = true;
        this.updateSelectedCount();
        
        setTimeout(() => strip.classList.add('visible'), 10);
    }
    
    hideThumbnailStrip() {
        const strip = document.querySelector('.selected-thumbnails-strip');
        if (strip && this.isStripVisible) {
            strip.classList.remove('visible');
            
            this.isStripVisible = false;
            
            const countElement = document.getElementById('selectedCount');
            if (countElement) {
                const caret = countElement.querySelector('.dropdown-caret');
                if (caret) {
                    caret.className = 'fas fa-caret-up dropdown-caret';
                }
            }
            
            setTimeout(() => {
                if (strip.parentNode) {
                    strip.parentNode.removeChild(strip);
                }
            }, 300);
        }
    }
    
    updateThumbnailStrip() {
        const container = document.querySelector('.thumbnails-container');
        if (!container) return;
        
        container.innerHTML = '';
        
        const selectedModels = Array.from(state.selectedModels);
        
        if (selectedModels.length > this.stripMaxThumbnails) {
            const counter = document.createElement('div');
            counter.className = 'strip-counter';
            counter.textContent = `Showing ${this.stripMaxThumbnails} of ${selectedModels.length} selected`;
            container.appendChild(counter);
        }
        
        const thumbnailsToShow = selectedModels.slice(0, this.stripMaxThumbnails);
        const metadataCache = this.getMetadataCache();
        
        thumbnailsToShow.forEach(filepath => {
            const metadata = metadataCache.get(filepath);
            if (!metadata) return;
            
            const thumbnail = document.createElement('div');
            thumbnail.className = 'selected-thumbnail';
            thumbnail.dataset.filepath = filepath;
            
            if (metadata.isVideo) {
                thumbnail.innerHTML = `
                    <video autoplay loop muted playsinline>
                        <source src="${metadata.previewUrl}" type="video/mp4">
                    </video>
                    <span class="thumbnail-name" title="${metadata.modelName}">${metadata.modelName}</span>
                    <button class="thumbnail-remove"><i class="fas fa-times"></i></button>
                `;
            } else {
                thumbnail.innerHTML = `
                    <img src="${metadata.previewUrl}" alt="${metadata.modelName}">
                    <span class="thumbnail-name" title="${metadata.modelName}">${metadata.modelName}</span>
                    <button class="thumbnail-remove"><i class="fas fa-times"></i></button>
                `;
            }
            
            thumbnail.addEventListener('click', (e) => {
                if (!e.target.closest('.thumbnail-remove')) {
                    this.deselectItem(filepath);
                }
            });
            
            thumbnail.querySelector('.thumbnail-remove').addEventListener('click', (e) => {
                e.stopPropagation();
                this.deselectItem(filepath);
            });
            
            container.appendChild(thumbnail);
        });
    }
    
    deselectItem(filepath) {
        const card = document.querySelector(`.model-card[data-filepath="${filepath}"]`);
        if (card) {
            card.classList.remove('selected');
        }
        
        state.selectedModels.delete(filepath);
        
        this.updateSelectedCount();
        this.updateThumbnailStrip();
        
        if (state.selectedModels.size === 0) {
            this.hideThumbnailStrip();
        }
    }

    selectAllVisibleModels() {
        if (!state.virtualScroller || !state.virtualScroller.items) {
            showToast('Unable to select all items', 'error');
            return;
        }
        
        const oldCount = state.selectedModels.size;
        const metadataCache = this.getMetadataCache();
        
        state.virtualScroller.items.forEach(item => {
            if (item && item.file_path) {
                state.selectedModels.add(item.file_path);
                
                if (!metadataCache.has(item.file_path)) {
                    metadataCache.set(item.file_path, {
                        fileName: item.file_name,
                        usageTips: item.usage_tips || '{}',
                        previewUrl: item.preview_url || '/loras_static/images/no-preview.png',
                        isVideo: item.is_video || false,
                        modelName: item.name || item.file_name
                    });
                }
            }
        });
        
        this.applySelectionState();
        
        const newlySelected = state.selectedModels.size - oldCount;
        const currentConfig = MODEL_CONFIG[state.currentPageType];
        showToast(`Selected ${newlySelected} additional ${currentConfig.displayName.toLowerCase()}(s)`, 'success');
        
        if (this.isStripVisible) {
            this.updateThumbnailStrip();
        }
    }

    async refreshAllMetadata() {
        if (state.selectedModels.size === 0) {
            showToast('No models selected', 'warning');
            return;
        }
        
        try {
            const apiClient = getModelApiClient();
            const filePaths = Array.from(state.selectedModels);
            
            const result = await apiClient.refreshBulkModelMetadata(filePaths);
            
            if (result.success) {
                const metadataCache = this.getMetadataCache();
                for (const filepath of state.selectedModels) {
                    const metadata = metadataCache.get(filepath);
                    if (metadata) {
                        const card = document.querySelector(`.model-card[data-filepath="${filepath}"]`);
                        if (card) {
                            metadataCache.set(filepath, {
                                ...metadata,
                                fileName: card.dataset.file_name,
                                usageTips: card.dataset.usage_tips,
                                previewUrl: this.getCardPreviewUrl(card),
                                isVideo: this.isCardPreviewVideo(card),
                                modelName: card.dataset.name
                            });
                        }
                    }
                }
                
                if (this.isStripVisible) {
                    this.updateThumbnailStrip();
                }
            }
            
        } catch (error) {
            console.error('Error during bulk metadata refresh:', error);
            showToast('Failed to refresh metadata', 'error');
        }
    }
}

export const bulkManager = new BulkManager();
