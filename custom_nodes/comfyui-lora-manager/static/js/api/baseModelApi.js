import { state, getCurrentPageState } from '../state/index.js';
import { showToast, updateFolderTags } from '../utils/uiHelpers.js';
import { getSessionItem, saveMapToStorage } from '../utils/storageHelpers.js';
import { 
    getCompleteApiConfig, 
    getCurrentModelType, 
    isValidModelType,
    DOWNLOAD_ENDPOINTS,
    WS_ENDPOINTS
} from './apiConfig.js';
import { resetAndReload } from './modelApiFactory.js';

/**
 * Abstract base class for all model API clients
 */
export class BaseModelApiClient {
    constructor(modelType = null) {
        if (this.constructor === BaseModelApiClient) {
            throw new Error("BaseModelApiClient is abstract and cannot be instantiated directly");
        }
        this.modelType = modelType || getCurrentModelType();
        this.apiConfig = getCompleteApiConfig(this.modelType);
    }

    /**
     * Set the model type for this client instance
     * @param {string} modelType - The model type to use
     */
    setModelType(modelType) {
        if (!isValidModelType(modelType)) {
            throw new Error(`Invalid model type: ${modelType}`);
        }
        this.modelType = modelType;
        this.apiConfig = getCompleteApiConfig(modelType);
    }

    /**
     * Get the current page state for this model type
     */
    getPageState() {
        const currentType = state.currentPageType;
        // Temporarily switch to get the right page state
        state.currentPageType = this.modelType;
        const pageState = getCurrentPageState();
        state.currentPageType = currentType; // Restore
        return pageState;
    }

    async fetchModelsPage(page = 1, pageSize = null) {
        const pageState = this.getPageState();
        const actualPageSize = pageSize || pageState.pageSize || this.apiConfig.config.defaultPageSize;
        
        try {
            const params = this._buildQueryParams({
                page,
                page_size: actualPageSize,
                sort_by: pageState.sortBy
            }, pageState);

            const response = await fetch(`${this.apiConfig.endpoints.list}?${params}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch ${this.apiConfig.config.displayName}s: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            return {
                items: data.items,
                totalItems: data.total,
                totalPages: data.total_pages,
                currentPage: page,
                hasMore: page < data.total_pages,
                folders: data.folders
            };
            
        } catch (error) {
            console.error(`Error fetching ${this.apiConfig.config.displayName}s:`, error);
            showToast(`Failed to fetch ${this.apiConfig.config.displayName}s: ${error.message}`, 'error');
            throw error;
        }
    }

    async loadMoreWithVirtualScroll(resetPage = false, updateFolders = false) {
        const pageState = this.getPageState();
        
        try {
            state.loadingManager.showSimpleLoading(`Loading more ${this.apiConfig.config.displayName}s...`);

            pageState.isLoading = true;
            if (resetPage) {
                pageState.currentPage = 1; // Reset to first page
            }
            
            const result = await this.fetchModelsPage(pageState.currentPage, pageState.pageSize);
            
            state.virtualScroller.refreshWithData(
                result.items,
                result.totalItems,
                result.hasMore
            );
            
            pageState.hasMore = result.hasMore;
            pageState.currentPage = pageState.currentPage + 1;
            
            if (updateFolders) {
                const response = await fetch(this.apiConfig.endpoints.folders);
                if (response.ok) {
                    const data = await response.json();
                    updateFolderTags(data.folders);
                } else {
                    const errorData = await response.json().catch(() => ({}));
                    const errorMsg = errorData && errorData.error ? errorData.error : response.statusText;
                    console.error(`Error getting folders: ${errorMsg}`);
                }
            }
            
            return result;
        } catch (error) {
            console.error(`Error reloading ${this.apiConfig.config.displayName}s:`, error);
            showToast(`Failed to reload ${this.apiConfig.config.displayName}s: ${error.message}`, 'error');
            throw error;
        } finally {
            pageState.isLoading = false;
            state.loadingManager.hide();
        }
    }

    async deleteModel(filePath) {
        try {
            state.loadingManager.showSimpleLoading(`Deleting ${this.apiConfig.config.singularName}...`);

            const response = await fetch(this.apiConfig.endpoints.delete, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_path: filePath })
            });
            
            if (!response.ok) {
                throw new Error(`Failed to delete ${this.apiConfig.config.singularName}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                if (state.virtualScroller) {
                    state.virtualScroller.removeItemByFilePath(filePath);
                }
                showToast(`${this.apiConfig.config.displayName} deleted successfully`, 'success');
                return true;
            } else {
                throw new Error(data.error || `Failed to delete ${this.apiConfig.config.singularName}`);
            }
        } catch (error) {
            console.error(`Error deleting ${this.apiConfig.config.singularName}:`, error);
            showToast(`Failed to delete ${this.apiConfig.config.singularName}: ${error.message}`, 'error');
            return false;
        } finally {
            state.loadingManager.hide();
        }
    }

    async excludeModel(filePath) {
        try {
            state.loadingManager.showSimpleLoading(`Excluding ${this.apiConfig.config.singularName}...`);

            const response = await fetch(this.apiConfig.endpoints.exclude, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_path: filePath })
            });
            
            if (!response.ok) {
                throw new Error(`Failed to exclude ${this.apiConfig.config.singularName}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                if (state.virtualScroller) {
                    state.virtualScroller.removeItemByFilePath(filePath);
                }
                showToast(`${this.apiConfig.config.displayName} excluded successfully`, 'success');
                return true;
            } else {
                throw new Error(data.error || `Failed to exclude ${this.apiConfig.config.singularName}`);
            }
        } catch (error) {
            console.error(`Error excluding ${this.apiConfig.config.singularName}:`, error);
            showToast(`Failed to exclude ${this.apiConfig.config.singularName}: ${error.message}`, 'error');
            return false;
        } finally {
            state.loadingManager.hide();
        }
    }

    async renameModelFile(filePath, newFileName) {
        try {
            state.loadingManager.showSimpleLoading(`Renaming ${this.apiConfig.config.singularName} file...`);
            
            const response = await fetch(this.apiConfig.endpoints.rename, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    file_path: filePath,
                    new_file_name: newFileName
                })
            });

            const result = await response.json();

            if (result.success) {
                state.virtualScroller.updateSingleItem(filePath, { 
                    file_name: newFileName, 
                    file_path: result.new_file_path,
                    preview_url: result.new_preview_path
                });
    
                showToast('File name updated successfully', 'success');
            } else {
                showToast('Failed to rename file: ' + (result.error || 'Unknown error'), 'error');
            }

            return result;
        } catch (error) {
            console.error(`Error renaming ${this.apiConfig.config.singularName} file:`, error);
            throw error;
        } finally {
            state.loadingManager.hide();
        }
    }

    replaceModelPreview(filePath) {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*,video/mp4';
        
        input.onchange = async () => {
            if (!input.files || !input.files[0]) return;
            
            const file = input.files[0];
            await this.uploadPreview(filePath, file);
        };
        
        input.click();
    }

    async uploadPreview(filePath, file, nsfwLevel = 0) {
        try {
            state.loadingManager.showSimpleLoading('Uploading preview...');
            
            const formData = new FormData();
            formData.append('preview_file', file);
            formData.append('model_path', filePath);
            formData.append('nsfw_level', nsfwLevel.toString());

            const response = await fetch(this.apiConfig.endpoints.replacePreview, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Upload failed');
            }

            const data = await response.json();
            const pageState = this.getPageState();
            
            const timestamp = Date.now();
            if (pageState.previewVersions) {
                pageState.previewVersions.set(filePath, timestamp);
                
                const storageKey = `${this.modelType}_preview_versions`;
                saveMapToStorage(storageKey, pageState.previewVersions);
            }

            const updateData = {
                preview_url: data.preview_url,
                preview_nsfw_level: data.preview_nsfw_level
            };

            state.virtualScroller.updateSingleItem(filePath, updateData);
            showToast('Preview updated successfully', 'success');
        } catch (error) {
            console.error('Error uploading preview:', error);
            showToast('Failed to upload preview image', 'error');
        } finally {
            state.loadingManager.hide();
        }
    }

    async saveModelMetadata(filePath, data) {
        try {
            state.loadingManager.showSimpleLoading('Saving metadata...');
            
            const response = await fetch(this.apiConfig.endpoints.save, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    file_path: filePath,
                    ...data
                })
            });

            if (!response.ok) {
                throw new Error('Failed to save metadata');
            }

            state.virtualScroller.updateSingleItem(filePath, data);
            return response.json();
        } finally {
            state.loadingManager.hide();
        }
    }

    async refreshModels(fullRebuild = false) {
        try {
            state.loadingManager.showSimpleLoading(
                `${fullRebuild ? 'Full rebuild' : 'Refreshing'} ${this.apiConfig.config.displayName}s...`
            );
            
            const url = new URL(this.apiConfig.endpoints.scan, window.location.origin);
            url.searchParams.append('full_rebuild', fullRebuild);
            
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`Failed to refresh ${this.apiConfig.config.displayName}s: ${response.status} ${response.statusText}`);
            }

            resetAndReload(true);
            
            showToast(`${fullRebuild ? 'Full rebuild' : 'Refresh'} complete`, 'success');
        } catch (error) {
            console.error('Refresh failed:', error);
            showToast(`Failed to ${fullRebuild ? 'rebuild' : 'refresh'} ${this.apiConfig.config.displayName}s`, 'error');
        } finally {
            state.loadingManager.hide();
            state.loadingManager.restoreProgressBar();
        }
    }

    async refreshSingleModelMetadata(filePath) {
        try {
            state.loadingManager.showSimpleLoading('Refreshing metadata...');
            
            const response = await fetch(this.apiConfig.endpoints.fetchCivitai, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_path: filePath })
            });

            if (!response.ok) {
                throw new Error('Failed to refresh metadata');
            }

            const data = await response.json();
            
            if (data.success) {
                if (data.metadata && state.virtualScroller) {
                    state.virtualScroller.updateSingleItem(filePath, data.metadata);
                }

                showToast('Metadata refreshed successfully', 'success');
                return true;
            } else {
                throw new Error(data.error || 'Failed to refresh metadata');
            }
        } catch (error) {
            console.error('Error refreshing metadata:', error);
            showToast(error.message, 'error');
            return false;
        } finally {
            state.loadingManager.hide();
            state.loadingManager.restoreProgressBar();
        }
    }

    async fetchCivitaiMetadata() {
        let ws = null;
        
        await state.loadingManager.showWithProgress(async (loading) => {
            try {
                const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
                ws = new WebSocket(`${wsProtocol}${window.location.host}${WS_ENDPOINTS.fetchProgress}`);
                
                const operationComplete = new Promise((resolve, reject) => {
                    ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        
                        switch(data.status) {
                            case 'started':
                                loading.setStatus('Starting metadata fetch...');
                                break;
                                
                            case 'processing':
                                const percent = ((data.processed / data.total) * 100).toFixed(1);
                                loading.setProgress(percent);
                                loading.setStatus(
                                    `Processing (${data.processed}/${data.total}) ${data.current_name}`
                                );
                                break;
                                
                            case 'completed':
                                loading.setProgress(100);
                                loading.setStatus(
                                    `Completed: Updated ${data.success} of ${data.processed} ${this.apiConfig.config.displayName}s`
                                );
                                resolve();
                                break;
                                
                            case 'error':
                                reject(new Error(data.error));
                                break;
                        }
                    };
                    
                    ws.onerror = (error) => {
                        reject(new Error('WebSocket error: ' + error.message));
                    };
                });
                
                await new Promise((resolve, reject) => {
                    ws.onopen = resolve;
                    ws.onerror = reject;
                });
                
                const response = await fetch(this.apiConfig.endpoints.fetchAllCivitai, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                
                if (!response.ok) {
                    throw new Error('Failed to fetch metadata');
                }
                
                await operationComplete;
                
            } catch (error) {
                console.error('Error fetching metadata:', error);
                showToast('Failed to fetch metadata: ' + error.message, 'error');
            } finally {
                if (ws) {
                    ws.close();
                }
            }
        }, {
            initialMessage: 'Connecting...',
            completionMessage: 'Metadata update complete'
        });
    }

    async refreshBulkModelMetadata(filePaths) {
        if (!filePaths || filePaths.length === 0) {
            throw new Error('No file paths provided');
        }

        const totalItems = filePaths.length;
        let processedCount = 0;
        let successCount = 0;
        let failedItems = [];

        const progressController = state.loadingManager.showEnhancedProgress('Starting metadata refresh...');

        try {
            for (let i = 0; i < filePaths.length; i++) {
                const filePath = filePaths[i];
                const fileName = filePath.split('/').pop();
                
                try {
                    const overallProgress = Math.floor((i / totalItems) * 100);
                    progressController.updateProgress(
                        overallProgress, 
                        fileName, 
                        `Processing ${i + 1}/${totalItems}: ${fileName}`
                    );
                    
                    const response = await fetch(this.apiConfig.endpoints.fetchCivitai, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ file_path: filePath })
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }

                    const data = await response.json();
                    
                    if (data.success) {
                        if (data.metadata && state.virtualScroller) {
                            state.virtualScroller.updateSingleItem(filePath, data.metadata);
                        }
                        successCount++;
                    } else {
                        throw new Error(data.error || 'Failed to refresh metadata');
                    }
                    
                } catch (error) {
                    console.error(`Error refreshing metadata for ${fileName}:`, error);
                    failedItems.push({ filePath, fileName, error: error.message });
                }
                
                processedCount++;
            }

            let completionMessage;
            if (successCount === totalItems) {
                completionMessage = `Successfully refreshed all ${successCount} ${this.apiConfig.config.displayName}s`;
                showToast(completionMessage, 'success');
            } else if (successCount > 0) {
                completionMessage = `Refreshed ${successCount} of ${totalItems} ${this.apiConfig.config.displayName}s`;
                showToast(completionMessage, 'warning');
                
                if (failedItems.length > 0) {
                    const failureMessage = failedItems.length <= 3 
                        ? failedItems.map(item => `${item.fileName}: ${item.error}`).join('\n')
                        : failedItems.slice(0, 3).map(item => `${item.fileName}: ${item.error}`).join('\n') + 
                          `\n(and ${failedItems.length - 3} more)`;
                    showToast(`Failed refreshes:\n${failureMessage}`, 'warning', 6000);
                }
            } else {
                completionMessage = `Failed to refresh metadata for any ${this.apiConfig.config.displayName}s`;
                showToast(completionMessage, 'error');
            }

            await progressController.complete(completionMessage);

            return {
                success: successCount > 0,
                total: totalItems,
                processed: processedCount,
                successful: successCount,
                failed: failedItems.length,
                errors: failedItems
            };

        } catch (error) {
            console.error('Error in bulk metadata refresh:', error);
            showToast(`Failed to refresh metadata: ${error.message}`, 'error');
            await progressController.complete('Operation failed');
            throw error;
        }
    }

    async fetchCivitaiVersions(modelId) {
        try {
            const response = await fetch(`${this.apiConfig.endpoints.civitaiVersions}/${modelId}`);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                if (errorData && errorData.error && errorData.error.includes('Model type mismatch')) {
                    throw new Error(`This model is not a ${this.apiConfig.config.displayName}. Please switch to the appropriate page to download this model type.`);
                }
                throw new Error('Failed to fetch model versions');
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching Civitai versions:', error);
            throw error;
        }
    }

    async fetchModelRoots() {
        try {
            const response = await fetch(this.apiConfig.endpoints.roots);
            if (!response.ok) {
                throw new Error(`Failed to fetch ${this.apiConfig.config.displayName} roots`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching model roots:', error);
            throw error;
        }
    }

    async fetchModelFolders() {
        try {
            const response = await fetch(this.apiConfig.endpoints.folders);
            if (!response.ok) {
                throw new Error(`Failed to fetch ${this.apiConfig.config.displayName} folders`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching model folders:', error);
            throw error;
        }
    }

    async fetchUnifiedFolderTree() {
        try {
            const response = await fetch(this.apiConfig.endpoints.unifiedFolderTree);
            if (!response.ok) {
                throw new Error(`Failed to fetch unified folder tree`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching unified folder tree:', error);
            throw error;
        }
    }

    async fetchFolderTree(modelRoot) {
        try {
            const params = new URLSearchParams({ model_root: modelRoot });
            const response = await fetch(`${this.apiConfig.endpoints.folderTree}?${params}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch folder tree for root: ${modelRoot}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching folder tree:', error);
            throw error;
        }
    }

    async downloadModel(modelId, versionId, modelRoot, relativePath, useDefaultPaths = false, downloadId) {
        try {
            const response = await fetch(DOWNLOAD_ENDPOINTS.download, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_id: modelId,
                    model_version_id: versionId,
                    model_root: modelRoot,
                    relative_path: relativePath,
                    use_default_paths: useDefaultPaths,
                    download_id: downloadId
                })
            });

            if (!response.ok) {
                throw new Error(await response.text());
            }

            return await response.json();
        } catch (error) {
            console.error('Error downloading model:', error);
            throw error;
        }
    }

    _buildQueryParams(baseParams, pageState) {
        const params = new URLSearchParams(baseParams);
        
        if (pageState.activeFolder !== null) {
            params.append('folder', pageState.activeFolder);
        }

        if (pageState.showFavoritesOnly) {
            params.append('favorites_only', 'true');
        }
        
        if (this.apiConfig.config.supportsLetterFilter && pageState.activeLetterFilter) {
            params.append('first_letter', pageState.activeLetterFilter);
        }

        if (pageState.filters?.search) {
            params.append('search', pageState.filters.search);
            params.append('fuzzy', 'true');
            
            if (pageState.searchOptions) {
                params.append('search_filename', pageState.searchOptions.filename.toString());
                params.append('search_modelname', pageState.searchOptions.modelname.toString());
                if (pageState.searchOptions.tags !== undefined) {
                    params.append('search_tags', pageState.searchOptions.tags.toString());
                }
                if (pageState.searchOptions.creator !== undefined) {
                    params.append('search_creator', pageState.searchOptions.creator.toString());
                }
                params.append('recursive', (pageState.searchOptions?.recursive ?? false).toString());
            }
        }
        
        if (pageState.filters) {
            if (pageState.filters.tags && pageState.filters.tags.length > 0) {
                pageState.filters.tags.forEach(tag => {
                    params.append('tag', tag);
                });
            }
            
            if (pageState.filters.baseModel && pageState.filters.baseModel.length > 0) {
                pageState.filters.baseModel.forEach(model => {
                    params.append('base_model', model);
                });
            }
        }

        this._addModelSpecificParams(params, pageState);

        return params;
    }

    _addModelSpecificParams(params, pageState) {
        if (this.modelType === 'loras') {
            const filterLoraHash = getSessionItem('recipe_to_lora_filterLoraHash');
            const filterLoraHashes = getSessionItem('recipe_to_lora_filterLoraHashes');

            if (filterLoraHash) {
                params.append('lora_hash', filterLoraHash);
            } else if (filterLoraHashes) {
                try {
                    if (Array.isArray(filterLoraHashes) && filterLoraHashes.length > 0) {
                        params.append('lora_hashes', filterLoraHashes.join(','));
                    }
                } catch (error) {
                    console.error('Error parsing lora hashes from session storage:', error);
                }
            }
        }
    }

    async moveSingleModel(filePath, targetPath) {
        // Only allow move if supported
        if (!this.apiConfig.config.supportsMove) {
            showToast(`Moving ${this.apiConfig.config.displayName}s is not supported`, 'warning');
            return null;
        }
        if (filePath.substring(0, filePath.lastIndexOf('/')) === targetPath) {
            showToast(`${this.apiConfig.config.displayName} is already in the selected folder`, 'info');
            return null;
        }

        const response = await fetch(this.apiConfig.endpoints.moveModel, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                file_path: filePath,
                target_path: targetPath
            })
        });

        const result = await response.json();

        if (!response.ok) {
            if (result && result.error) {
                throw new Error(result.error);
            }
            throw new Error(`Failed to move ${this.apiConfig.config.displayName}`);
        }

        if (result && result.message) {
            showToast(result.message, 'info');
        } else {
            showToast(`${this.apiConfig.config.displayName} moved successfully`, 'success');
        }

        if (result.success) {
            return result.new_file_path;
        }
        return null;
    }

    async moveBulkModels(filePaths, targetPath) {
        if (!this.apiConfig.config.supportsMove) {
            showToast(`Moving ${this.apiConfig.config.displayName}s is not supported`, 'warning');
            return [];
        }
        const movedPaths = filePaths.filter(path => {
            return path.substring(0, path.lastIndexOf('/')) !== targetPath;
        });

        if (movedPaths.length === 0) {
            showToast(`All selected ${this.apiConfig.config.displayName}s are already in the target folder`, 'info');
            return [];
        }

        const response = await fetch(this.apiConfig.endpoints.moveBulk, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                file_paths: movedPaths,
                target_path: targetPath
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(`Failed to move ${this.apiConfig.config.displayName}s`);
        }

        let successFilePaths = [];
        if (result.success) {
            if (result.failure_count > 0) {
                showToast(`Moved ${result.success_count} ${this.apiConfig.config.displayName}s, ${result.failure_count} failed`, 'warning');
                console.log('Move operation results:', result.results);
                const failedFiles = result.results
                    .filter(r => !r.success)
                    .map(r => {
                        const fileName = r.path.substring(r.path.lastIndexOf('/') + 1);
                        return `${fileName}: ${r.message}`;
                    });
                if (failedFiles.length > 0) {
                    const failureMessage = failedFiles.length <= 3 
                        ? failedFiles.join('\n')
                        : failedFiles.slice(0, 3).join('\n') + `\n(and ${failedFiles.length - 3} more)`;
                    showToast(`Failed moves:\n${failureMessage}`, 'warning', 6000);
                }
            } else {
                showToast(`Successfully moved ${result.success_count} ${this.apiConfig.config.displayName}s`, 'success');
            }
            successFilePaths = result.results
                .filter(r => r.success)
                .map(r => r.path);
        } else {
            throw new Error(result.message || `Failed to move ${this.apiConfig.config.displayName}s`);
        }
        return successFilePaths;
    }

    async bulkDeleteModels(filePaths) {
        if (!filePaths || filePaths.length === 0) {
            throw new Error('No file paths provided');
        }

        try {
            state.loadingManager.showSimpleLoading(`Deleting ${this.apiConfig.config.displayName.toLowerCase()}s...`);
            
            const response = await fetch(this.apiConfig.endpoints.bulkDelete, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    file_paths: filePaths
                })
            });
            
            if (!response.ok) {
                throw new Error(`Failed to delete ${this.apiConfig.config.displayName.toLowerCase()}s: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                return {
                    success: true,
                    deleted_count: result.deleted_count,
                    failed_count: result.failed_count || 0,
                    errors: result.errors || []
                };
            } else {
                throw new Error(result.error || `Failed to delete ${this.apiConfig.config.displayName.toLowerCase()}s`);
            }
        } catch (error) {
            console.error(`Error during bulk delete of ${this.apiConfig.config.displayName.toLowerCase()}s:`, error);
            throw error;
        } finally {
            state.loadingManager.hide();
        }
    }
}