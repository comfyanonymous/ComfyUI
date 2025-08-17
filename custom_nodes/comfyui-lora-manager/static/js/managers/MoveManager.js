import { showToast, updateFolderTags } from '../utils/uiHelpers.js';
import { state, getCurrentPageState } from '../state/index.js';
import { modalManager } from './ModalManager.js';
import { bulkManager } from './BulkManager.js';
import { getStorageItem } from '../utils/storageHelpers.js';
import { getModelApiClient } from '../api/modelApiFactory.js';
import { FolderTreeManager } from '../components/FolderTreeManager.js';

class MoveManager {
    constructor() {
        this.currentFilePath = null;
        this.bulkFilePaths = null;
        this.folderTreeManager = new FolderTreeManager();
        this.initialized = false;
        
        // Bind methods
        this.updateTargetPath = this.updateTargetPath.bind(this);
    }

    initializeEventListeners() {
        if (this.initialized) return;
        
        const modelRootSelect = document.getElementById('moveModelRoot');
        
        // Initialize model root directory selector
        modelRootSelect.addEventListener('change', async () => {
            await this.initializeFolderTree();
            this.updateTargetPath();
        });
        
        this.initialized = true;
    }

    async showMoveModal(filePath, modelType = null) {
        // Reset state
        this.currentFilePath = null;
        this.bulkFilePaths = null;
        
        const apiClient = getModelApiClient();
        const currentPageType = state.currentPageType;
        const modelConfig = apiClient.apiConfig.config;
        
        // Handle bulk mode
        if (filePath === 'bulk') {
            const selectedPaths = Array.from(state.selectedModels);
            if (selectedPaths.length === 0) {
                showToast('No models selected', 'warning');
                return;
            }
            this.bulkFilePaths = selectedPaths;
            document.getElementById('moveModalTitle').textContent = `Move ${selectedPaths.length} ${modelConfig.displayName}s`;
        } else {
            // Single file mode
            this.currentFilePath = filePath;
            document.getElementById('moveModalTitle').textContent = `Move ${modelConfig.displayName}`;
        }
        
        // Update UI labels based on model type
        document.getElementById('moveRootLabel').textContent = `Select ${modelConfig.displayName} Root:`;
        document.getElementById('moveTargetPathDisplay').querySelector('.path-text').textContent = `Select a ${modelConfig.displayName.toLowerCase()} root directory`;
        
        // Clear folder path input
        const folderPathInput = document.getElementById('moveFolderPath');
        if (folderPathInput) {
            folderPathInput.value = '';
        }

        try {
            // Fetch model roots
            const modelRootSelect = document.getElementById('moveModelRoot');
            let rootsData;
            if (modelType) {
                rootsData = await apiClient.fetchModelRoots(modelType);
            } else {
                rootsData = await apiClient.fetchModelRoots();
            }
            
            if (!rootsData.roots || rootsData.roots.length === 0) {
                throw new Error(`No ${modelConfig.displayName.toLowerCase()} roots found`);
            }

            // Populate model root selector
            modelRootSelect.innerHTML = rootsData.roots.map(root => 
                `<option value="${root}">${root}</option>`
            ).join('');

            // Set default root if available
            const settingsKey = `default_${currentPageType.slice(0, -1)}_root`;
            const defaultRoot = getStorageItem('settings', {})[settingsKey];
            if (defaultRoot && rootsData.roots.includes(defaultRoot)) {
                modelRootSelect.value = defaultRoot;
            }

            // Initialize event listeners
            this.initializeEventListeners();
            
            // Setup folder tree manager
            this.folderTreeManager.init({
                onPathChange: (path) => {
                    this.updateTargetPath();
                },
                elementsPrefix: 'move'
            });
            
            // Initialize folder tree
            await this.initializeFolderTree();

            this.updateTargetPath();
            modalManager.showModal('moveModal', null, () => {
                // Cleanup on modal close
                if (this.folderTreeManager) {
                    this.folderTreeManager.destroy();
                }
            });
            
        } catch (error) {
            console.error(`Error fetching ${modelConfig.displayName.toLowerCase()} roots or folders:`, error);
            showToast(error.message, 'error');
        }
    }

    async initializeFolderTree() {
        try {
            const apiClient = getModelApiClient();
            // Fetch unified folder tree
            const treeData = await apiClient.fetchUnifiedFolderTree();
            
            if (treeData.success) {
                // Load tree data into folder tree manager
                await this.folderTreeManager.loadTree(treeData.tree);
            } else {
                console.error('Failed to fetch folder tree:', treeData.error);
                showToast('Failed to load folder tree', 'error');
            }
        } catch (error) {
            console.error('Error initializing folder tree:', error);
            showToast('Error loading folder tree', 'error');
        }
    }

    updateTargetPath() {
        const pathDisplay = document.getElementById('moveTargetPathDisplay');
        const modelRoot = document.getElementById('moveModelRoot').value;
        const apiClient = getModelApiClient();
        const config = apiClient.apiConfig.config;
        
        let fullPath = modelRoot || `Select a ${config.displayName.toLowerCase()} root directory`;
        
        if (modelRoot) {
            const selectedPath = this.folderTreeManager ? this.folderTreeManager.getSelectedPath() : '';
            if (selectedPath) {
                fullPath += '/' + selectedPath;
            }
        }

        pathDisplay.innerHTML = `<span class="path-text">${fullPath}</span>`;
    }

    async moveModel() {
        const selectedRoot = document.getElementById('moveModelRoot').value;
        const apiClient = getModelApiClient();
        const config = apiClient.apiConfig.config;
        
        if (!selectedRoot) {
            showToast(`Please select a ${config.displayName.toLowerCase()} root directory`, 'error');
            return;
        }

        // Get selected folder path from folder tree manager
        const targetFolder = this.folderTreeManager.getSelectedPath();
        
        let targetPath = selectedRoot;
        if (targetFolder) {
            targetPath = `${targetPath}/${targetFolder}`;
        }

        try {
            if (this.bulkFilePaths) {
                // Bulk move mode
                const movedFilePaths = await apiClient.moveBulkModels(this.bulkFilePaths, targetPath);

                // Update virtual scroller if in active folder view
                const pageState = getCurrentPageState();
                if (pageState.activeFolder !== null && state.virtualScroller) {
                    // Remove only successfully moved items
                    movedFilePaths.forEach(newFilePath => {
                        // Find original filePath by matching filename
                        const filename = newFilePath.substring(newFilePath.lastIndexOf('/') + 1);
                        const originalFilePath = this.bulkFilePaths.find(fp => fp.endsWith('/' + filename));
                        if (originalFilePath) {
                            state.virtualScroller.removeItemByFilePath(originalFilePath);
                        }
                    });
                } else {
                    // Update the model cards' filepath in the DOM
                    movedFilePaths.forEach(newFilePath => {
                        const filename = newFilePath.substring(newFilePath.lastIndexOf('/') + 1);
                        const originalFilePath = this.bulkFilePaths.find(fp => fp.endsWith('/' + filename));
                        if (originalFilePath) {
                            state.virtualScroller.updateSingleItem(originalFilePath, {file_path: newFilePath});
                        }
                    });
                }
            } else {
                // Single move mode
                const newFilePath = await apiClient.moveSingleModel(this.currentFilePath, targetPath);

                const pageState = getCurrentPageState();
                if (newFilePath) {
                    if (pageState.activeFolder !== null && state.virtualScroller) {
                        state.virtualScroller.removeItemByFilePath(this.currentFilePath);
                    } else {
                        state.virtualScroller.updateSingleItem(this.currentFilePath, {file_path: newFilePath});
                    }
                }
            }

            // Refresh folder tags after successful move
            try {
                const foldersData = await apiClient.fetchModelFolders();
                updateFolderTags(foldersData.folders);
            } catch (error) {
                console.error('Error refreshing folder tags:', error);
            }

            modalManager.closeModal('moveModal');
            
            // If we were in bulk mode, exit it after successful move
            if (this.bulkFilePaths && state.bulkMode) {
                bulkManager.toggleBulkMode();
            }

        } catch (error) {
            console.error('Error moving model(s):', error);
            showToast('Failed to move model(s): ' + error.message, 'error');
        }
    }
}

export const moveManager = new MoveManager();
