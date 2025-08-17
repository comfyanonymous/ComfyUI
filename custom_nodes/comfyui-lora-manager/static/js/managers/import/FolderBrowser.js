import { showToast } from '../../utils/uiHelpers.js';
import { getStorageItem } from '../../utils/storageHelpers.js';

export class FolderBrowser {
    constructor(importManager) {
        this.importManager = importManager;
        this.folderClickHandler = null;
        this.updateTargetPath = this.updateTargetPath.bind(this);
    }

    async proceedToLocation() {
        // Show the location step with special handling
        this.importManager.stepManager.showStep('locationStep');
        
        // Double-check after a short delay to ensure the step is visible
        setTimeout(() => {
            const locationStep = document.getElementById('locationStep');
            if (locationStep.style.display !== 'block' || 
                window.getComputedStyle(locationStep).display !== 'block') {
                // Force display again
                locationStep.style.display = 'block';
                
                // If still not visible, try with injected style
                if (window.getComputedStyle(locationStep).display !== 'block') {
                    this.importManager.stepManager.injectedStyles = document.createElement('style');
                    this.importManager.stepManager.injectedStyles.innerHTML = `
                        #locationStep {
                            display: block !important;
                            opacity: 1 !important;
                            visibility: visible !important;
                        }
                    `;
                    document.head.appendChild(this.importManager.stepManager.injectedStyles);
                }
            }
        }, 100);
        
        try {
            // Display missing LoRAs that will be downloaded
            const missingLorasList = document.getElementById('missingLorasList');
            if (missingLorasList && this.importManager.downloadableLoRAs.length > 0) {
                // Calculate total size
                const totalSize = this.importManager.downloadableLoRAs.reduce((sum, lora) => {
                    return sum + (lora.size ? parseInt(lora.size) : 0);
                }, 0);
                
                // Update total size display
                const totalSizeDisplay = document.getElementById('totalDownloadSize');
                if (totalSizeDisplay) {
                    totalSizeDisplay.textContent = this.importManager.formatFileSize(totalSize);
                }
                
                // Update header to include count of missing LoRAs
                const missingLorasHeader = document.querySelector('.summary-header h3');
                if (missingLorasHeader) {
                    missingLorasHeader.innerHTML = `Missing LoRAs <span class="lora-count-badge">(${this.importManager.downloadableLoRAs.length})</span> <span id="totalDownloadSize" class="total-size-badge">${this.importManager.formatFileSize(totalSize)}</span>`;
                }
                
                // Generate missing LoRAs list
                missingLorasList.innerHTML = this.importManager.downloadableLoRAs.map(lora => {
                    const sizeDisplay = lora.size ? 
                        this.importManager.formatFileSize(lora.size) : 'Unknown size';
                    const baseModel = lora.baseModel ? 
                        `<span class="lora-base-model">${lora.baseModel}</span>` : '';
                    const isEarlyAccess = lora.isEarlyAccess;
                    
                    // Early access badge
                    let earlyAccessBadge = '';
                    if (isEarlyAccess) {
                        earlyAccessBadge = `<span class="early-access-badge">
                            <i class="fas fa-clock"></i> Early Access
                        </span>`;
                    }
                    
                    return `
                        <div class="missing-lora-item ${isEarlyAccess ? 'is-early-access' : ''}">
                            <div class="missing-lora-info">
                                <div class="missing-lora-name">${lora.name}</div>
                                ${baseModel}
                                ${earlyAccessBadge}
                            </div>
                            <div class="missing-lora-size">${sizeDisplay}</div>
                        </div>
                    `;
                }).join('');
                
                // Set up toggle for missing LoRAs list
                const toggleBtn = document.getElementById('toggleMissingLorasList');
                if (toggleBtn) {
                    toggleBtn.addEventListener('click', () => {
                        missingLorasList.classList.toggle('collapsed');
                        const icon = toggleBtn.querySelector('i');
                        if (icon) {
                            icon.classList.toggle('fa-chevron-down');
                            icon.classList.toggle('fa-chevron-up');
                        }
                    });
                }
            }
            
            // Fetch LoRA roots
            const rootsResponse = await fetch('/api/loras/roots');
            if (!rootsResponse.ok) {
                throw new Error(`Failed to fetch LoRA roots: ${rootsResponse.status}`);
            }
            
            const rootsData = await rootsResponse.json();
            const loraRoot = document.getElementById('importLoraRoot');
            if (loraRoot) {
                loraRoot.innerHTML = rootsData.roots.map(root => 
                    `<option value="${root}">${root}</option>`
                ).join('');
                
                // Set default lora root if available
                const defaultRoot = getStorageItem('settings', {}).default_lora_root;
                if (defaultRoot && rootsData.roots.includes(defaultRoot)) {
                    loraRoot.value = defaultRoot;
                }
            }
            
            // Fetch folders
            const foldersResponse = await fetch('/api/loras/folders');
            if (!foldersResponse.ok) {
                throw new Error(`Failed to fetch folders: ${foldersResponse.status}`);
            }
            
            const foldersData = await foldersResponse.json();
            const folderBrowser = document.getElementById('importFolderBrowser');
            if (folderBrowser) {
                folderBrowser.innerHTML = foldersData.folders.map(folder => 
                    folder ? `<div class="folder-item" data-folder="${folder}">${folder}</div>` : ''
                ).join('');
            }

            // Initialize folder browser after loading data
            this.initializeFolderBrowser();
        } catch (error) {
            console.error('Error in API calls:', error);
            showToast(error.message, 'error');
        }
    }

    initializeFolderBrowser() {
        const folderBrowser = document.getElementById('importFolderBrowser');
        if (!folderBrowser) return;

        // Cleanup existing handler if any
        this.cleanup();

        // Create new handler
        this.folderClickHandler = (event) => {
            const folderItem = event.target.closest('.folder-item');
            if (!folderItem) return;

            if (folderItem.classList.contains('selected')) {
                folderItem.classList.remove('selected');
                this.importManager.selectedFolder = '';
            } else {
                folderBrowser.querySelectorAll('.folder-item').forEach(f => 
                    f.classList.remove('selected'));
                folderItem.classList.add('selected');
                this.importManager.selectedFolder = folderItem.dataset.folder;
            }
            
            // Update path display after folder selection
            this.updateTargetPath();
        };

        // Add the new handler
        folderBrowser.addEventListener('click', this.folderClickHandler);
        
        // Add event listeners for path updates
        const loraRoot = document.getElementById('importLoraRoot');
        const newFolder = document.getElementById('importNewFolder');
        
        if (loraRoot) loraRoot.addEventListener('change', this.updateTargetPath);
        if (newFolder) newFolder.addEventListener('input', this.updateTargetPath);
        
        // Update initial path
        this.updateTargetPath();
    }

    cleanup() {
        if (this.folderClickHandler) {
            const folderBrowser = document.getElementById('importFolderBrowser');
            if (folderBrowser) {
                folderBrowser.removeEventListener('click', this.folderClickHandler);
                this.folderClickHandler = null;
            }
        }
        
        // Remove path update listeners
        const loraRoot = document.getElementById('importLoraRoot');
        const newFolder = document.getElementById('importNewFolder');
        
        if (loraRoot) loraRoot.removeEventListener('change', this.updateTargetPath);
        if (newFolder) newFolder.removeEventListener('input', this.updateTargetPath);
    }
    
    updateTargetPath() {
        const pathDisplay = document.getElementById('importTargetPathDisplay');
        if (!pathDisplay) return;
        
        const loraRoot = document.getElementById('importLoraRoot')?.value || '';
        const newFolder = document.getElementById('importNewFolder')?.value?.trim() || '';
        
        let fullPath = loraRoot || 'Select a LoRA root directory'; 
        
        if (loraRoot) {
            if (this.importManager.selectedFolder) {
                fullPath += '/' + this.importManager.selectedFolder;
            }
            if (newFolder) {
                fullPath += '/' + newFolder;
            }
        }
    
        pathDisplay.innerHTML = `<span class="path-text">${fullPath}</span>`;
    }
}
