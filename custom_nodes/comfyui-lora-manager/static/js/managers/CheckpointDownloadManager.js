import { modalManager } from './ModalManager.js';
import { showToast } from '../utils/uiHelpers.js';
import { LoadingManager } from './LoadingManager.js';
import { state } from '../state/index.js';
import { resetAndReload } from '../api/checkpointApi.js';
import { getStorageItem, setStorageItem } from '../utils/storageHelpers.js';

export class CheckpointDownloadManager {
    constructor() {
        this.currentVersion = null;
        this.versions = [];
        this.modelInfo = null;
        this.modelVersionId = null;
        
        this.initialized = false;
        this.selectedFolder = '';

        this.loadingManager = new LoadingManager();
        this.folderClickHandler = null;
        this.updateTargetPath = this.updateTargetPath.bind(this);
    }

    showDownloadModal() {
        console.log('Showing checkpoint download modal...');
        if (!this.initialized) {
            const modal = document.getElementById('checkpointDownloadModal');
            if (!modal) {
                console.error('Checkpoint download modal element not found');
                return;
            }
            this.initialized = true;
        }
        
        modalManager.showModal('checkpointDownloadModal', null, () => {
            // Cleanup handler when modal closes
            this.cleanupFolderBrowser();
        });
        this.resetSteps();
        
        // Auto-focus on the URL input
        setTimeout(() => {
            const urlInput = document.getElementById('checkpointUrl');
            if (urlInput) {
                urlInput.focus();
            }
        }, 100); // Small delay to ensure the modal is fully displayed
    }

    resetSteps() {
        document.querySelectorAll('#checkpointDownloadModal .download-step').forEach(step => step.style.display = 'none');
        document.getElementById('cpUrlStep').style.display = 'block';
        document.getElementById('checkpointUrl').value = '';
        document.getElementById('cpUrlError').textContent = '';
        
        // Clear new folder input
        const newFolderInput = document.getElementById('cpNewFolder');
        if (newFolderInput) {
            newFolderInput.value = '';
        }
        
        this.currentVersion = null;
        this.versions = [];
        this.modelInfo = null;
        this.modelId = null;
        this.modelVersionId = null;
        
        // Clear selected folder and remove selection from UI
        this.selectedFolder = '';
        const folderBrowser = document.getElementById('cpFolderBrowser');
        if (folderBrowser) {
            folderBrowser.querySelectorAll('.folder-item').forEach(f => 
                f.classList.remove('selected'));
        }
    }

    async validateAndFetchVersions() {
        const url = document.getElementById('checkpointUrl').value.trim();
        const errorElement = document.getElementById('cpUrlError');
        
        try {
            this.loadingManager.showSimpleLoading('Fetching model versions...');
            
            this.modelId = this.extractModelId(url);
            if (!this.modelId) {
                throw new Error('Invalid Civitai URL format');
            }

            const response = await fetch(`/api/checkpoints/civitai/versions/${this.modelId}`);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                if (errorData && errorData.error && errorData.error.includes('Model type mismatch')) {
                    throw new Error('This model is not a Checkpoint. Please switch to the LoRAs page to download LoRA models.');
                }
                throw new Error('Failed to fetch model versions');
            }
            
            this.versions = await response.json();
            if (!this.versions.length) {
                throw new Error('No versions available for this model');
            }
            
            // If we have a version ID from URL, pre-select it
            if (this.modelVersionId) {
                this.currentVersion = this.versions.find(v => v.id.toString() === this.modelVersionId);
            }
            
            this.showVersionStep();
        } catch (error) {
            errorElement.textContent = error.message;
        } finally {
            this.loadingManager.hide();
        }
    }

    extractModelId(url) {
        const modelMatch = url.match(/civitai\.com\/models\/(\d+)/);
        const versionMatch = url.match(/modelVersionId=(\d+)/);
        
        if (modelMatch) {
            this.modelVersionId = versionMatch ? versionMatch[1] : null;
            return modelMatch[1];
        }
        return null;
    }

    showVersionStep() {
        document.getElementById('cpUrlStep').style.display = 'none';
        document.getElementById('cpVersionStep').style.display = 'block';
        
        const versionList = document.getElementById('cpVersionList');
        versionList.innerHTML = this.versions.map(version => {
            const firstImage = version.images?.find(img => !img.url.endsWith('.mp4'));
            const thumbnailUrl = firstImage ? firstImage.url : '/loras_static/images/no-preview.png';
            
            // Use version-level size or fallback to first file
            const fileSize = version.modelSizeKB ? 
                (version.modelSizeKB / 1024).toFixed(2) : 
                (version.files[0]?.sizeKB / 1024).toFixed(2);
            
            // Use version-level existsLocally flag
            const existsLocally = version.existsLocally;
            const localPath = version.localPath;
            
            // Check if this is an early access version
            const isEarlyAccess = version.availability === 'EarlyAccess';
            
            // Create early access badge if needed
            let earlyAccessBadge = '';
            if (isEarlyAccess) {
                earlyAccessBadge = `
                    <div class="early-access-badge" title="Early access required">
                        <i class="fas fa-clock"></i> Early Access
                    </div>
                `;
            }
            
            // Status badge for local models
            const localStatus = existsLocally ? 
                `<div class="local-badge">
                    <i class="fas fa-check"></i> In Library
                    <div class="local-path">${localPath || ''}</div>
                 </div>` : '';

            return `
                <div class="version-item ${this.currentVersion?.id === version.id ? 'selected' : ''} 
                     ${existsLocally ? 'exists-locally' : ''} 
                     ${isEarlyAccess ? 'is-early-access' : ''}"
                     onclick="checkpointDownloadManager.selectVersion('${version.id}')">
                    <div class="version-thumbnail">
                        <img src="${thumbnailUrl}" alt="Version preview">
                    </div>
                    <div class="version-content">
                        <div class="version-header">
                            <h3>${version.name}</h3>
                            ${localStatus}
                        </div>
                        <div class="version-info">
                            ${version.baseModel ? `<div class="base-model">${version.baseModel}</div>` : ''}
                            ${earlyAccessBadge}
                        </div>
                        <div class="version-meta">
                            <span><i class="fas fa-calendar"></i> ${new Date(version.createdAt).toLocaleDateString()}</span>
                            <span><i class="fas fa-file-archive"></i> ${fileSize} MB</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        // Auto-select the version if there's only one
        if (this.versions.length === 1 && !this.currentVersion) {
            this.selectVersion(this.versions[0].id.toString());
        }
        
        // Update Next button state based on initial selection
        this.updateNextButtonState();
    }

    selectVersion(versionId) {
        this.currentVersion = this.versions.find(v => v.id.toString() === versionId.toString());
        if (!this.currentVersion) return;

        document.querySelectorAll('#cpVersionList .version-item').forEach(item => {
            item.classList.toggle('selected', item.querySelector('h3').textContent === this.currentVersion.name);
        });
        
        // Update Next button state after selection
        this.updateNextButtonState();
    }
    
    updateNextButtonState() {
        const nextButton = document.querySelector('#cpVersionStep .primary-btn');
        if (!nextButton) return;
        
        const existsLocally = this.currentVersion?.existsLocally;
        
        if (existsLocally) {
            nextButton.disabled = true;
            nextButton.classList.add('disabled');
            nextButton.textContent = 'Already in Library';
        } else {
            nextButton.disabled = false;
            nextButton.classList.remove('disabled');
            nextButton.textContent = 'Next';
        }
    }

    async proceedToLocation() {
        if (!this.currentVersion) {
            showToast('Please select a version', 'error');
            return;
        }
        
        // Double-check if the version exists locally
        const existsLocally = this.currentVersion.existsLocally;
        if (existsLocally) {
            showToast('This version already exists in your library', 'info');
            return;
        }

        document.getElementById('cpVersionStep').style.display = 'none';
        document.getElementById('cpLocationStep').style.display = 'block';
        
        try {
            // Use checkpoint roots endpoint instead of lora roots
            const response = await fetch('/api/checkpoints/roots');
            if (!response.ok) {
                throw new Error('Failed to fetch checkpoint roots');
            }
            
            const data = await response.json();
            const checkpointRoot = document.getElementById('checkpointRoot');
            checkpointRoot.innerHTML = data.roots.map(root => 
                `<option value="${root}">${root}</option>`
            ).join('');

            // Set default checkpoint root if available
            const defaultRoot = getStorageItem('settings', {}).default_checkpoint_root;
            if (defaultRoot && data.roots.includes(defaultRoot)) {
                checkpointRoot.value = defaultRoot;
            }

            // Initialize folder browser after loading roots
            this.initializeFolderBrowser();
        } catch (error) {
            showToast(error.message, 'error');
        }
    }

    backToUrl() {
        document.getElementById('cpVersionStep').style.display = 'none';
        document.getElementById('cpUrlStep').style.display = 'block';
    }

    backToVersions() {
        document.getElementById('cpLocationStep').style.display = 'none';
        document.getElementById('cpVersionStep').style.display = 'block';
    }

    async startDownload() {
        const checkpointRoot = document.getElementById('checkpointRoot').value;
        const newFolder = document.getElementById('cpNewFolder').value.trim();
        
        if (!checkpointRoot) {
            showToast('Please select a checkpoint root directory', 'error');
            return;
        }

        // Construct relative path
        let targetFolder = '';
        if (this.selectedFolder) {
            targetFolder = this.selectedFolder;
        }
        if (newFolder) {
            targetFolder = targetFolder ? 
                `${targetFolder}/${newFolder}` : newFolder;
        }

        try {
            // Show enhanced loading with progress details
            const updateProgress = this.loadingManager.showDownloadProgress(1);
            updateProgress(0, 0, this.currentVersion.name);

            // Generate a unique ID for this download
            const downloadId = Date.now().toString();
            
            // Setup WebSocket for progress updates using download-specific endpoint
            const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
            const ws = new WebSocket(`${wsProtocol}${window.location.host}/ws/download-progress?id=${downloadId}`);
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                // Handle download ID confirmation
                if (data.type === 'download_id') {
                    console.log(`Connected to checkpoint download progress with ID: ${data.download_id}`);
                    return;
                }
                
                // Only process progress updates for our download
                if (data.status === 'progress' && data.download_id === downloadId) {
                    // Update progress display with current progress
                    updateProgress(data.progress, 0, this.currentVersion.name);
                    
                    // Add more detailed status messages based on progress
                    if (data.progress < 3) {
                        this.loadingManager.setStatus(`Preparing download...`);
                    } else if (data.progress === 3) {
                        this.loadingManager.setStatus(`Downloaded preview image`);
                    } else if (data.progress > 3 && data.progress < 100) {
                        this.loadingManager.setStatus(`Downloading checkpoint file`);
                    } else {
                        this.loadingManager.setStatus(`Finalizing download...`);
                    }
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                // Continue with download even if WebSocket fails
            };

            // Start download using checkpoint download endpoint with download ID
            const response = await fetch('/api/download-model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_id: this.modelId,
                    model_version_id: this.currentVersion.id,
                    model_root: checkpointRoot,
                    relative_path: targetFolder,
                    download_id: downloadId
                })
            });

            if (!response.ok) {
                throw new Error(await response.text());
            }

            showToast('Download completed successfully', 'success');
            modalManager.closeModal('checkpointDownloadModal');
            
            // Update state specifically for the checkpoints page
            state.pages.checkpoints.activeFolder = targetFolder;
            
            // Save the active folder preference to storage
            setStorageItem('checkpoints_activeFolder', targetFolder);
            
            // Update UI to show the folder as selected
            document.querySelectorAll('.folder-tags .tag').forEach(tag => {
                const isActive = tag.dataset.folder === targetFolder;
                tag.classList.toggle('active', isActive);
                if (isActive && !tag.parentNode.classList.contains('collapsed')) {
                    // Scroll the tag into view if folder tags are not collapsed
                    tag.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            });

            await resetAndReload(true); // Pass true to update folders

        } catch (error) {
            showToast(error.message, 'error');
        } finally {
            this.loadingManager.hide();
        }
    }

    initializeFolderBrowser() {
        const folderBrowser = document.getElementById('cpFolderBrowser');
        if (!folderBrowser) return;

        // Cleanup existing handler if any
        this.cleanupFolderBrowser();

        // Create new handler
        this.folderClickHandler = (event) => {
            const folderItem = event.target.closest('.folder-item');
            if (!folderItem) return;

            if (folderItem.classList.contains('selected')) {
                folderItem.classList.remove('selected');
                this.selectedFolder = '';
            } else {
                folderBrowser.querySelectorAll('.folder-item').forEach(f => 
                    f.classList.remove('selected'));
                folderItem.classList.add('selected');
                this.selectedFolder = folderItem.dataset.folder;
            }
            
            // Update path display after folder selection
            this.updateTargetPath();
        };

        // Add the new handler
        folderBrowser.addEventListener('click', this.folderClickHandler);
        
        // Add event listeners for path updates
        const checkpointRoot = document.getElementById('checkpointRoot');
        const newFolder = document.getElementById('cpNewFolder');
        
        checkpointRoot.addEventListener('change', this.updateTargetPath);
        newFolder.addEventListener('input', this.updateTargetPath);
        
        // Update initial path
        this.updateTargetPath();
    }

    cleanupFolderBrowser() {
        if (this.folderClickHandler) {
            const folderBrowser = document.getElementById('cpFolderBrowser');
            if (folderBrowser) {
                folderBrowser.removeEventListener('click', this.folderClickHandler);
                this.folderClickHandler = null;
            }
        }
        
        // Remove path update listeners
        const checkpointRoot = document.getElementById('checkpointRoot');
        const newFolder = document.getElementById('cpNewFolder');
        
        if (checkpointRoot) checkpointRoot.removeEventListener('change', this.updateTargetPath);
        if (newFolder) newFolder.removeEventListener('input', this.updateTargetPath);
    }
    
    updateTargetPath() {
        const pathDisplay = document.getElementById('cpTargetPathDisplay');
        const checkpointRoot = document.getElementById('checkpointRoot').value;
        const newFolder = document.getElementById('cpNewFolder').value.trim();
        
        let fullPath = checkpointRoot || 'Select a checkpoint root directory';
        
        if (checkpointRoot) {
            if (this.selectedFolder) {
                fullPath += '/' + this.selectedFolder;
            }
            if (newFolder) {
                fullPath += '/' + newFolder;
            }
        }

        pathDisplay.innerHTML = `<span class="path-text">${fullPath}</span>`;
    }
}