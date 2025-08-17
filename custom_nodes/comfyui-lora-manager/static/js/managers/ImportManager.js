import { modalManager } from './ModalManager.js';
import { LoadingManager } from './LoadingManager.js';
import { ImportStepManager } from './import/ImportStepManager.js';
import { ImageProcessor } from './import/ImageProcessor.js';
import { RecipeDataManager } from './import/RecipeDataManager.js';
import { DownloadManager } from './import/DownloadManager.js';
import { FolderBrowser } from './import/FolderBrowser.js';
import { formatFileSize } from '../utils/formatters.js';

export class ImportManager {
    constructor() {
        // Core state properties
        this.recipeImage = null;
        this.recipeData = null;
        this.recipeName = '';
        this.recipeTags = [];
        this.missingLoras = [];
        this.initialized = false;
        this.selectedFolder = '';
        this.downloadableLoRAs = [];
        this.recipeId = null;
        this.importMode = 'url'; // Default mode: 'url' or 'upload'
        
        // Initialize sub-managers
        this.loadingManager = new LoadingManager();
        this.stepManager = new ImportStepManager();
        this.imageProcessor = new ImageProcessor(this);
        this.recipeDataManager = new RecipeDataManager(this);
        this.downloadManager = new DownloadManager(this);
        this.folderBrowser = new FolderBrowser(this);
        
        // Bind methods
        this.formatFileSize = formatFileSize;
    }

    showImportModal(recipeData = null, recipeId = null) {
        if (!this.initialized) {
            const modal = document.getElementById('importModal');
            if (!modal) {
                console.error('Import modal element not found');
                return;
            }
            this.initialized = true;
        }
        
        // Reset state
        this.resetSteps();
        if (recipeData) {
            this.downloadableLoRAs = recipeData.loras;
            this.recipeId = recipeId;
        }
        
        // Show modal
        modalManager.showModal('importModal', null, () => {
            this.folderBrowser.cleanup();
            this.stepManager.removeInjectedStyles();
        });
        
        // Verify visibility and focus on URL input
        setTimeout(() => {
            this.ensureModalVisible();
            
            // Ensure URL option is selected and focus on the input
            this.toggleImportMode('url');
            const urlInput = document.getElementById('imageUrlInput');
            if (urlInput) {
                urlInput.focus();
            }
        }, 50);
    }

    resetSteps() {
        // Clear UI state
        this.stepManager.removeInjectedStyles();
        this.stepManager.showStep('uploadStep');
        
        // Reset form inputs
        const fileInput = document.getElementById('recipeImageUpload');
        if (fileInput) fileInput.value = '';
        
        const urlInput = document.getElementById('imageUrlInput');
        if (urlInput) urlInput.value = '';
        
        const uploadError = document.getElementById('uploadError');
        if (uploadError) uploadError.textContent = '';
        
        const importUrlError = document.getElementById('importUrlError');
        if (importUrlError) importUrlError.textContent = '';
        
        const recipeName = document.getElementById('recipeName');
        if (recipeName) recipeName.value = '';
        
        const tagsContainer = document.getElementById('tagsContainer');
        if (tagsContainer) tagsContainer.innerHTML = '<div class="empty-tags">No tags added</div>';
        
        // Reset state variables
        this.recipeImage = null;
        this.recipeData = null;
        this.recipeName = '';
        this.recipeTags = [];
        this.missingLoras = [];
        this.downloadableLoRAs = [];
        
        // Reset import mode
        this.importMode = 'url';
        this.toggleImportMode('url');
        
        // Reset folder browser
        this.selectedFolder = '';
        const folderBrowser = document.getElementById('importFolderBrowser');
        if (folderBrowser) {
            folderBrowser.querySelectorAll('.folder-item').forEach(f => 
                f.classList.remove('selected'));
        }
        
        // Clear missing LoRAs list
        const missingLorasList = document.getElementById('missingLorasList');
        if (missingLorasList) missingLorasList.innerHTML = '';
        
        // Reset total download size
        const totalSizeDisplay = document.getElementById('totalDownloadSize');
        if (totalSizeDisplay) totalSizeDisplay.textContent = 'Calculating...';
        
        // Remove warnings
        const deletedLorasWarning = document.getElementById('deletedLorasWarning');
        if (deletedLorasWarning) deletedLorasWarning.remove();
        
        const earlyAccessWarning = document.getElementById('earlyAccessWarning');
        if (earlyAccessWarning) earlyAccessWarning.remove();
        
        // Reset duplicate related properties
        this.duplicateRecipes = [];
    }

    toggleImportMode(mode) {
        this.importMode = mode;
        
        // Update toggle buttons
        const uploadBtn = document.querySelector('.toggle-btn[data-mode="upload"]');
        const urlBtn = document.querySelector('.toggle-btn[data-mode="url"]');
        
        if (uploadBtn && urlBtn) {
            if (mode === 'upload') {
                uploadBtn.classList.add('active');
                urlBtn.classList.remove('active');
            } else {
                uploadBtn.classList.remove('active');
                urlBtn.classList.add('active');
            }
        }
        
        // Show/hide appropriate sections
        const uploadSection = document.getElementById('uploadSection');
        const urlSection = document.getElementById('urlSection');
        
        if (uploadSection && urlSection) {
            if (mode === 'upload') {
                uploadSection.style.display = 'block';
                urlSection.style.display = 'none';
            } else {
                uploadSection.style.display = 'none';
                urlSection.style.display = 'block';
            }
        }
        
        // Clear error messages
        const uploadError = document.getElementById('uploadError');
        const importUrlError = document.getElementById('importUrlError');
        
        if (uploadError) uploadError.textContent = '';
        if (importUrlError) importUrlError.textContent = '';
    }

    handleImageUpload(event) {
        this.imageProcessor.handleFileUpload(event);
    }

    async handleUrlInput() {
        await this.imageProcessor.handleUrlInput();
    }

    async uploadAndAnalyzeImage() {
        await this.imageProcessor.uploadAndAnalyzeImage();
    }

    showRecipeDetailsStep() {
        this.recipeDataManager.showRecipeDetailsStep();
    }

    handleRecipeNameChange(event) {
        this.recipeName = event.target.value.trim();
    }

    addTag() {
        this.recipeDataManager.addTag();
    }
    
    removeTag(tag) {
        this.recipeDataManager.removeTag(tag);
    }

    proceedFromDetails() {
        this.recipeDataManager.proceedFromDetails();
    }

    async proceedToLocation() {
        await this.folderBrowser.proceedToLocation();
    }

    backToUpload() {
        this.stepManager.showStep('uploadStep');
        
        // Reset file input
        const fileInput = document.getElementById('recipeImageUpload');
        if (fileInput) fileInput.value = '';
        
        // Reset URL input
        const urlInput = document.getElementById('imageUrlInput');
        if (urlInput) urlInput.value = '';
        
        // Clear error messages
        const uploadError = document.getElementById('uploadError');
        if (uploadError) uploadError.textContent = '';
        
        const importUrlError = document.getElementById('importUrlError');
        if (importUrlError) importUrlError.textContent = '';
    }

    backToDetails() {
        this.stepManager.showStep('detailsStep');
    }

    async saveRecipe() {
        await this.downloadManager.saveRecipe();
    }

    updateTargetPath() {
        this.folderBrowser.updateTargetPath();
    }

    ensureModalVisible() {
        const importModal = document.getElementById('importModal');
        if (!importModal) {
            console.error('Import modal element not found');
            return false;
        }
        
        // Check if modal is actually visible
        const modalDisplay = window.getComputedStyle(importModal).display;
        if (modalDisplay !== 'block') {
            console.error('Import modal is not visible, display: ' + modalDisplay);
            return false;
        }
        
        return true;
    }

    /**
     * NOTE: This function is no longer needed with the simplified duplicates flow.
     * We're keeping it as a no-op stub to avoid breaking existing code that might call it.
     */
    markDuplicateForDeletion(recipeId, buttonElement) {
        // This functionality has been removed
        console.log('markDuplicateForDeletion is deprecated');
    }

    /**
     * NOTE: This function is no longer needed with the simplified duplicates flow.
     * We're keeping it as a no-op stub to avoid breaking existing code that might call it.
     */
    importRecipeAnyway() {
        // This functionality has been simplified
        // Just proceed with normal flow
        this.proceedFromDetails();
    }

    downloadMissingLoras(recipeData, recipeId) {
        // Store the recipe data and ID
        this.recipeData = recipeData;
        this.recipeId = recipeId;
        
        // Show the modal and go to location step
        this.showImportModal(recipeData, recipeId);
        this.proceedToLocation();
        
        // Update the modal title
        const modalTitle = document.querySelector('#importModal h2');
        if (modalTitle) modalTitle.textContent = 'Download Missing LoRAs';
        
        // Update the save button text
        const saveButton = document.querySelector('#locationStep .primary-btn');
        if (saveButton) saveButton.textContent = 'Download Missing LoRAs';
        
        // Hide the back button
        const backButton = document.querySelector('#locationStep .secondary-btn');
        if (backButton) backButton.style.display = 'none';
    }
}
