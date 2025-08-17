import { modalManager } from '../managers/ModalManager.js';
import { getModelApiClient } from '../api/modelApiFactory.js';

const apiClient = getModelApiClient();

let pendingDeletePath = null;
let pendingExcludePath = null;

export function showDeleteModal(filePath) {
    pendingDeletePath = filePath;
    
    const card = document.querySelector(`.model-card[data-filepath="${filePath}"]`);
    const modelName = card ? card.dataset.name : filePath.split('/').pop();
    const modal = modalManager.getModal('deleteModal').element;
    const modelInfo = modal.querySelector('.delete-model-info');
    
    modelInfo.innerHTML = `
        <strong>Model:</strong> ${modelName}
        <br>
        <strong>File:</strong> ${filePath}
    `;
    
    modalManager.showModal('deleteModal');
}

export async function confirmDelete() {
    if (!pendingDeletePath) return;
    
    try {
        await apiClient.deleteModel(pendingDeletePath);
        
        closeDeleteModal();

        if (window.modelDuplicatesManager) {
            window.modelDuplicatesManager.updateDuplicatesBadgeAfterRefresh();
        }
    } catch (error) {
        console.error('Error deleting model:', error);
        alert(`Error deleting model: ${error}`);
    }
}

export function closeDeleteModal() {
    modalManager.closeModal('deleteModal');
    pendingDeletePath = null;
}

// Functions for the exclude modal
export function showExcludeModal(filePath) {
    pendingExcludePath = filePath;
    
    const card = document.querySelector(`.model-card[data-filepath="${filePath}"]`);
    const modelName = card ? card.dataset.name : filePath.split('/').pop();
    const modal = modalManager.getModal('excludeModal').element;
    const modelInfo = modal.querySelector('.exclude-model-info');
    
    modelInfo.innerHTML = `
        <strong>Model:</strong> ${modelName}
        <br>
        <strong>File:</strong> ${filePath}
    `;
    
    modalManager.showModal('excludeModal');
}

export function closeExcludeModal() {
    modalManager.closeModal('excludeModal');
    pendingExcludePath = null;
}

export async function confirmExclude() {
    if (!pendingExcludePath) return;
    
    try {
        await apiClient.excludeModel(pendingExcludePath);
        
        closeExcludeModal();

        if (window.modelDuplicatesManager) {
            window.modelDuplicatesManager.updateDuplicatesBadgeAfterRefresh();
        }
    } catch (error) {
        console.error('Error excluding model:', error);
    }
}