import { BaseModelApiClient } from './baseModelApi.js';
import { showToast } from '../utils/uiHelpers.js';

/**
 * Checkpoint-specific API client
 */
export class CheckpointApiClient extends BaseModelApiClient {
    /**
     * Get checkpoint information
     */
    async getCheckpointInfo(filePath) {
        try {
            const response = await fetch(this.apiConfig.endpoints.specific.info, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ file_path: filePath })
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch checkpoint info');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error fetching checkpoint info:', error);
            throw error;
        }
    }

    /**
     * Get checkpoint roots
     */
    async getCheckpointsRoots() {
        try {
            const response = await fetch(this.apiConfig.endpoints.specific.checkpoints_roots, {
                method: 'GET'
            });
            if (!response.ok) {
                throw new Error('Failed to fetch checkpoints roots');
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching checkpoints roots:', error);
            throw error;
        }
    }

    /**
     * Get unet roots
     */
    async getUnetRoots() {
        try {
            const response = await fetch(this.apiConfig.endpoints.specific.unet_roots, {
                method: 'GET'
            });
            if (!response.ok) {
                throw new Error('Failed to fetch unet roots');
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching unet roots:', error);
            throw error;
        }
    }

    /**
     * Get appropriate roots based on model type
     */
    async fetchModelRoots(modelType = 'checkpoint') {
        try {
            let response;
            if (modelType === 'diffusion_model') {
                response = await fetch(this.apiConfig.endpoints.specific.unet_roots, {
                    method: 'GET'
                });
            } else {
                response = await fetch(this.apiConfig.endpoints.specific.checkpoints_roots, {
                    method: 'GET'
                });
            }
            
            if (!response.ok) {
                throw new Error(`Failed to fetch ${modelType} roots`);
            }
            return await response.json();
        } catch (error) {
            console.error(`Error fetching ${modelType} roots:`, error);
            throw error;
        }
    }
}
