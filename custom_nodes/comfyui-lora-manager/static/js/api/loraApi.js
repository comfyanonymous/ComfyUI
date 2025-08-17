import { BaseModelApiClient } from './baseModelApi.js';
import { showToast } from '../utils/uiHelpers.js';
import { getSessionItem } from '../utils/storageHelpers.js';

/**
 * LoRA-specific API client
 */
export class LoraApiClient extends BaseModelApiClient {
    /**
     * Add LoRA-specific parameters to query
     */
    _addModelSpecificParams(params, pageState) {
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

    /**
     * Get LoRA notes
     */
    async getLoraNote(filePath) {
        try {
            const response = await fetch(this.apiConfig.endpoints.specific.notes,
                {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ file_path: filePath })
                }
            );
            
            if (!response.ok) {
                throw new Error('Failed to fetch LoRA notes');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error fetching LoRA notes:', error);
            throw error;
        }
    }

    /**
     * Get LoRA trigger words
     */
    async getLoraTriggerWords(filePath) {
        try {
            const response = await fetch(this.apiConfig.endpoints.specific.triggerWords, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ file_path: filePath })
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch trigger words');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error fetching trigger words:', error);
            throw error;
        }
    }

    /**
     * Get letter counts for LoRAs
     */
    async getLetterCounts() {
        try {
            const response = await fetch(this.apiConfig.endpoints.specific.letterCounts);
            if (!response.ok) {
                throw new Error('Failed to fetch letter counts');
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching letter counts:', error);
            throw error;
        }
    }
}
