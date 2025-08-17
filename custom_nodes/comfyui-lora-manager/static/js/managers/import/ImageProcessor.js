import { showToast } from '../../utils/uiHelpers.js';

export class ImageProcessor {
    constructor(importManager) {
        this.importManager = importManager;
    }

    handleFileUpload(event) {
        const file = event.target.files[0];
        const errorElement = document.getElementById('uploadError');
        
        if (!file) return;
        
        // Validate file type
        if (!file.type.match('image.*')) {
            errorElement.textContent = 'Please select an image file';
            return;
        }
        
        // Reset error
        errorElement.textContent = '';
        this.importManager.recipeImage = file;
        
        // Auto-proceed to next step if file is selected
        this.importManager.uploadAndAnalyzeImage();
    }

    async handleUrlInput() {
        const urlInput = document.getElementById('imageUrlInput');
        const errorElement = document.getElementById('importUrlError');
        const input = urlInput.value.trim();
        
        // Validate input
        if (!input) {
            errorElement.textContent = 'Please enter a URL or file path';
            return;
        }
        
        // Reset error
        errorElement.textContent = '';
        
        // Show loading indicator
        this.importManager.loadingManager.showSimpleLoading('Processing input...');
        
        try {
            // Check if it's a URL or a local file path
            if (input.startsWith('http://') || input.startsWith('https://')) {
                // Handle as URL
                await this.analyzeImageFromUrl(input);
            } else {
                // Handle as local file path
                await this.analyzeImageFromLocalPath(input);
            }
        } catch (error) {
            errorElement.textContent = error.message || 'Failed to process input';
        } finally {
            this.importManager.loadingManager.hide();
        }
    }

    async analyzeImageFromUrl(url) {
        try {
            // Call the API with URL data
            const response = await fetch('/api/recipes/analyze-image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url: url })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to analyze image from URL');
            }
            
            // Get recipe data from response
            this.importManager.recipeData = await response.json();
            
            // Check if we have an error message
            if (this.importManager.recipeData.error) {
                throw new Error(this.importManager.recipeData.error);
            }
            
            // Check if we have valid recipe data
            if (!this.importManager.recipeData || 
                !this.importManager.recipeData.loras || 
                this.importManager.recipeData.loras.length === 0) {
                throw new Error('No LoRA information found in this image');
            }
            
            // Find missing LoRAs
            this.importManager.missingLoras = this.importManager.recipeData.loras.filter(
                lora => !lora.existsLocally
            );
            
            // Reset import as new flag
            this.importManager.importAsNew = false;
            
            // Proceed to recipe details step
            this.importManager.showRecipeDetailsStep();
            
        } catch (error) {
            console.error('Error analyzing URL:', error);
            throw error;
        }
    }

    async analyzeImageFromLocalPath(path) {
        try {
            // Call the API with local path data
            const response = await fetch('/api/recipes/analyze-local-image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ path: path })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to load image from local path');
            }
            
            // Get recipe data from response
            this.importManager.recipeData = await response.json();
            
            // Check if we have an error message
            if (this.importManager.recipeData.error) {
                throw new Error(this.importManager.recipeData.error);
            }
            
            // Check if we have valid recipe data
            if (!this.importManager.recipeData || 
                !this.importManager.recipeData.loras || 
                this.importManager.recipeData.loras.length === 0) {
                throw new Error('No LoRA information found in this image');
            }
            
            // Find missing LoRAs
            this.importManager.missingLoras = this.importManager.recipeData.loras.filter(
                lora => !lora.existsLocally
            );
            
            // Reset import as new flag
            this.importManager.importAsNew = false;
            
            // Proceed to recipe details step
            this.importManager.showRecipeDetailsStep();
            
        } catch (error) {
            console.error('Error analyzing local path:', error);
            throw error;
        }
    }

    async uploadAndAnalyzeImage() {
        if (!this.importManager.recipeImage) {
            showToast('Please select an image first', 'error');
            return;
        }
        
        try {
            this.importManager.loadingManager.showSimpleLoading('Analyzing image metadata...');
            
            // Create form data for upload
            const formData = new FormData();
            formData.append('image', this.importManager.recipeImage);
            
            // Upload image for analysis
            const response = await fetch('/api/recipes/analyze-image', {
                method: 'POST',
                body: formData
            });
             
            // Get recipe data from response
            this.importManager.recipeData = await response.json();

            // Check if we have an error message
            if (this.importManager.recipeData.error) {
                throw new Error(this.importManager.recipeData.error);
            }
            
            // Check if we have valid recipe data
            if (!this.importManager.recipeData || 
                !this.importManager.recipeData.loras || 
                this.importManager.recipeData.loras.length === 0) {
                throw new Error('No LoRA information found in this image');
            }
            
            // Find missing LoRAs
            this.importManager.missingLoras = this.importManager.recipeData.loras.filter(
                lora => !lora.existsLocally
            );
            
            // Reset import as new flag
            this.importManager.importAsNew = false;
            
            // Proceed to recipe details step
            this.importManager.showRecipeDetailsStep();
            
        } catch (error) {
            document.getElementById('uploadError').textContent = error.message;
        } finally {
            this.importManager.loadingManager.hide();
        }
    }
}
