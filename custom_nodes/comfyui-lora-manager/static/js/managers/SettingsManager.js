import { modalManager } from './ModalManager.js';
import { showToast } from '../utils/uiHelpers.js';
import { state } from '../state/index.js';
import { resetAndReload } from '../api/modelApiFactory.js';
import { setStorageItem, getStorageItem } from '../utils/storageHelpers.js';
import { DOWNLOAD_PATH_TEMPLATES, MAPPABLE_BASE_MODELS, PATH_TEMPLATE_PLACEHOLDERS, DEFAULT_PATH_TEMPLATES } from '../utils/constants.js';

export class SettingsManager {
    constructor() {
        this.initialized = false;
        this.isOpen = false;
        
        // Add initialization to sync with modal state
        this.currentPage = document.body.dataset.page || 'loras';
        
        // Ensure settings are loaded from localStorage
        this.loadSettingsFromStorage();

        // Sync settings to backend if needed
        this.syncSettingsToBackendIfNeeded();

        this.initialize();
    }

    loadSettingsFromStorage() {
        // Get saved settings from localStorage
        const savedSettings = getStorageItem('settings');

        // Migrate legacy default_loras_root to default_lora_root if present
        if (savedSettings && savedSettings.default_loras_root && !savedSettings.default_lora_root) {
            savedSettings.default_lora_root = savedSettings.default_loras_root;
            delete savedSettings.default_loras_root;
            setStorageItem('settings', savedSettings);
        }

        // Apply saved settings to state if available
        if (savedSettings) {
            state.global.settings = { ...state.global.settings, ...savedSettings };
        }

        // Initialize default values for new settings if they don't exist
        if (state.global.settings.compactMode === undefined) {
            state.global.settings.compactMode = false;
        }

        // Set default for optimizeExampleImages if undefined
        if (state.global.settings.optimizeExampleImages === undefined) {
            state.global.settings.optimizeExampleImages = true;
        }

        // Set default for autoDownloadExampleImages if undefined
        if (state.global.settings.autoDownloadExampleImages === undefined) {
            state.global.settings.autoDownloadExampleImages = true;
        }

        // Set default for cardInfoDisplay if undefined
        if (state.global.settings.cardInfoDisplay === undefined) {
            state.global.settings.cardInfoDisplay = 'always';
        }

        // Set default for defaultCheckpointRoot if undefined
        if (state.global.settings.default_checkpoint_root === undefined) {
            state.global.settings.default_checkpoint_root = '';
        }

        // Convert old boolean compactMode to new displayDensity string
        if (typeof state.global.settings.displayDensity === 'undefined') {
            if (state.global.settings.compactMode === true) {
                state.global.settings.displayDensity = 'compact';
            } else {
                state.global.settings.displayDensity = 'default';
            }
            // We can delete the old setting, but keeping it for backwards compatibility
        }

        // Migrate legacy download_path_template to new structure
        if (state.global.settings.download_path_template && !state.global.settings.download_path_templates) {
            const legacyTemplate = state.global.settings.download_path_template;
            state.global.settings.download_path_templates = {
                lora: legacyTemplate,
                checkpoint: legacyTemplate,
                embedding: legacyTemplate
            };
            delete state.global.settings.download_path_template;
            setStorageItem('settings', state.global.settings);
        }

        // Set default for download path templates if undefined
        if (state.global.settings.download_path_templates === undefined) {
            state.global.settings.download_path_templates = { ...DEFAULT_PATH_TEMPLATES };
        }

        // Ensure all model types have templates
        Object.keys(DEFAULT_PATH_TEMPLATES).forEach(modelType => {
            if (typeof state.global.settings.download_path_templates[modelType] === 'undefined') {
                state.global.settings.download_path_templates[modelType] = DEFAULT_PATH_TEMPLATES[modelType];
            }
        });

        // Set default for base model path mappings if undefined
        if (state.global.settings.base_model_path_mappings === undefined) {
            state.global.settings.base_model_path_mappings = {};
        }

        // Set default for defaultEmbeddingRoot if undefined
        if (state.global.settings.default_embedding_root === undefined) {
            state.global.settings.default_embedding_root = '';
        }

        // Set default for includeTriggerWords if undefined
        if (state.global.settings.includeTriggerWords === undefined) {
            state.global.settings.includeTriggerWords = false;
        }
    }

    async syncSettingsToBackendIfNeeded() {
        // Get local settings from storage
        const localSettings = getStorageItem('settings') || {};

        // Fields that need to be synced to backend
        const fieldsToSync = [
            'civitai_api_key',
            'default_lora_root',
            'default_checkpoint_root',
            'default_embedding_root',
            'base_model_path_mappings',
            'download_path_templates'
        ];

        // Build payload for syncing
        const payload = {};

        fieldsToSync.forEach(key => {
            if (localSettings[key] !== undefined) {
                if (key === 'base_model_path_mappings' || key === 'download_path_templates') {
                    payload[key] = JSON.stringify(localSettings[key]);
                } else {
                    payload[key] = localSettings[key];
                }
            }
        });

        // Only send request if there is something to sync
        if (Object.keys(payload).length > 0) {
            try {
                await fetch('/api/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                // Log success to console
                console.log('Settings synced to backend');
            } catch (e) {
                // Log error to console
                console.error('Failed to sync settings to backend:', e);
            }
        }
    }

    initialize() {
        if (this.initialized) return;
        
        // Add event listener to sync state when modal is closed via other means (like Escape key)
        const settingsModal = document.getElementById('settingsModal');
        if (settingsModal) {
            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                        this.isOpen = settingsModal.style.display === 'block';
                        
                        // When modal is opened, update checkbox state from current settings
                        if (this.isOpen) {
                            this.loadSettingsToUI();
                        }
                    }
                });
            });
            
            observer.observe(settingsModal, { attributes: true });
        }
        
        // Add event listeners for all toggle-visibility buttons
        document.querySelectorAll('.toggle-visibility').forEach(button => {
            button.addEventListener('click', () => this.toggleInputVisibility(button));
        });

        ['lora', 'checkpoint', 'embedding'].forEach(modelType => {
            const customInput = document.getElementById(`${modelType}CustomTemplate`);
            if (customInput) {
                customInput.addEventListener('input', (e) => {
                    const template = e.target.value;
                    settingsManager.validateTemplate(modelType, template);
                    settingsManager.updateTemplatePreview(modelType, template);
                });
                
                customInput.addEventListener('blur', (e) => {
                    const template = e.target.value;
                    if (settingsManager.validateTemplate(modelType, template)) {
                        settingsManager.updateTemplate(modelType, template);
                    }
                });
                
                customInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        e.target.blur();
                    }
                });
            }
        });
        
        this.initialized = true;
    }

    async loadSettingsToUI() {
        // Set frontend settings from state
        const blurMatureContentCheckbox = document.getElementById('blurMatureContent');
        if (blurMatureContentCheckbox) {
            blurMatureContentCheckbox.checked = state.global.settings.blurMatureContent;
        }
        
        const showOnlySFWCheckbox = document.getElementById('showOnlySFW');
        if (showOnlySFWCheckbox) {
            // Sync with state (backend will set this via template)
            state.global.settings.show_only_sfw = showOnlySFWCheckbox.checked;
        }
        
        // Set video autoplay on hover setting
        const autoplayOnHoverCheckbox = document.getElementById('autoplayOnHover');
        if (autoplayOnHoverCheckbox) {
            autoplayOnHoverCheckbox.checked = state.global.settings.autoplayOnHover || false;
        }
        
        // Set display density setting
        const displayDensitySelect = document.getElementById('displayDensity');
        if (displayDensitySelect) {
            displayDensitySelect.value = state.global.settings.displayDensity || 'default';
        }
        
        // Set card info display setting
        const cardInfoDisplaySelect = document.getElementById('cardInfoDisplay');
        if (cardInfoDisplaySelect) {
            cardInfoDisplaySelect.value = state.global.settings.cardInfoDisplay || 'always';
        }

        // Set optimize example images setting
        const optimizeExampleImagesCheckbox = document.getElementById('optimizeExampleImages');
        if (optimizeExampleImagesCheckbox) {
            optimizeExampleImagesCheckbox.checked = state.global.settings.optimizeExampleImages || false;
        }

        // Set auto download example images setting
        const autoDownloadExampleImagesCheckbox = document.getElementById('autoDownloadExampleImages');
        if (autoDownloadExampleImagesCheckbox) {
            autoDownloadExampleImagesCheckbox.checked = state.global.settings.autoDownloadExampleImages || false;
        }

        // Load download path templates
        this.loadDownloadPathTemplates();

        // Set include trigger words setting
        const includeTriggerWordsCheckbox = document.getElementById('includeTriggerWords');
        if (includeTriggerWordsCheckbox) {
            includeTriggerWordsCheckbox.checked = state.global.settings.includeTriggerWords || false;
        }

        // Load base model path mappings
        this.loadBaseModelMappings();

        // Load default lora root
        await this.loadLoraRoots();
        
        // Load default checkpoint root
        await this.loadCheckpointRoots();

        // Load default embedding root
        await this.loadEmbeddingRoots();
    }

    async loadLoraRoots() {
        try {
            const defaultLoraRootSelect = document.getElementById('defaultLoraRoot');
            if (!defaultLoraRootSelect) return;
            
            // Fetch lora roots
            const response = await fetch('/api/loras/roots');
            if (!response.ok) {
                throw new Error('Failed to fetch LoRA roots');
            }
            
            const data = await response.json();
            if (!data.roots || data.roots.length === 0) {
                throw new Error('No LoRA roots found');
            }
            
            // Clear existing options except the first one (No Default)
            const noDefaultOption = defaultLoraRootSelect.querySelector('option[value=""]');
            defaultLoraRootSelect.innerHTML = '';
            defaultLoraRootSelect.appendChild(noDefaultOption);
            
            // Add options for each root
            data.roots.forEach(root => {
                const option = document.createElement('option');
                option.value = root;
                option.textContent = root;
                defaultLoraRootSelect.appendChild(option);
            });
            
            // Set selected value from settings
            const defaultRoot = state.global.settings.default_lora_root || '';
            defaultLoraRootSelect.value = defaultRoot;
            
        } catch (error) {
            console.error('Error loading LoRA roots:', error);
            showToast('Failed to load LoRA roots: ' + error.message, 'error');
        }
    }

    async loadCheckpointRoots() {
        try {
            const defaultCheckpointRootSelect = document.getElementById('defaultCheckpointRoot');
            if (!defaultCheckpointRootSelect) return;
            
            // Fetch checkpoint roots
            const response = await fetch('/api/checkpoints/roots');
            if (!response.ok) {
                throw new Error('Failed to fetch checkpoint roots');
            }
            
            const data = await response.json();
            if (!data.roots || data.roots.length === 0) {
                throw new Error('No checkpoint roots found');
            }
            
            // Clear existing options except the first one (No Default)
            const noDefaultOption = defaultCheckpointRootSelect.querySelector('option[value=""]');
            defaultCheckpointRootSelect.innerHTML = '';
            defaultCheckpointRootSelect.appendChild(noDefaultOption);
            
            // Add options for each root
            data.roots.forEach(root => {
                const option = document.createElement('option');
                option.value = root;
                option.textContent = root;
                defaultCheckpointRootSelect.appendChild(option);
            });
            
            // Set selected value from settings
            const defaultRoot = state.global.settings.default_checkpoint_root || '';
            defaultCheckpointRootSelect.value = defaultRoot;
            
        } catch (error) {
            console.error('Error loading checkpoint roots:', error);
            showToast('Failed to load checkpoint roots: ' + error.message, 'error');
        }
    }

    async loadEmbeddingRoots() {
        try {
            const defaultEmbeddingRootSelect = document.getElementById('defaultEmbeddingRoot');
            if (!defaultEmbeddingRootSelect) return;

            // Fetch embedding roots
            const response = await fetch('/api/embeddings/roots');
            if (!response.ok) {
                throw new Error('Failed to fetch embedding roots');
            }

            const data = await response.json();
            if (!data.roots || data.roots.length === 0) {
                throw new Error('No embedding roots found');
            }

            // Clear existing options except the first one (No Default)
            const noDefaultOption = defaultEmbeddingRootSelect.querySelector('option[value=""]');
            defaultEmbeddingRootSelect.innerHTML = '';
            defaultEmbeddingRootSelect.appendChild(noDefaultOption);

            // Add options for each root
            data.roots.forEach(root => {
                const option = document.createElement('option');
                option.value = root;
                option.textContent = root;
                defaultEmbeddingRootSelect.appendChild(option);
            });

            // Set selected value from settings
            const defaultRoot = state.global.settings.default_embedding_root || '';
            defaultEmbeddingRootSelect.value = defaultRoot;

        } catch (error) {
            console.error('Error loading embedding roots:', error);
            showToast('Failed to load embedding roots: ' + error.message, 'error');
        }
    }

    loadBaseModelMappings() {
        const mappingsContainer = document.getElementById('baseModelMappingsContainer');
        if (!mappingsContainer) return;

        const mappings = state.global.settings.base_model_path_mappings || {};
        
        // Clear existing mappings
        mappingsContainer.innerHTML = '';

        // Add existing mappings
        Object.entries(mappings).forEach(([baseModel, pathValue]) => {
            this.addMappingRow(baseModel, pathValue);
        });

        // Add empty row for new mappings if none exist
        if (Object.keys(mappings).length === 0) {
            this.addMappingRow('', '');
        }
    }

    addMappingRow(baseModel = '', pathValue = '') {
        const mappingsContainer = document.getElementById('baseModelMappingsContainer');
        if (!mappingsContainer) return;

        const row = document.createElement('div');
        row.className = 'mapping-row';
        
        const availableModels = MAPPABLE_BASE_MODELS.filter(model => {
            const existingMappings = state.global.settings.base_model_path_mappings || {};
            return !existingMappings.hasOwnProperty(model) || model === baseModel;
        });

        row.innerHTML = `
            <div class="mapping-controls">
                <select class="base-model-select">
                    <option value="">Select Base Model</option>
                    ${availableModels.map(model => 
                        `<option value="${model}" ${model === baseModel ? 'selected' : ''}>${model}</option>`
                    ).join('')}
                </select>
                <input type="text" class="path-value-input" placeholder="Custom path (e.g., flux)" value="${pathValue}">
                <button type="button" class="remove-mapping-btn" title="Remove mapping">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;

        // Add event listeners
        const baseModelSelect = row.querySelector('.base-model-select');
        const pathValueInput = row.querySelector('.path-value-input');
        const removeBtn = row.querySelector('.remove-mapping-btn');

        // Save on select change immediately
        baseModelSelect.addEventListener('change', () => this.updateBaseModelMappings());
        
        // Save on input blur or Enter key
        pathValueInput.addEventListener('blur', () => this.updateBaseModelMappings());
        pathValueInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.target.blur();
            }
        });
        
        removeBtn.addEventListener('click', () => {
            row.remove();
            this.updateBaseModelMappings();
        });

        mappingsContainer.appendChild(row);
    }

    updateBaseModelMappings() {
        const mappingsContainer = document.getElementById('baseModelMappingsContainer');
        if (!mappingsContainer) return;

        const rows = mappingsContainer.querySelectorAll('.mapping-row');
        const newMappings = {};
        let hasValidMapping = false;

        rows.forEach(row => {
            const baseModelSelect = row.querySelector('.base-model-select');
            const pathValueInput = row.querySelector('.path-value-input');
            
            const baseModel = baseModelSelect.value.trim();
            const pathValue = pathValueInput.value.trim();
            
            if (baseModel && pathValue) {
                newMappings[baseModel] = pathValue;
                hasValidMapping = true;
            }
        });

        // Check if mappings have actually changed
        const currentMappings = state.global.settings.base_model_path_mappings || {};
        const mappingsChanged = JSON.stringify(currentMappings) !== JSON.stringify(newMappings);

        if (mappingsChanged) {
            // Update state and save
            state.global.settings.base_model_path_mappings = newMappings;
            this.saveBaseModelMappings();
        }

        // Add empty row if no valid mappings exist
        const hasEmptyRow = Array.from(rows).some(row => {
            const baseModelSelect = row.querySelector('.base-model-select');
            const pathValueInput = row.querySelector('.path-value-input');
            return !baseModelSelect.value && !pathValueInput.value;
        });

        if (!hasEmptyRow) {
            this.addMappingRow('', '');
        }

        // Update available options in all selects
        this.updateAvailableBaseModels();
    }

    updateAvailableBaseModels() {
        const mappingsContainer = document.getElementById('baseModelMappingsContainer');
        if (!mappingsContainer) return;

        const existingMappings = state.global.settings.base_model_path_mappings || {};
        const rows = mappingsContainer.querySelectorAll('.mapping-row');

        rows.forEach(row => {
            const select = row.querySelector('.base-model-select');
            const currentValue = select.value;
            
            // Get available models (not already mapped, except current)
            const availableModels = MAPPABLE_BASE_MODELS.filter(model => 
                !existingMappings.hasOwnProperty(model) || model === currentValue
            );

            // Rebuild options
            select.innerHTML = '<option value="">Select Base Model</option>' +
                availableModels.map(model => 
                    `<option value="${model}" ${model === currentValue ? 'selected' : ''}>${model}</option>`
                ).join('');
        });
    }

    async saveBaseModelMappings() {
        try {
            // Save to localStorage
            setStorageItem('settings', state.global.settings);

            // Save to backend
            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    base_model_path_mappings: JSON.stringify(state.global.settings.base_model_path_mappings)
                })
            });

            if (!response.ok) {
                throw new Error('Failed to save base model mappings');
            }

            // Show success toast
            const mappingCount = Object.keys(state.global.settings.base_model_path_mappings).length;
            if (mappingCount > 0) {
                showToast(`Base model path mappings updated (${mappingCount} mapping${mappingCount !== 1 ? 's' : ''})`, 'success');
            } else {
                showToast('Base model path mappings cleared', 'success');
            }

        } catch (error) {
            console.error('Error saving base model mappings:', error);
            showToast('Failed to save base model mappings: ' + error.message, 'error');
        }
    }

    loadDownloadPathTemplates() {
        const templates = state.global.settings.download_path_templates || DEFAULT_PATH_TEMPLATES;
        
        Object.keys(templates).forEach(modelType => {
            this.loadTemplateForModelType(modelType, templates[modelType]);
        });
    }

    loadTemplateForModelType(modelType, template) {
        const presetSelect = document.getElementById(`${modelType}TemplatePreset`);
        const customRow = document.getElementById(`${modelType}CustomRow`);
        const customInput = document.getElementById(`${modelType}CustomTemplate`);
        
        if (!presetSelect) return;

        // Find matching preset
        const matchingPreset = this.findMatchingPreset(template);
        
        if (matchingPreset !== null) {
            presetSelect.value = matchingPreset;
            if (customRow) customRow.style.display = 'none';
        } else {
            // Custom template
            presetSelect.value = 'custom';
            if (customRow) customRow.style.display = 'block';
            if (customInput) {
                customInput.value = template;
                this.validateTemplate(modelType, template);
            }
        }
        
        this.updateTemplatePreview(modelType, template);
    }

    findMatchingPreset(template) {
        const presetValues = Object.values(DOWNLOAD_PATH_TEMPLATES)
            .map(t => t.value)
            .filter(v => v !== 'custom');
        
        return presetValues.includes(template) ? template : null;
    }

    updateTemplatePreset(modelType, value) {
        const customRow = document.getElementById(`${modelType}CustomRow`);
        const customInput = document.getElementById(`${modelType}CustomTemplate`);
        
        if (value === 'custom') {
            if (customRow) customRow.style.display = 'block';
            if (customInput) customInput.focus();
            return;
        } else {
            if (customRow) customRow.style.display = 'none';
        }
        
        // Update template
        this.updateTemplate(modelType, value);
    }

    updateTemplate(modelType, template) {
        // Validate template if it's custom
        if (document.getElementById(`${modelType}TemplatePreset`).value === 'custom') {
            if (!this.validateTemplate(modelType, template)) {
                return; // Don't save invalid templates
            }
        }
        
        // Update state
        if (!state.global.settings.download_path_templates) {
            state.global.settings.download_path_templates = { ...DEFAULT_PATH_TEMPLATES };
        }
        state.global.settings.download_path_templates[modelType] = template;
        
        // Update preview
        this.updateTemplatePreview(modelType, template);
        
        // Save settings
        this.saveDownloadPathTemplates();
    }

    validateTemplate(modelType, template) {
        const validationElement = document.getElementById(`${modelType}Validation`);
        if (!validationElement) return true;
        
        // Reset validation state
        validationElement.innerHTML = '';
        validationElement.className = 'template-validation';
        
        if (!template) {
            validationElement.innerHTML = '<i class="fas fa-check"></i> Valid (flat structure)';
            validationElement.classList.add('valid');
            return true;
        }
        
        // Check for invalid characters
        const invalidChars = /[<>:"|?*]/;
        if (invalidChars.test(template)) {
            validationElement.innerHTML = '<i class="fas fa-times"></i> Invalid characters detected';
            validationElement.classList.add('invalid');
            return false;
        }
        
        // Check for double slashes
        if (template.includes('//')) {
            validationElement.innerHTML = '<i class="fas fa-times"></i> Double slashes not allowed';
            validationElement.classList.add('invalid');
            return false;
        }
        
        // Check if it starts or ends with slash
        if (template.startsWith('/') || template.endsWith('/')) {
            validationElement.innerHTML = '<i class="fas fa-times"></i> Cannot start or end with slash';
            validationElement.classList.add('invalid');
            return false;
        }
        
        // Extract placeholders
        const placeholderRegex = /\{([^}]+)\}/g;
        const matches = template.match(placeholderRegex) || [];
        
        // Check for invalid placeholders
        const invalidPlaceholders = matches.filter(match => 
            !PATH_TEMPLATE_PLACEHOLDERS.includes(match)
        );
        
        if (invalidPlaceholders.length > 0) {
            validationElement.innerHTML = `<i class="fas fa-times"></i> Invalid placeholder: ${invalidPlaceholders[0]}`;
            validationElement.classList.add('invalid');
            return false;
        }
        
        // Template is valid
        validationElement.innerHTML = '<i class="fas fa-check"></i> Valid template';
        validationElement.classList.add('valid');
        return true;
    }

    updateTemplatePreview(modelType, template) {
        const previewElement = document.getElementById(`${modelType}Preview`);
        if (!previewElement) return;
        
        if (!template) {
            previewElement.textContent = 'model-name.safetensors';
        } else {
            // Generate example preview
            const exampleTemplate = template
                .replace('{base_model}', 'Flux.1 D')
                .replace('{author}', 'authorname')
                .replace('{first_tag}', 'style');
            previewElement.textContent = `${exampleTemplate}/model-name.safetensors`;
        }
        previewElement.style.display = 'block';
    }

    async saveDownloadPathTemplates() {
        try {
            // Save to localStorage
            setStorageItem('settings', state.global.settings);

            // Save to backend
            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    download_path_templates: JSON.stringify(state.global.settings.download_path_templates)
                })
            });

            if (!response.ok) {
                throw new Error('Failed to save download path templates');
            }

            showToast('Download path templates updated', 'success');

        } catch (error) {
            console.error('Error saving download path templates:', error);
            showToast('Failed to save download path templates: ' + error.message, 'error');
        }
    }

    toggleSettings() {
        if (this.isOpen) {
            modalManager.closeModal('settingsModal');
        } else {
            modalManager.showModal('settingsModal');
        }
        this.isOpen = !this.isOpen;
    }

    async saveToggleSetting(elementId, settingKey) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        const value = element.checked;
        
        // Update frontend state
        if (settingKey === 'blur_mature_content') {
            state.global.settings.blurMatureContent = value;
        } else if (settingKey === 'show_only_sfw') {
            state.global.settings.show_only_sfw = value;
        } else if (settingKey === 'autoplay_on_hover') {
            state.global.settings.autoplayOnHover = value;
        } else if (settingKey === 'optimize_example_images') {
            state.global.settings.optimizeExampleImages = value;
        } else if (settingKey === 'auto_download_example_images') {
            state.global.settings.autoDownloadExampleImages = value;
        } else if (settingKey === 'compact_mode') {
            state.global.settings.compactMode = value;
        } else if (settingKey === 'include_trigger_words') {
            state.global.settings.includeTriggerWords = value;
        } else {
            // For any other settings that might be added in the future
            state.global.settings[settingKey] = value;
        }
        
        // Save to localStorage
        setStorageItem('settings', state.global.settings);
        
        try {
            // For backend settings, make API call
            if (['show_only_sfw'].includes(settingKey)) {
                const payload = {};
                payload[settingKey] = value;
                
                const response = await fetch('/api/settings', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    throw new Error('Failed to save setting');
                }
            }
                
            showToast(`Settings updated: ${settingKey.replace(/_/g, ' ')}`, 'success');
            
            // Apply frontend settings immediately
            this.applyFrontendSettings();
            
            // Trigger auto download setup/teardown when setting changes
            if (settingKey === 'auto_download_example_images' && window.exampleImagesManager) {
                if (value) {
                    window.exampleImagesManager.setupAutoDownload();
                } else {
                    window.exampleImagesManager.clearAutoDownload();
                }
            }
            
            if (settingKey === 'show_only_sfw' || settingKey === 'blur_mature_content') {
                this.reloadContent();
            }
            
            // Recalculate layout when compact mode changes
            if (settingKey === 'compact_mode' && state.virtualScroller) {
                state.virtualScroller.calculateLayout();
                showToast(`Compact Mode ${value ? 'enabled' : 'disabled'}`, 'success');
            }
            
        } catch (error) {
            showToast('Failed to save setting: ' + error.message, 'error');
        }
    }
    
    async saveSelectSetting(elementId, settingKey) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        const value = element.value;
        
        // Update frontend state
        if (settingKey === 'default_lora_root') {
            state.global.settings.default_lora_root = value;
        } else if (settingKey === 'default_checkpoint_root') {
            state.global.settings.default_checkpoint_root = value;
        } else if (settingKey === 'default_embedding_root') {
            state.global.settings.default_embedding_root = value;
        } else if (settingKey === 'display_density') {
            state.global.settings.displayDensity = value;
            
            // Also update compactMode for backwards compatibility
            state.global.settings.compactMode = (value !== 'default');
        } else if (settingKey === 'card_info_display') {
            state.global.settings.cardInfoDisplay = value;
        } else {
            // For any other settings that might be added in the future
            state.global.settings[settingKey] = value;
        }
        
        // Save to localStorage
        setStorageItem('settings', state.global.settings);
        
        try {
            // For backend settings, make API call
            if (settingKey === 'default_lora_root' || settingKey === 'default_checkpoint_root' || settingKey === 'default_embedding_root' || settingKey === 'download_path_templates') {
                const payload = {};
                if (settingKey === 'download_path_templates') {
                    payload[settingKey] = JSON.stringify(state.global.settings.download_path_templates);
                } else {
                    payload[settingKey] = value;
                }
                
                const response = await fetch('/api/settings', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    throw new Error('Failed to save setting');
                }
                
                showToast(`Settings updated: ${settingKey.replace(/_/g, ' ')}`, 'success');
            }
            
            // Apply frontend settings immediately
            this.applyFrontendSettings();
            
            // Recalculate layout when display density changes
            if (settingKey === 'display_density' && state.virtualScroller) {
                state.virtualScroller.calculateLayout();
                
                let densityName = "Default";
                if (value === 'medium') densityName = "Medium";
                if (value === 'compact') densityName = "Compact";
                
                showToast(`Display Density set to ${densityName}`, 'success');
            }
            
        } catch (error) {
            showToast('Failed to save setting: ' + error.message, 'error');
        }
    }
    
    async saveInputSetting(elementId, settingKey) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        const value = element.value;
        
        // For API key or other inputs that need to be saved on backend
        try {
            // Check if value has changed from existing value
            const currentValue = state.global.settings[settingKey] || '';
            if (value === currentValue) {
                return; // No change, exit early
            }
            
            // Update state
            state.global.settings[settingKey] = value;
            
            setStorageItem('settings', state.global.settings);
            
            // For backend settings, make API call
            const payload = {};
            payload[settingKey] = value;
            
            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error('Failed to save setting');
            }
            
            showToast(`Settings updated: ${settingKey.replace(/_/g, ' ')}`, 'success');
            
        } catch (error) {
            showToast('Failed to save setting: ' + error.message, 'error');
        }
    }

    toggleInputVisibility(button) {
        const input = button.parentElement.querySelector('input');
        const icon = button.querySelector('i');
        
        if (input.type === 'password') {
            input.type = 'text';
            icon.className = 'fas fa-eye-slash';
        } else {
            input.type = 'password';
            icon.className = 'fas fa-eye';
        }
    }

    confirmClearCache() {
        // Show confirmation modal
        modalManager.showModal('clearCacheModal');
    }

    async executeClearCache() {
        try {
            // Call the API endpoint to clear cache files
            const response = await fetch('/api/clear-cache', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const result = await response.json();
            
            if (result.success) {
                showToast('Cache files have been cleared successfully. Cache will rebuild on next action.', 'success');
            } else {
                showToast(`Failed to clear cache: ${result.error}`, 'error');
            }
            
            // Close the confirmation modal
            modalManager.closeModal('clearCacheModal');
        } catch (error) {
            showToast(`Error clearing cache: ${error.message}`, 'error');
            modalManager.closeModal('clearCacheModal');
        }
    }

    async reloadContent() {
        if (this.currentPage === 'loras') {
            // Reload the loras without updating folders
            await resetAndReload(false);
        } else if (this.currentPage === 'recipes') {
            // Reload the recipes without updating folders
            await window.recipeManager.loadRecipes();
        } else if (this.currentPage === 'checkpoints') {
            // Reload the checkpoints without updating folders
            await resetAndReload(false);
        } else if (this.currentPage === 'embeddings') {
            // Reload the embeddings without updating folders
            await resetAndReload(false);
        }
    }

    applyFrontendSettings() {
        // Apply autoplay setting to existing videos in card previews
        const autoplayOnHover = state.global.settings.autoplayOnHover;
        document.querySelectorAll('.card-preview video').forEach(video => {
            // Remove previous event listeners by cloning and replacing the element
            const videoParent = video.parentElement;
            const videoClone = video.cloneNode(true);
            
            if (autoplayOnHover) {
                // Pause video initially and set up mouse events for hover playback
                videoClone.removeAttribute('autoplay');
                videoClone.pause();
                
                // Add mouse events to the parent element
                videoParent.onmouseenter = () => videoClone.play();
                videoParent.onmouseleave = () => {
                    videoClone.pause();
                    videoClone.currentTime = 0;
                };
            } else {
                // Use default autoplay behavior
                videoClone.setAttribute('autoplay', '');
                videoParent.onmouseenter = null;
                videoParent.onmouseleave = null;
            }
            
            videoParent.replaceChild(videoClone, video);
        });
        
        // Apply display density class to grid
        const grid = document.querySelector('.card-grid');
        if (grid) {
            const density = state.global.settings.displayDensity || 'default';
            
            // Remove all density classes first
            grid.classList.remove('default-density', 'medium-density', 'compact-density');
            
            // Add the appropriate density class
            grid.classList.add(`${density}-density`);
        }
        
        // Apply card info display setting
        const cardInfoDisplay = state.global.settings.cardInfoDisplay || 'always';
        document.body.classList.toggle('hover-reveal', cardInfoDisplay === 'hover');
    }
}

// Create singleton instance
export const settingsManager = new SettingsManager();
