/**
 * PresetTags.js
 * Handles LoRA model preset parameter tags - Shared version
 */
import { getModelApiClient } from '../../api/modelApiFactory.js';

/**
 * Parse preset parameters
 * @param {string} usageTips - JSON string containing preset parameters
 * @returns {Object} Parsed preset parameters object
 */
export function parsePresets(usageTips) {
    if (!usageTips) return {};
    try {
        return JSON.parse(usageTips);
    } catch {
        return {};
    }
}

/**
 * Render preset tags
 * @param {Object} presets - Preset parameters object
 * @returns {string} HTML content
 */
export function renderPresetTags(presets) {
    return Object.entries(presets).map(([key, value]) => `
        <div class="preset-tag" data-key="${key}">
            <span>${formatPresetKey(key)}: ${value}</span>
            <i class="fas fa-times" onclick="removePreset('${key}')"></i>
        </div>
    `).join('');
}

/**
 * Format preset key name
 * @param {string} key - Preset key name
 * @returns {string} Formatted key name
 */
function formatPresetKey(key) {
    return key.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
}

/**
 * Remove preset parameter
 * @param {string} key - Preset key name to remove
 */
window.removePreset = async function(key) {
    const filePath = document.querySelector('#modelModal .modal-content')
            .querySelector('.file-path').textContent + 
            document.querySelector('#modelModal .modal-content')
            .querySelector('#file-name').textContent + '.safetensors';
    const loraCard = document.querySelector(`.model-card[data-filepath="${filePath}"]`);
    const currentPresets = parsePresets(loraCard.dataset.usage_tips);
    
    delete currentPresets[key];
    const newPresetsJson = JSON.stringify(currentPresets);

    await getModelApiClient().saveModelMetadata(filePath, { 
        usage_tips: newPresetsJson 
    });
    
    document.querySelector('.preset-tags').innerHTML = renderPresetTags(currentPresets);
};