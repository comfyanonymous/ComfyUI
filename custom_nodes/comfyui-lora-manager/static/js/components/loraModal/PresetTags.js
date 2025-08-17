/**
 * PresetTags.js
 * 处理LoRA模型预设参数标签相关的功能模块
 */
import { saveModelMetadata } from '../../api/loraApi.js';

/**
 * 解析预设参数
 * @param {string} usageTips - 包含预设参数的JSON字符串
 * @returns {Object} 解析后的预设参数对象
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
 * 渲染预设标签
 * @param {Object} presets - 预设参数对象
 * @returns {string} HTML内容
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
 * 格式化预设键名
 * @param {string} key - 预设键名
 * @returns {string} 格式化后的键名
 */
function formatPresetKey(key) {
    return key.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
}

/**
 * 移除预设参数
 * @param {string} key - 要移除的预设键名
 */
window.removePreset = async function(key) {
    const filePath = document.querySelector('#loraModal .modal-content')
            .querySelector('.file-path').textContent + 
            document.querySelector('#loraModal .modal-content')
            .querySelector('#file-name').textContent + '.safetensors';
    const loraCard = document.querySelector(`.lora-card[data-filepath="${filePath}"]`);
    const currentPresets = parsePresets(loraCard.dataset.usage_tips);
    
    delete currentPresets[key];
    const newPresetsJson = JSON.stringify(currentPresets);

    await saveModelMetadata(filePath, { 
        usage_tips: newPresetsJson 
    });
    
    document.querySelector('.preset-tags').innerHTML = renderPresetTags(currentPresets);
};