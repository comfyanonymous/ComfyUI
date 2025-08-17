/**
 * MetadataPanel.js
 * Generates metadata panels for showcase media items
 */

/**
 * Generate metadata panel HTML
 * @param {boolean} hasParams - Whether there are generation parameters
 * @param {boolean} hasPrompts - Whether there are prompts
 * @param {string} prompt - Prompt text
 * @param {string} negativePrompt - Negative prompt text
 * @param {string} size - Image size
 * @param {string} seed - Generation seed
 * @param {string} model - Model used
 * @param {string} steps - Steps used
 * @param {string} sampler - Sampler used
 * @param {string} cfgScale - CFG scale
 * @param {string} clipSkip - Clip skip value
 * @returns {string} HTML content
 */
export function generateMetadataPanel(hasParams, hasPrompts, prompt, negativePrompt, size, seed, model, steps, sampler, cfgScale, clipSkip) {
    // Create unique IDs for prompt copying
    const promptIndex = Math.random().toString(36).substring(2, 15);
    const negPromptIndex = Math.random().toString(36).substring(2, 15);
    
    let content = '<div class="image-metadata-panel"><div class="metadata-content">';
    
    if (hasParams) {
        content += `
            <div class="params-tags">
                ${size ? `<div class="param-tag"><span class="param-name">Size:</span><span class="param-value">${size}</span></div>` : ''}
                ${seed ? `<div class="param-tag"><span class="param-name">Seed:</span><span class="param-value">${seed}</span></div>` : ''}
                ${model ? `<div class="param-tag"><span class="param-name">Model:</span><span class="param-value">${model}</span></div>` : ''}
                ${steps ? `<div class="param-tag"><span class="param-name">Steps:</span><span class="param-value">${steps}</span></div>` : ''}
                ${sampler ? `<div class="param-tag"><span class="param-name">Sampler:</span><span class="param-value">${sampler}</span></div>` : ''}
                ${cfgScale ? `<div class="param-tag"><span class="param-name">CFG:</span><span class="param-value">${cfgScale}</span></div>` : ''}
                ${clipSkip ? `<div class="param-tag"><span class="param-name">Clip Skip:</span><span class="param-value">${clipSkip}</span></div>` : ''}
            </div>
        `;
    }
    
    if (!hasParams && !hasPrompts) {
        content += `
            <div class="no-metadata-message">
                <i class="fas fa-info-circle"></i>
                <span>No generation parameters available</span>
            </div>
        `;
    }
    
    if (prompt) {
        content += `
            <div class="metadata-row prompt-row">
                <span class="metadata-label">Prompt:</span>
                <div class="metadata-prompt-wrapper">
                    <div class="metadata-prompt">${prompt}</div>
                    <button class="copy-prompt-btn" data-prompt-index="${promptIndex}">
                        <i class="fas fa-copy"></i>
                    </button>
                </div>
            </div>
            <div class="hidden-prompt" id="prompt-${promptIndex}" style="display:none;">${prompt}</div>
        `;
    }
    
    if (negativePrompt) {
        content += `
            <div class="metadata-row prompt-row">
                <span class="metadata-label">Negative Prompt:</span>
                <div class="metadata-prompt-wrapper">
                    <div class="metadata-prompt">${negativePrompt}</div>
                    <button class="copy-prompt-btn" data-prompt-index="${negPromptIndex}">
                        <i class="fas fa-copy"></i>
                    </button>
                </div>
            </div>
            <div class="hidden-prompt" id="prompt-${negPromptIndex}" style="display:none;">${negativePrompt}</div>
        `;
    }
    
    content += '</div></div>';
    return content;
}