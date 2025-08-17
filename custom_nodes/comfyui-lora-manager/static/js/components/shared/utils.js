/**
 * utils.js
 * Helper functions for the Model Modal component - General version
 */

/**
 * Format file size
 * @param {number} bytes - Number of bytes
 * @returns {string} Formatted file size
 */
export function formatFileSize(bytes) {
    if (!bytes) return 'N/A';
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
}

/**
 * Render compact tags
 * @param {Array} tags - Array of tags
 * @param {string} filePath - File path for the edit button
 * @returns {string} HTML content
 */
export function renderCompactTags(tags, filePath = '') {
    // Remove the early return and always render the container
    const tagsList = tags || [];
    
    // Display up to 5 tags, with a tooltip indicator if there are more
    const visibleTags = tagsList.slice(0, 5);
    const remainingCount = Math.max(0, tagsList.length - 5);
    
    return `
        <div class="model-tags-container">
            <div class="model-tags-header">
                <div class="model-tags-compact">
                    ${visibleTags.map(tag => `<span class="model-tag-compact">${tag}</span>`).join('')}
                    ${remainingCount > 0 ? 
                        `<span class="model-tag-more" data-count="${remainingCount}">+${remainingCount}</span>` : 
                        ''}
                    ${tagsList.length === 0 ? `<span class="model-tag-empty">No tags</span>` : ''}
                </div>
                <button class="edit-tags-btn" data-file-path="${filePath}" title="Edit tags">
                    <i class="fas fa-pencil-alt"></i>
                </button>
            </div>
            ${tagsList.length > 0 ? 
                `<div class="model-tags-tooltip">
                    <div class="tooltip-content">
                        ${tagsList.map(tag => `<span class="tooltip-tag">${tag}</span>`).join('')}
                    </div>
                </div>` : 
                ''}
        </div>
    `;
}

/**
 * Set up tag tooltip functionality
 */
export function setupTagTooltip() {
    const tagsContainer = document.querySelector('.model-tags-container');
    const tooltip = document.querySelector('.model-tags-tooltip');
    
    if (tagsContainer && tooltip) {
        tagsContainer.addEventListener('mouseenter', () => {
            tooltip.classList.add('visible');
        });
        
        tagsContainer.addEventListener('mouseleave', () => {
            tooltip.classList.remove('visible');
        });
    }
}