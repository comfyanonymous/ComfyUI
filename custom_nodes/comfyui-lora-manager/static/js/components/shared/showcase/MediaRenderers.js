/**
 * MediaRenderers.js
 * HTML generators for media items (images/videos) in the showcase
 */

/**
 * Generate video wrapper HTML
 * @param {Object} media - Media metadata
 * @param {number} heightPercent - Height percentage for container
 * @param {boolean} shouldBlur - Whether content should be blurred
 * @param {string} nsfwText - NSFW warning text
 * @param {string} metadataPanel - Metadata panel HTML
 * @param {string} localUrl - Local file URL
 * @param {string} remoteUrl - Remote file URL
 * @param {string} mediaControlsHtml - HTML for media control buttons
 * @returns {string} HTML content
 */
export function generateVideoWrapper(media, heightPercent, shouldBlur, nsfwText, metadataPanel, localUrl, remoteUrl, mediaControlsHtml = '') {
    const nsfwLevel = media.nsfwLevel !== undefined ? media.nsfwLevel : 0;
    
    return `
        <div class="media-wrapper ${shouldBlur ? 'nsfw-media-wrapper' : ''}" style="padding-bottom: ${heightPercent}%" data-short-id="${media.id || ''}" data-nsfw-level="${nsfwLevel}">
            ${shouldBlur ? `
                <button class="toggle-blur-btn showcase-toggle-btn" title="Toggle blur">
                    <i class="fas fa-eye"></i>
                </button>
            ` : ''}
            ${mediaControlsHtml}
            <video controls autoplay muted loop crossorigin="anonymous" 
                referrerpolicy="no-referrer" 
                data-local-src="${localUrl || ''}"
                data-remote-src="${remoteUrl}"
                data-nsfw-level="${nsfwLevel}"
                class="lazy ${shouldBlur ? 'blurred' : ''}">
                <source data-local-src="${localUrl || ''}" data-remote-src="${remoteUrl}" type="video/mp4">
                Your browser does not support video playback
            </video>
            ${shouldBlur ? `
                <div class="nsfw-overlay">
                    <div class="nsfw-warning">
                        <p>${nsfwText}</p>
                        <button class="show-content-btn">Show</button>
                    </div>
                </div>
            ` : ''}
            ${metadataPanel}
        </div>
    `;
}

/**
 * Generate image wrapper HTML
 * @param {Object} media - Media metadata
 * @param {number} heightPercent - Height percentage for container
 * @param {boolean} shouldBlur - Whether content should be blurred
 * @param {string} nsfwText - NSFW warning text
 * @param {string} metadataPanel - Metadata panel HTML
 * @param {string} localUrl - Local file URL
 * @param {string} remoteUrl - Remote file URL
 * @param {string} mediaControlsHtml - HTML for media control buttons
 * @returns {string} HTML content
 */
export function generateImageWrapper(media, heightPercent, shouldBlur, nsfwText, metadataPanel, localUrl, remoteUrl, mediaControlsHtml = '') {
    const nsfwLevel = media.nsfwLevel !== undefined ? media.nsfwLevel : 0;
    
    return `
        <div class="media-wrapper ${shouldBlur ? 'nsfw-media-wrapper' : ''}" style="padding-bottom: ${heightPercent}%" data-short-id="${media.id || ''}" data-nsfw-level="${nsfwLevel}">
            ${shouldBlur ? `
                <button class="toggle-blur-btn showcase-toggle-btn" title="Toggle blur">
                    <i class="fas fa-eye"></i>
                </button>
            ` : ''}
            ${mediaControlsHtml}
            <img data-local-src="${localUrl || ''}" 
                data-remote-src="${remoteUrl}"
                data-nsfw-level="${nsfwLevel}"
                alt="Preview" 
                crossorigin="anonymous" 
                referrerpolicy="no-referrer"
                width="${media.width}"
                height="${media.height}"
                class="lazy ${shouldBlur ? 'blurred' : ''}"> 
            ${shouldBlur ? `
                <div class="nsfw-overlay">
                    <div class="nsfw-warning">
                        <p>${nsfwText}</p>
                        <button class="show-content-btn">Show</button>
                    </div>
                </div>
            ` : ''}
            ${metadataPanel}
        </div>
    `;
}