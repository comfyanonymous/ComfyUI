/**
 * ShowcaseView.js
 * Shared showcase component for displaying examples in model modals (Lora/Checkpoint)
 */
import { showToast } from '../../../utils/uiHelpers.js';
import { state } from '../../../state/index.js';
import { NSFW_LEVELS } from '../../../utils/constants.js';
import { 
    initLazyLoading,
    initNsfwBlurHandlers, 
    initMetadataPanelHandlers,
    initMediaControlHandlers,
    positionAllMediaControls
} from './MediaUtils.js';
import { generateMetadataPanel } from './MetadataPanel.js';
import { generateImageWrapper, generateVideoWrapper } from './MediaRenderers.js';

/**
 * Load example images asynchronously
 * @param {Array} images - Array of image objects (both regular and custom)
 * @param {string} modelHash - Model hash for fetching local files
 */
export async function loadExampleImages(images, modelHash) {
    try {
        const showcaseTab = document.getElementById('showcase-tab');
        if (!showcaseTab) return;
        
        // First fetch local example files
        let localFiles = [];

        try {
            const endpoint = '/api/example-image-files';
            const params = `model_hash=${modelHash}`;
            
            const response = await fetch(`${endpoint}?${params}`);
            const result = await response.json();
            
            if (result.success) {
                localFiles = result.files;
            }
        } catch (error) {
            console.error("Failed to get example files:", error);
        }
        
        // Then render with both remote images and local files
        showcaseTab.innerHTML = renderShowcaseContent(images, localFiles);
        
        // Re-initialize the showcase event listeners
        const carousel = showcaseTab.querySelector('.carousel');
        if (carousel && !carousel.classList.contains('collapsed')) {
            initShowcaseContent(carousel);
        }
        
        // Initialize the example import functionality
        initExampleImport(modelHash, showcaseTab);
    } catch (error) {
        console.error('Error loading example images:', error);
        const showcaseTab = document.getElementById('showcase-tab');
        if (showcaseTab) {
            showcaseTab.innerHTML = `
                <div class="error-message">
                    <i class="fas fa-exclamation-circle"></i>
                    Error loading example images
                </div>
            `;
        }
    }
}

/**
 * Render showcase content
 * @param {Array} images - Array of images/videos to show
 * @param {Array} exampleFiles - Local example files
 * @param {boolean} startExpanded - Whether to start in expanded state
 * @returns {string} HTML content
 */
export function renderShowcaseContent(images, exampleFiles = [], startExpanded = false) {
    if (!images?.length) {
        // Show empty state with import interface
        return renderImportInterface(true);
    }
    
    // Filter images based on SFW setting
    const showOnlySFW = state.settings.show_only_sfw;
    let filteredImages = images;
    let hiddenCount = 0;
    
    if (showOnlySFW) {
        filteredImages = images.filter(img => {
            const nsfwLevel = img.nsfwLevel !== undefined ? img.nsfwLevel : 0;
            const isSfw = nsfwLevel < NSFW_LEVELS.R;
            if (!isSfw) hiddenCount++;
            return isSfw;
        });
    }
    
    // Show message if no images are available after filtering
    if (filteredImages.length === 0) {
        return `
            <div class="no-examples">
                <p>All example images are filtered due to NSFW content settings</p>
                <p class="nsfw-filter-info">Your settings are currently set to show only safe-for-work content</p>
                <p>You can change this in Settings <i class="fas fa-cog"></i></p>
            </div>
        `;
    }
    
    // Show hidden content notification if applicable
    const hiddenNotification = hiddenCount > 0 ? 
        `<div class="nsfw-filter-notification">
            <i class="fas fa-eye-slash"></i> ${hiddenCount} ${hiddenCount === 1 ? 'image' : 'images'} hidden due to SFW-only setting
        </div>` : '';
    
    return `
        <div class="scroll-indicator">
            <i class="fas fa-chevron-${startExpanded ? 'up' : 'down'}"></i>
            <span>Scroll or click to ${startExpanded ? 'hide' : 'show'} ${filteredImages.length} examples</span>
        </div>
        <div class="carousel ${startExpanded ? '' : 'collapsed'}">
            ${hiddenNotification}
            <div class="carousel-container">
                ${filteredImages.map((img, index) => renderMediaItem(img, index, exampleFiles)).join('')}
            </div>
            
            ${renderImportInterface(false)}
        </div>
    `;
}

/**
 * Render a single media item (image or video)
 * @param {Object} img - Image/video metadata
 * @param {number} index - Index in the array
 * @param {Array} exampleFiles - Local files
 * @returns {string} HTML for the media item
 */
function renderMediaItem(img, index, exampleFiles) {
    // Find matching file in our list of actual files
    let localFile = findLocalFile(img, index, exampleFiles);
    
    const remoteUrl = img.url || '';
    const localUrl = localFile ? localFile.path : '';
    const isVideo = localFile ? localFile.is_video : 
                  remoteUrl.endsWith('.mp4') || remoteUrl.endsWith('.webm');
    
    // Calculate appropriate aspect ratio
    const aspectRatio = (img.height / img.width) * 100;
    const containerWidth = 800; // modal content maximum width
    const minHeightPercent = 40; 
    const maxHeightPercent = (window.innerHeight * 0.6 / containerWidth) * 100;
    const heightPercent = Math.max(
        minHeightPercent,
        Math.min(maxHeightPercent, aspectRatio)
    );
    
    // Check if media should be blurred
    const nsfwLevel = img.nsfwLevel !== undefined ? img.nsfwLevel : 0;
    const shouldBlur = state.settings.blurMatureContent && nsfwLevel > NSFW_LEVELS.PG13;
    
    // Determine NSFW warning text based on level
    let nsfwText = "Mature Content";
    if (nsfwLevel >= NSFW_LEVELS.XXX) {
        nsfwText = "XXX-rated Content";
    } else if (nsfwLevel >= NSFW_LEVELS.X) {
        nsfwText = "X-rated Content";
    } else if (nsfwLevel >= NSFW_LEVELS.R) {
        nsfwText = "R-rated Content";
    }
    
    // Extract metadata from the image
    const meta = img.meta || {};
    const prompt = meta.prompt || '';
    const negativePrompt = meta.negative_prompt || meta.negativePrompt || '';
    const size = meta.Size || `${img.width}x${img.height}`;
    const seed = meta.seed || '';
    const model = meta.Model || '';
    const steps = meta.steps || '';
    const sampler = meta.sampler || '';
    const cfgScale = meta.cfgScale || '';
    const clipSkip = meta.clipSkip || '';
    
    // Check if we have any meaningful generation parameters
    const hasParams = seed || model || steps || sampler || cfgScale || clipSkip;
    const hasPrompts = prompt || negativePrompt;
    
    // Create metadata panel content
    const metadataPanel = generateMetadataPanel(
        hasParams, hasPrompts, 
        prompt, negativePrompt, 
        size, seed, model, steps, sampler, cfgScale, clipSkip
    );
    
    // Determine if this is a custom image (has id property)
    const isCustomImage = Boolean(img.id);
    
    // Create the media control buttons HTML
    const mediaControlsHtml = `
        <div class="media-controls">
            <button class="media-control-btn set-preview-btn" title="Set as preview">
                <i class="fas fa-image"></i>
            </button>
            <button class="media-control-btn example-delete-btn ${!isCustomImage ? 'disabled' : ''}" 
                    title="${isCustomImage ? 'Delete this example' : 'Only custom images can be deleted'}" 
                    data-short-id="${img.id || ''}" 
                    ${!isCustomImage ? 'disabled' : ''}>
                <i class="fas fa-trash-alt"></i>
                <i class="fas fa-check confirm-icon"></i>
            </button>
        </div>
    `;
    
    // Generate the appropriate wrapper based on media type
    if (isVideo) {
        return generateVideoWrapper(
            img, heightPercent, shouldBlur, nsfwText, metadataPanel, 
            localUrl, remoteUrl, mediaControlsHtml
        );
    }
    
    return generateImageWrapper(
        img, heightPercent, shouldBlur, nsfwText, metadataPanel, 
        localUrl, remoteUrl, mediaControlsHtml
    );
}

/**
 * Find the matching local file for an image
 * @param {Object} img - Image metadata
 * @param {number} index - Image index
 * @param {Array} exampleFiles - Array of local files
 * @returns {Object|null} Matching local file or null
 */
function findLocalFile(img, index, exampleFiles) {
    if (!exampleFiles || exampleFiles.length === 0) return null;
    
    let localFile = null;
    
    if (img.id) {
        // This is a custom image, find by custom_<id>
        const customPrefix = `custom_${img.id}`;
        localFile = exampleFiles.find(file => file.name.startsWith(customPrefix));
    } else {
        // This is a regular image from civitai, find by index
        localFile = exampleFiles.find(file => {
            const match = file.name.match(/image_(\d+)\./);
            return match && parseInt(match[1]) === index;
        });
    }
    
    return localFile;
}

/**
 * Render the import interface for example images
 * @param {boolean} isEmpty - Whether there are no existing examples
 * @returns {string} HTML content for import interface
 */
function renderImportInterface(isEmpty) {
    return `
        <div class="example-import-area ${isEmpty ? 'empty' : ''}">
            <div class="import-container" id="exampleImportContainer">
                <div class="import-placeholder">
                    <i class="fas fa-cloud-upload-alt"></i>
                    <h3>${isEmpty ? 'No example images available' : 'Add more examples'}</h3>
                    <p>Drag & drop images or videos here</p>
                    <p class="sub-text">or</p>
                    <button class="select-files-btn" id="selectExampleFilesBtn">
                        <i class="fas fa-folder-open"></i> Select Files
                    </button>
                    <p class="import-formats">Supported formats: jpg, png, gif, webp, mp4, webm</p>
                </div>
                <input type="file" id="exampleFilesInput" multiple accept="image/*,video/mp4,video/webm" style="display: none;">
                <div class="import-progress-container" style="display: none;">
                    <div class="import-progress">
                        <div class="progress-bar"></div>
                    </div>
                    <span class="progress-text">Importing files...</span>
                </div>
            </div>
        </div>
    `;
}

/**
 * Initialize the example import functionality
 * @param {string} modelHash - The SHA256 hash of the model
 * @param {Element} container - The container element for the import area
 */
export function initExampleImport(modelHash, container) {
    if (!container) return;
    
    const importContainer = container.querySelector('#exampleImportContainer');
    const fileInput = container.querySelector('#exampleFilesInput');
    const selectFilesBtn = container.querySelector('#selectExampleFilesBtn');
    
    // Set up file selection button
    if (selectFilesBtn) {
        selectFilesBtn.addEventListener('click', () => {
            fileInput.click();
        });
    }
    
    // Handle file selection
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleImportFiles(Array.from(e.target.files), modelHash, importContainer);
            }
        });
    }
    
    // Set up drag and drop
    if (importContainer) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            importContainer.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        // Highlight drop area on drag over
        ['dragenter', 'dragover'].forEach(eventName => {
            importContainer.addEventListener(eventName, () => {
                importContainer.classList.add('highlight');
            }, false);
        });
        
        // Remove highlight on drag leave
        ['dragleave', 'drop'].forEach(eventName => {
            importContainer.addEventListener(eventName, () => {
                importContainer.classList.remove('highlight');
            }, false);
        });
        
        // Handle dropped files
        importContainer.addEventListener('drop', (e) => {
            const files = Array.from(e.dataTransfer.files);
            handleImportFiles(files, modelHash, importContainer);
        }, false);
    }
}

/**
 * Handle the file import process
 * @param {File[]} files - Array of files to import
 * @param {string} modelHash - The SHA256 hash of the model
 * @param {Element} importContainer - The container element for import UI
 */
async function handleImportFiles(files, modelHash, importContainer) {
    // Filter for supported file types
    const supportedImages = ['.jpg', '.jpeg', '.png', '.gif', '.webp'];
    const supportedVideos = ['.mp4', '.webm'];
    const supportedExtensions = [...supportedImages, ...supportedVideos];
    
    const validFiles = files.filter(file => {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        return supportedExtensions.includes(ext);
    });
    
    if (validFiles.length === 0) {
        alert('No supported files selected. Please select image or video files.');
        return;
    }
    
    try {
        // Use FormData to upload files
        const formData = new FormData();
        formData.append('model_hash', modelHash);
        
        validFiles.forEach(file => {
            formData.append('files', file);
        });
        
        // Call API to import files
        const response = await fetch('/api/import-example-images', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Failed to import example files');
        }
        
        // Get updated local files
        const updatedFilesResponse = await fetch(`/api/example-image-files?model_hash=${modelHash}`);
        const updatedFilesResult = await updatedFilesResponse.json();
        
        if (!updatedFilesResult.success) {
            throw new Error(updatedFilesResult.error || 'Failed to get updated file list');
        }
        
        // Re-render the showcase content
        const showcaseTab = document.getElementById('showcase-tab');
        if (showcaseTab) {
            // Get the updated images from the result
            const regularImages = result.regular_images || [];
            const customImages = result.custom_images || [];
            // Combine both arrays for rendering
            const allImages = [...regularImages, ...customImages];
            showcaseTab.innerHTML = renderShowcaseContent(allImages, updatedFilesResult.files, true);
            
            // Re-initialize showcase functionality
            const carousel = showcaseTab.querySelector('.carousel');
            if (carousel && !carousel.classList.contains('collapsed')) {
                initShowcaseContent(carousel);
            }
            
            // Initialize the import UI for the new content
            initExampleImport(modelHash, showcaseTab);
            
            showToast('Example images imported successfully', 'success');
            
            // Update VirtualScroller if available
            if (state.virtualScroller && result.model_file_path) {
                // Create an update object with only the necessary properties
                const updateData = {
                    civitai: {
                        images: regularImages,
                        customImages: customImages
                    }
                };
                
                // Update the item in the virtual scroller
                state.virtualScroller.updateSingleItem(result.model_file_path, updateData);
            }
        }
    } catch (error) {
        console.error('Error importing examples:', error);
        showToast(`Failed to import example images: ${error.message}`, 'error');
    }
}

/**
 * Toggle showcase expansion
 * @param {HTMLElement} element - The scroll indicator element
 */
export function toggleShowcase(element) {
    const carousel = element.nextElementSibling;
    const isCollapsed = carousel.classList.contains('collapsed');
    const indicator = element.querySelector('span');
    const icon = element.querySelector('i');
    
    carousel.classList.toggle('collapsed');
    
    if (isCollapsed) {
        const count = carousel.querySelectorAll('.media-wrapper').length;
        indicator.textContent = `Scroll or click to hide examples`;
        icon.classList.replace('fa-chevron-down', 'fa-chevron-up');
        initShowcaseContent(carousel);
    } else {
        const count = carousel.querySelectorAll('.media-wrapper').length;
        indicator.textContent = `Scroll or click to show ${count} examples`;
        icon.classList.replace('fa-chevron-up', 'fa-chevron-down');
        
        // Make sure any open metadata panels get closed
        const carouselContainer = carousel.querySelector('.carousel-container');
        if (carouselContainer) {
            carouselContainer.style.height = '0';
            setTimeout(() => {
                carouselContainer.style.height = '';
            }, 300);
        }
    }
}

/**
 * Initialize all showcase content interactions
 * @param {HTMLElement} carousel - The carousel element
 */
export function initShowcaseContent(carousel) {
    if (!carousel) return;
    
    initLazyLoading(carousel);
    initNsfwBlurHandlers(carousel);
    initMetadataPanelHandlers(carousel);
    initMediaControlHandlers(carousel);
    positionAllMediaControls(carousel);

    // Bind scroll-indicator click to toggleShowcase
    const scrollIndicator = carousel.previousElementSibling;
    if (scrollIndicator && scrollIndicator.classList.contains('scroll-indicator')) {
        // Remove previous click listeners to avoid duplicates
        scrollIndicator.onclick = null;
        scrollIndicator.removeEventListener('click', scrollIndicator._toggleShowcaseHandler);
        scrollIndicator._toggleShowcaseHandler = () => toggleShowcase(scrollIndicator);
        scrollIndicator.addEventListener('click', scrollIndicator._toggleShowcaseHandler);
    }
    
    // Add window resize handler
    const resizeHandler = () => positionAllMediaControls(carousel);
    window.removeEventListener('resize', resizeHandler);
    window.addEventListener('resize', resizeHandler);
    
    // Handle images loading which might change dimensions
    const mediaElements = carousel.querySelectorAll('img, video');
    mediaElements.forEach(media => {
        media.addEventListener('load', () => positionAllMediaControls(carousel));
        if (media.tagName === 'VIDEO') {
            media.addEventListener('loadedmetadata', () => positionAllMediaControls(carousel));
        }
    });
}

/**
 * Scroll to top of modal content
 * @param {HTMLElement} button - Back to top button
 */
export function scrollToTop(button) {
    const modalContent = button.closest('.modal-content');
    if (modalContent) {
        modalContent.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    }
}

/**
 * Set up showcase scroll functionality
 * @param {string} modalId - ID of the modal element
 */
export function setupShowcaseScroll(modalId) {
    // Listen for wheel events
    document.addEventListener('wheel', (event) => {
        const modalContent = document.querySelector(`#${modalId} .modal-content`);
        if (!modalContent) return;
        
        const showcase = modalContent.querySelector('.showcase-section');
        if (!showcase) return;
        
        const carousel = showcase.querySelector('.carousel');
        const scrollIndicator = showcase.querySelector('.scroll-indicator');
        
        if (carousel?.classList.contains('collapsed') && event.deltaY > 0) {
            const isNearBottom = modalContent.scrollHeight - modalContent.scrollTop - modalContent.clientHeight < 100;
            
            if (isNearBottom) {
                toggleShowcase(scrollIndicator);
                event.preventDefault();
            }
        }
    }, { passive: false });
    
    // Use MutationObserver to set up back-to-top button when modal content is added
    const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            if (mutation.type === 'childList' && mutation.addedNodes.length) {
                const modal = document.getElementById(modalId);
                if (modal && modal.querySelector('.modal-content')) {
                    setupBackToTopButton(modal.querySelector('.modal-content'));
                }
            }
        }
    });
    
    observer.observe(document.body, { childList: true, subtree: true });
    
    // Try to set up the button immediately in case the modal is already open
    const modalContent = document.querySelector(`#${modalId} .modal-content`);
    if (modalContent) {
        setupBackToTopButton(modalContent);
    }
}

/**
 * Set up back-to-top button
 * @param {HTMLElement} modalContent - Modal content element
 */
function setupBackToTopButton(modalContent) {
    // Remove any existing scroll listeners to avoid duplicates
    modalContent.onscroll = null;
    
    // Add new scroll listener
    modalContent.addEventListener('scroll', () => {
        const backToTopBtn = modalContent.querySelector('.back-to-top');
        if (backToTopBtn) {
            if (modalContent.scrollTop > 300) {
                backToTopBtn.classList.add('visible');
            } else {
                backToTopBtn.classList.remove('visible');
            }
        }
    });
    
    // Trigger a scroll event to check initial position
    modalContent.dispatchEvent(new Event('scroll'));
}