/**
 * MediaUtils.js
 * Media-specific utility functions for showcase components
 * (Moved from uiHelpers.js to better organize code)
 */
import { showToast, copyToClipboard } from '../../../utils/uiHelpers.js';
import { state } from '../../../state/index.js';
import { getModelApiClient } from '../../../api/modelApiFactory.js';

/**
 * Try to load local image first, fall back to remote if local fails
 * @param {HTMLImageElement} imgElement - The image element to update
 * @param {Object} urls - Object with local URLs {primary, fallback} and remote URL
 */
export function tryLocalImageOrFallbackToRemote(imgElement, urls) {
    const { primary: localUrl, fallback: fallbackUrl } = urls.local || {};
    const remoteUrl = urls.remote;
    
    // If no local options, use remote directly
    if (!localUrl) {
        imgElement.src = remoteUrl;
        return;
    }
    
    // Try primary local URL
    const testImg = new Image();
    testImg.onload = () => {
        // Primary local image loaded successfully
        imgElement.src = localUrl;
    };
    testImg.onerror = () => {
        // Try fallback URL if available
        if (fallbackUrl) {
            const fallbackImg = new Image();
            fallbackImg.onload = () => {
                imgElement.src = fallbackUrl;
            };
            fallbackImg.onerror = () => {
                // Both local options failed, use remote
                imgElement.src = remoteUrl;
            };
            fallbackImg.src = fallbackUrl;
        } else {
            // No fallback, use remote
            imgElement.src = remoteUrl;
        }
    };
    testImg.src = localUrl;
}

/**
 * Try to load local video first, fall back to remote if local fails
 * @param {HTMLVideoElement} videoElement - The video element to update
 * @param {Object} urls - Object with local URLs {primary} and remote URL
 */
export function tryLocalVideoOrFallbackToRemote(videoElement, urls) {
    const { primary: localUrl } = urls.local || {};
    const remoteUrl = urls.remote;
    
    // Only try local if we have a local path
    if (localUrl) {
        // Try to fetch local file headers to see if it exists
        fetch(localUrl, { method: 'HEAD' })
            .then(response => {
                if (response.ok) {
                    // Local video exists, use it
                    videoElement.src = localUrl;
                    const source = videoElement.querySelector('source');
                    if (source) source.src = localUrl;
                } else {
                    // Local video doesn't exist, use remote
                    videoElement.src = remoteUrl;
                    const source = videoElement.querySelector('source');
                    if (source) source.src = remoteUrl;
                }
                videoElement.load();
            })
            .catch(() => {
                // Error fetching, use remote
                videoElement.src = remoteUrl;
                const source = videoElement.querySelector('source');
                if (source) source.src = remoteUrl;
                videoElement.load();
            });
    } else {
        // No local path, use remote directly
        videoElement.src = remoteUrl;
        const source = videoElement.querySelector('source');
        if (source) source.src = remoteUrl;
        videoElement.load();
    }
}

/**
 * Initialize lazy loading for images and videos in a container
 * @param {HTMLElement} container - The container with lazy-loadable elements
 */
export function initLazyLoading(container) {
    const lazyElements = container.querySelectorAll('.lazy');
    
    const lazyLoad = (element) => {
        // Get URLs from data attributes
        const localUrls = {
            primary: element.dataset.localSrc || null,
            fallback: element.dataset.localFallbackSrc || null
        };
        const remoteUrl = element.dataset.remoteSrc;
        
        const urls = {
            local: localUrls,
            remote: remoteUrl
        };
        
        // Check if element is a video or image
        if (element.tagName.toLowerCase() === 'video') {
            tryLocalVideoOrFallbackToRemote(element, urls);
        } else {
            tryLocalImageOrFallbackToRemote(element, urls);
        }
        
        element.classList.remove('lazy');
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                lazyLoad(entry.target);
                observer.unobserve(entry.target);
            }
        });
    });

    lazyElements.forEach(element => observer.observe(element));
}

/**
 * Get the actual rendered rectangle of a media element with object-fit: contain
 * @param {HTMLElement} mediaElement - The img or video element
 * @param {number} containerWidth - Width of the container
 * @param {number} containerHeight - Height of the container
 * @returns {Object} - Rect with left, top, right, bottom coordinates
 */
export function getRenderedMediaRect(mediaElement, containerWidth, containerHeight) {
    // Get natural dimensions of the media
    const naturalWidth = mediaElement.naturalWidth || mediaElement.videoWidth || mediaElement.clientWidth;
    const naturalHeight = mediaElement.naturalHeight || mediaElement.videoHeight || mediaElement.clientHeight;
    
    if (!naturalWidth || !naturalHeight) {
        // Fallback if dimensions cannot be determined
        return { left: 0, top: 0, right: containerWidth, bottom: containerHeight };
    }
    
    // Calculate aspect ratios
    const containerRatio = containerWidth / containerHeight;
    const mediaRatio = naturalWidth / naturalHeight;
    
    let renderedWidth, renderedHeight, left = 0, top = 0;
    
    // Apply object-fit: contain logic
    if (containerRatio > mediaRatio) {
        // Container is wider than media - will have empty space on sides
        renderedHeight = containerHeight;
        renderedWidth = renderedHeight * mediaRatio;
        left = (containerWidth - renderedWidth) / 2;
    } else {
        // Container is taller than media - will have empty space top/bottom
        renderedWidth = containerWidth;
        renderedHeight = renderedWidth / mediaRatio;
        top = (containerHeight - renderedHeight) / 2;
    }
    
    return {
        left,
        top,
        right: left + renderedWidth,
        bottom: top + renderedHeight
    };
}

/**
 * Initialize metadata panel interaction handlers
 * @param {HTMLElement} container - Container element with media wrappers
 */
export function initMetadataPanelHandlers(container) {
    const mediaWrappers = container.querySelectorAll('.media-wrapper');
    
    mediaWrappers.forEach(wrapper => {
        // Get the metadata panel and media element (img or video)
        const metadataPanel = wrapper.querySelector('.image-metadata-panel');
        const mediaControls = wrapper.querySelector('.media-controls');
        const mediaElement = wrapper.querySelector('img, video');
        
        if (!mediaElement) return;
        
        let isOverMetadataPanel = false;
        
        // Add event listeners to the wrapper for mouse tracking
        wrapper.addEventListener('mousemove', (e) => {
            // Get mouse position relative to wrapper
            const rect = wrapper.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            // Get the actual displayed dimensions of the media element
            const mediaRect = getRenderedMediaRect(mediaElement, rect.width, rect.height);
            
            // Check if mouse is over the actual media content
            const isOverMedia = (
                mouseX >= mediaRect.left && 
                mouseX <= mediaRect.right && 
                mouseY >= mediaRect.top && 
                mouseY <= mediaRect.bottom
            );
            
            // Show metadata panel and controls when over media content or metadata panel itself
            if (isOverMedia || isOverMetadataPanel) {
                if (metadataPanel) metadataPanel.classList.add('visible');
                if (mediaControls) mediaControls.classList.add('visible');
            } else {
                if (metadataPanel) metadataPanel.classList.remove('visible');
                if (mediaControls) mediaControls.classList.remove('visible');
            }
        });
        
        wrapper.addEventListener('mouseleave', () => {
            if (!isOverMetadataPanel) {
                if (metadataPanel) metadataPanel.classList.remove('visible');
                if (mediaControls) mediaControls.classList.remove('visible');
            }
        });
        
        // Add mouse enter/leave events for the metadata panel itself
        if (metadataPanel) {
            metadataPanel.addEventListener('mouseenter', () => {
                isOverMetadataPanel = true;
                metadataPanel.classList.add('visible');
                if (mediaControls) mediaControls.classList.add('visible');
            });
            
            metadataPanel.addEventListener('mouseleave', () => {
                isOverMetadataPanel = false;
                // Only hide if mouse is not over the media
                const rect = wrapper.getBoundingClientRect();
                const mediaRect = getRenderedMediaRect(mediaElement, rect.width, rect.height);
                const mouseX = event.clientX - rect.left;
                const mouseY = event.clientY - rect.top;
                
                const isOverMedia = (
                    mouseX >= mediaRect.left && 
                    mouseX <= mediaRect.right && 
                    mouseY >= mediaRect.top && 
                    mouseY <= mediaRect.bottom
                );
                
                if (!isOverMedia) {
                    metadataPanel.classList.remove('visible');
                    if (mediaControls) mediaControls.classList.remove('visible');
                }
            });
            
            // Prevent events from bubbling
            metadataPanel.addEventListener('click', (e) => {
                e.stopPropagation();
            });
            
            // Handle copy prompt buttons
            const copyBtns = metadataPanel.querySelectorAll('.copy-prompt-btn');
            copyBtns.forEach(copyBtn => {
                const promptIndex = copyBtn.dataset.promptIndex;
                const promptElement = wrapper.querySelector(`#prompt-${promptIndex}`);
                
                copyBtn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    
                    if (!promptElement) return;
                    
                    try {
                        await copyToClipboard(promptElement.textContent, 'Prompt copied to clipboard');
                    } catch (err) {
                        console.error('Copy failed:', err);
                        showToast('Copy failed', 'error');
                    }
                });
            });
            
            // Prevent panel scroll from causing modal scroll
            metadataPanel.addEventListener('wheel', (e) => {
                const isAtTop = metadataPanel.scrollTop === 0;
                const isAtBottom = metadataPanel.scrollHeight - metadataPanel.scrollTop === metadataPanel.clientHeight;
                
                // Only prevent default if scrolling would cause the panel to scroll
                if ((e.deltaY < 0 && !isAtTop) || (e.deltaY > 0 && !isAtBottom)) {
                    e.stopPropagation();
                }
            }, { passive: true });
        }
    });
}

/**
 * Initialize NSFW content blur toggle handlers
 * @param {HTMLElement} container - Container element with media wrappers
 */
export function initNsfwBlurHandlers(container) {
    // Handle toggle blur buttons
    const toggleButtons = container.querySelectorAll('.toggle-blur-btn');
    toggleButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const wrapper = btn.closest('.media-wrapper');
            const media = wrapper.querySelector('img, video');
            const isBlurred = media.classList.toggle('blurred');
            const icon = btn.querySelector('i');
            
            // Update the icon based on blur state
            if (isBlurred) {
                icon.className = 'fas fa-eye';
            } else {
                icon.className = 'fas fa-eye-slash';
            }
            
            // Toggle the overlay visibility
            const overlay = wrapper.querySelector('.nsfw-overlay');
            if (overlay) {
                overlay.style.display = isBlurred ? 'flex' : 'none';
            }
        });
    });
    
    // Handle "Show" buttons in overlays
    const showButtons = container.querySelectorAll('.show-content-btn');
    showButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const wrapper = btn.closest('.media-wrapper');
            const media = wrapper.querySelector('img, video');
            media.classList.remove('blurred');
            
            // Update the toggle button icon
            const toggleBtn = wrapper.querySelector('.toggle-blur-btn');
            if (toggleBtn) {
                toggleBtn.querySelector('i').className = 'fas fa-eye-slash';
            }
            
            // Hide the overlay
            const overlay = wrapper.querySelector('.nsfw-overlay');
            if (overlay) {
                overlay.style.display = 'none';
            }
        });
    });
}

/**
 * Initialize media control buttons event handlers
 * @param {HTMLElement} container - Container with media wrappers
 */
export function initMediaControlHandlers(container) {
    // Find all delete buttons in the container
    const deleteButtons = container.querySelectorAll('.example-delete-btn');
    
    deleteButtons.forEach(btn => {
        // Set initial state
        btn.dataset.state = 'initial';
        
        btn.addEventListener('click', async function(e) {
            e.stopPropagation();
            
            // Explicitly check for disabled state
            if (this.classList.contains('disabled')) {
                return; // Don't do anything if button is disabled
            }
            
            const shortId = this.dataset.shortId;
            const btnState = this.dataset.state;
            
            if (!shortId) return;
            
            // Handle two-step confirmation
            if (btnState === 'initial') {
                // First click: show confirmation state
                this.dataset.state = 'confirm';
                this.classList.add('confirm');
                this.title = 'Click again to confirm deletion';
                
                // Auto-reset after 3 seconds
                setTimeout(() => {
                    if (this.dataset.state === 'confirm') {
                        this.dataset.state = 'initial';
                        this.classList.remove('confirm');
                        this.title = 'Delete this example';
                    }
                }, 3000);
                
                return;
            }
            
            // Second click within 3 seconds: proceed with deletion
            if (btnState === 'confirm') {
                this.disabled = true;
                this.classList.remove('confirm');
                this.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
                
                // Get model hash from URL or data attribute
                const mediaWrapper = this.closest('.media-wrapper');
                const modelHashAttr = document.querySelector('.showcase-section')?.dataset;
                const modelHash = modelHashAttr?.modelHash;
                
                try {
                    // Call the API to delete the custom example
                    const response = await fetch('/api/delete-example-image', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            model_hash: modelHash,
                            short_id: shortId
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        // Success: remove the media wrapper from the DOM
                        mediaWrapper.style.opacity = '0';
                        mediaWrapper.style.height = '0';
                        mediaWrapper.style.transition = 'opacity 0.3s ease, height 0.3s ease 0.3s';
                        
                        setTimeout(() => {
                            mediaWrapper.remove();
                        }, 600);
                        
                        // Show success toast
                        showToast('Example image deleted', 'success');

                        // Create an update object with only the necessary properties
                        const updateData = {
                            civitai: {
                                customImages: result.custom_images || []
                            }
                        };
                        
                        // Update the item in the virtual scroller
                        state.virtualScroller.updateSingleItem(result.model_file_path, updateData);
                    } else {
                        // Show error message
                        showToast(result.error || 'Failed to delete example image', 'error');
                        
                        // Reset button state
                        this.disabled = false;
                        this.dataset.state = 'initial';
                        this.classList.remove('confirm');
                        this.innerHTML = '<i class="fas fa-trash-alt"></i>';
                        this.title = 'Delete this example';
                    }
                } catch (error) {
                    console.error('Error deleting example image:', error);
                    showToast('Failed to delete example image', 'error');
                    
                    // Reset button state
                    this.disabled = false;
                    this.dataset.state = 'initial';
                    this.classList.remove('confirm');
                    this.innerHTML = '<i class="fas fa-trash-alt"></i>';
                    this.title = 'Delete this example';
                }
            }
        });
    });
    
    // Initialize set preview buttons
    initSetPreviewHandlers(container);
    
    // Media control visibility is now handled in initMetadataPanelHandlers
    // Any click handlers or other functionality can still be added here
}

/**
 * Initialize set preview button handlers
 * @param {HTMLElement} container - Container with media wrappers
 */
function initSetPreviewHandlers(container) {
    const previewButtons = container.querySelectorAll('.set-preview-btn');
    const modelType = state.currentPageType == 'loras' ? 'lora' : 'checkpoint';
    
    previewButtons.forEach(btn => {
        btn.addEventListener('click', async function(e) {
            e.stopPropagation();
            
            // Show loading state
            this.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            this.disabled = true;
            
            try {
                // Get the model file path from showcase section data attribute
                const showcaseSection = document.querySelector('.showcase-section');
                const modelHash = showcaseSection?.dataset.modelHash;
                const modelFilePath = showcaseSection?.dataset.filepath;
                
                if (!modelFilePath) {
                    throw new Error('Could not determine model file path');
                }
                
                // Get the media wrapper and media element
                const mediaWrapper = this.closest('.media-wrapper');
                const mediaElement = mediaWrapper.querySelector('img, video');
                
                if (!mediaElement) {
                    throw new Error('Media element not found');
                }
                
                // Get NSFW level from the wrapper or media element
                const nsfwLevel = parseInt(mediaWrapper.dataset.nsfwLevel || mediaElement.dataset.nsfwLevel || '0', 10);
                
                // Get local file path if available
                const useLocalFile = mediaElement.dataset.localSrc && !mediaElement.dataset.localSrc.includes('undefined');
                const apiClient = getModelApiClient();
                
                if (useLocalFile) {
                    // We have a local file, use it directly
                    const response = await fetch(mediaElement.dataset.localSrc);
                    const blob = await response.blob();
                    const file = new File([blob], 'preview.jpg', { type: blob.type });
                    
                    // Use the existing baseModelApi uploadPreview method with nsfw level
                    await apiClient.uploadPreview(modelFilePath, file, modelType, nsfwLevel);
                } else {
                    // We need to download the remote file first
                    const response = await fetch(mediaElement.src);
                    const blob = await response.blob();
                    const file = new File([blob], 'preview.jpg', { type: blob.type });
                    
                    // Use the existing baseModelApi uploadPreview method with nsfw level
                    await apiClient.uploadPreview(modelFilePath, file, modelType, nsfwLevel);
                }
            } catch (error) {
                console.error('Error setting preview:', error);
                showToast('Failed to set preview image', 'error');
            } finally {
                // Restore button state
                this.innerHTML = '<i class="fas fa-image"></i>';
                this.disabled = false;
            }
        });
    });
}

/**
 * Position media controls within the actual rendered media rectangle
 * @param {HTMLElement} mediaWrapper - The wrapper containing the media and controls
 */
export function positionMediaControlsInMediaRect(mediaWrapper) {
    const mediaElement = mediaWrapper.querySelector('img, video');
    const controlsElement = mediaWrapper.querySelector('.media-controls');
    
    if (!mediaElement || !controlsElement) return;
    
    // Get wrapper dimensions
    const wrapperRect = mediaWrapper.getBoundingClientRect();
    
    // Calculate the actual rendered media rectangle
    const mediaRect = getRenderedMediaRect(
        mediaElement, 
        wrapperRect.width, 
        wrapperRect.height
    );
    
    // Calculate the position for controls - place them inside the actual media area
    const padding = 8; // Padding from the edge of the media
    
    // Position at top-right inside the actual media rectangle
    controlsElement.style.top = `${mediaRect.top + padding}px`;
    controlsElement.style.right = `${wrapperRect.width - mediaRect.right + padding}px`;
    
    // Also position any toggle blur buttons in the same way but on the left
    const toggleBlurBtn = mediaWrapper.querySelector('.toggle-blur-btn');
    if (toggleBlurBtn) {
        toggleBlurBtn.style.top = `${mediaRect.top + padding}px`;
        toggleBlurBtn.style.left = `${mediaRect.left + padding}px`;
    }
}

/**
 * Position all media controls in a container
 * @param {HTMLElement} container - Container with media wrappers
 */
export function positionAllMediaControls(container) {
    const mediaWrappers = container.querySelectorAll('.media-wrapper');
    mediaWrappers.forEach(wrapper => {
        positionMediaControlsInMediaRect(wrapper);
    });
}